from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import tqdm
import logging
from scipy.stats import chi2


from states import StateCV, MeasPos
from models import ModelImm
from sensors import SensorPosClutter
from senfuslib import (MultiVarGauss, DynamicModel, SensorModel, TimeSequence,
                       GaussianMixture)
from senfuslib.debug_config import DEBUG
from solution import filter as filter_solu


@dataclass
class EKF:
    dynamic_model: DynamicModel[StateCV]
    sensor_model: SensorModel[MeasPos]

    def pred(self, x_est_prev: MultiVarGauss[StateCV], dt: float
             ) -> tuple[MultiVarGauss[StateCV], MultiVarGauss[MeasPos]]:
        """Perform one EKF prediction step."""
        x_est_pred = self.dynamic_model.pred_from_est(x_est_prev, dt)
        z_est_pred = self.sensor_model.pred_from_est(x_est_pred)
        return x_est_pred, z_est_pred

    def update(self, x_est_pred: MultiVarGauss[StateCV],
               z_est_pred: MultiVarGauss[MeasPos],
               z: MeasPos) -> MultiVarGauss[StateCV]:
        """Perform one EKF update step."""
        H_mat = self.sensor_model.H(x_est_pred.mean)
        P_mat = x_est_pred.cov
        S_mat = z_est_pred.cov

        kalman_gain = P_mat @ H_mat.T @ np.linalg.inv(S_mat)
        innovation = z - z_est_pred.mean

        state_upd_mean = x_est_pred.mean + kalman_gain @ innovation
        state_upd_cov = P_mat - kalman_gain @ H_mat @ P_mat

        x_est_upd = MultiVarGauss(state_upd_mean, state_upd_cov)

        return x_est_upd


@dataclass
class FilterPDA:
    dynamic_model: DynamicModel[StateCV]
    sensor_model: SensorPosClutter
    gate_prob: float

    ekf: EKF = field(init=False)
    gate: float = field(init=False)

    def __post_init__(self):
        self.ekf = EKF(self.dynamic_model, self.sensor_model.sensor)
        self.gate = chi2.ppf(self.gate_prob, 2)  # g**2 on page 120

    def gate_zs(self,
                z_est_pred: MultiVarGauss[MeasPos],
                zs: Sequence[MeasPos]
                ) -> tuple[set[int], Sequence[MeasPos]]:
        """Gate the measurements.
        That is, remove measurements with a probability of being clutter
        greater than self.gate_prob.

        Hint: (7.3.5), use mahalobis distance and the self.gate attribute."""
        gated_indices = set()
        zs_gated = []

        for i, z in enumerate(zs):
            condition = z_est_pred.mahalanobis_distance(z) <= self.gate
            if condition:
                zs_gated.append(z)
                gated_indices.add(i)

        return gated_indices, zs_gated

    def get_assoc_probs(self, z_est_pred: MultiVarGauss[MeasPos],
                        zs: Sequence[MeasPos]) -> np.ndarray:
        """Compute the association probabilities.
        P{a_k|Z_{1:k}} = assoc_probs[a_k]    (corollary 7.3.3)

        Hint: use some_gauss.pdf(something), rememeber to normalize"""
        lamb = self.sensor_model.clutter_density
        P_D = self.sensor_model.prob_detect

        # Calculate unnormalized likelihoods
        assoc_probs = np.empty(len(zs) + 1)
        assoc_probs[0] = (1 - P_D) * lamb
        for i, z in enumerate(zs):
            assoc_probs[i+1] = P_D * z_est_pred.pdf(z)

        # Normalize to get association probabilities
        assoc_probs = assoc_probs / np.sum(assoc_probs)

        return assoc_probs

    def get_estimates(self,
                      x_est_pred: MultiVarGauss[StateCV],
                      z_est_pred: MultiVarGauss[MeasPos],
                      zs_gated: Sequence[MeasPos]
                      ) -> Sequence[MultiVarGauss[StateCV]]:
        """Get the estimates corresponding to each association hypothesis.

        Compared to the book that is:
        hat{x}_k^{a_k} = x_ests[a_k].mean   (7.20)
        P_k^{a_k} = x_ests[a_k].cov         (7.21)

        Hint: Use self.ekf"""
        x_ests = []
        gauss_ak0 = x_est_pred
        x_ests.append(gauss_ak0)
        for z in zs_gated:
            x_est_upd = self.ekf.update(x_est_pred, z_est_pred, z)
            x_ests.append(x_est_upd)

        return x_ests

    def step(self,
             x_est_prev: MultiVarGauss[StateCV],
             zs: Sequence[MeasPos],
             dt: float) -> tuple[MultiVarGauss[StateCV],
                                 MultiVarGauss[StateCV],
                                 MultiVarGauss[MeasPos],
                                 set[int]]:
        """Perform one step of the PDAF."""

        x_est_pred, z_est_pred = self.ekf.pred(x_est_prev, dt)  # Hint: (7.16) and (7.17)
        gated_indices, zs_gated = self.gate_zs(z_est_pred, zs)  #  Hint: (7.3.5)
        assoc_probs = self.get_assoc_probs(z_est_pred, zs_gated)  #  Hint (Corollary 7.3.3)
        x_ests = self.get_estimates(x_est_pred, z_est_pred, zs_gated)  #  Hint: (7.20) and (7.21)
        x_est_upd_mixture = GaussianMixture(assoc_probs, x_ests)  #  Hint: (7.3.6)

        x_est_upd = x_est_upd_mixture.reduce()  #  Hint: (7.27) use reduce()

        return x_est_upd, x_est_pred, z_est_pred, gated_indices

    def run(self,
            x0_est: MultiVarGauss[StateCV],
            zs_tseq: TimeSequence[Sequence[MeasPos]]
            ) -> tuple[TimeSequence[MultiVarGauss[StateCV]],
                       TimeSequence[MultiVarGauss[StateCV]],
                       TimeSequence[MultiVarGauss[MeasPos]],
                       TimeSequence[set[int]]]:
        """Run the PDAF filter."""
        logging.info("Running PDAF filter")
        x_est_upds = TimeSequence()
        x_est_preds = TimeSequence()
        z_est_preds = TimeSequence()
        gated_indices_tseq = TimeSequence()
        x_est_upds.insert(0, x0_est)
        t_prev = 0
        for t, zs in tqdm.tqdm(zs_tseq.items(), total=len(zs_tseq)):
            t_prev, x_est_prev = x_est_upds[-1]
            dt = np.round(t-t_prev, 8)

            x_est_upd, x_est_pred, z_est_pred, gated_indices = self.step(
                x_est_prev, zs, dt)
            x_est_upds.insert(t, x_est_upd)
            x_est_preds.insert(t, x_est_pred)
            z_est_preds.insert(t, z_est_pred)
            gated_indices_tseq.insert(t, gated_indices)
        return x_est_upds, x_est_preds, z_est_preds, gated_indices_tseq
