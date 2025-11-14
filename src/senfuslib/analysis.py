from functools import cache
from operator import attrgetter
from typing import Any, Sequence, Tuple, TypeVar, Union
import numpy as np
from scipy.stats import chi2
from dataclasses import dataclass, field

from .gaussian import MultiVarGauss
from .timesequence import TimeSequence
from .named_array import NamedArray

S = TypeVar('S', bound=np.ndarray)  # State type
M = TypeVar('M', bound=np.ndarray)  # Measurement type


@cache
def chi2_interval(alpha, dof):
    return chi2.interval(alpha, dof)


@cache
def chi2_mean(dof):
    return chi2.mean(dof)


@dataclass
class ConsistencyData:
    mahal_dist_tseq: TimeSequence[MultiVarGauss[S]]
    low_med_upp_tseq: TimeSequence[MultiVarGauss[Tuple[float, float, float]]]
    above_median: float
    in_interval: float
    alpha: float
    dofs: list[int]
    a: float
    adof: int
    aconf: Tuple[float, float]


@dataclass
class ConsistencyAnalysis:
    x_gts: TimeSequence[S]
    zs: TimeSequence[M]
    x_ests: TimeSequence[Union[MultiVarGauss[S], Any]]
    z_preds: TimeSequence[Union[MultiVarGauss[S], Any]]

    x_err_gauss: TimeSequence[MultiVarGauss[S]] = field(init=False)
    z_err_gauss: TimeSequence[MultiVarGauss[M]] = field(init=False)

    def __post_init__(self):
        def get_err_tseq(gts: TimeSequence, ests: TimeSequence):
            err_gauss_tseq = TimeSequence()
            for t, est in ests.items():
                if t not in gts:
                    continue
                gt = gts.get_t(t)
                if isinstance(est, MultiVarGauss):
                    err = MultiVarGauss(est.mean - gt, est.cov)
                else:
                    err = est.get_err_gauss(gt)
                err_gauss_tseq.insert(t, err)
            return err_gauss_tseq

        if self.x_gts is not None:
            self.x_err_gauss = get_err_tseq(self.x_gts, self.x_ests)
        self.z_err_gauss = get_err_tseq(self.zs, self.z_preds)

    def get_nis(self, indices: Sequence[Union[int, str]] = None,
                alpha=0.95) -> ConsistencyData:
        if indices == 'all':
            indices = None
        err_gauss_tseq = self._get_err(self.z_err_gauss, indices)
        return self._get_nisornees(err_gauss_tseq, alpha)

    def get_nees(self, indices: Sequence[Union[int, str]] = None,
                 alpha=0.95) -> ConsistencyData:
        if indices == 'all':
            indices = None
        err_gauss_tseq = self._get_err(self.x_err_gauss, indices)
        return self._get_nisornees(err_gauss_tseq, alpha)

    def get_x_err(self, indices: Sequence[Union[int, str]] = None):
        return self._get_err(self.x_err_gauss, indices)

    def get_z_err(self, indices: Sequence[Union[int, str]] = None):
        return self._get_err(self.z_err_gauss, indices)

    @staticmethod
    def _get_err(err_gauss_tseq: TimeSequence[MultiVarGauss[NamedArray]],
                 indices: Sequence[Union[int, str]]
                 ) -> TimeSequence[MultiVarGauss[NamedArray]]:
        if indices is None:
            # If no indices, use all dimensions of the state vector
            indices = np.arange(err_gauss_tseq.values[0].ndim)
        elif isinstance(indices, (int, str)):
            indices = [indices]

        def marginalize(err_gauss: MultiVarGauss[NamedArray]):
            # This inner function correctly resolves string names to integer indices
            def resolve_indices(idx_or_slice):
                if isinstance(idx_or_slice, str):
                    # Get the AtIndex object (e.g., AtIndex[2] or AtIndex[slice(8, None)])
                    resolved = attrgetter(idx_or_slice)(err_gauss.mean.indices)[0]
                else:
                    resolved = idx_or_slice

                # If it's a slice, convert it to a range of integers
                if isinstance(resolved, slice):
                    return np.arange(resolved.start, resolved.stop if resolved.stop is not None else err_gauss.ndim)
                return resolved

            # Build the final flat list of integer indices
            _indices = np.concatenate([np.atleast_1d(resolve_indices(i)) for i in indices])

            return err_gauss.get_marginalized(_indices)

        return err_gauss_tseq.map(marginalize)

    def _get_nisornees(self,
                       err_gauss_tseq: TimeSequence[MultiVarGauss[NamedArray]],
                       alpha: float,
                       ) -> ConsistencyData:

        def get_mahal(x: MultiVarGauss[NamedArray]):
            return x.mahalanobis_distance(np.zeros_like(x.mean))

        # If the sequence is empty, do nothing.
        if not err_gauss_tseq:
            # Return a dummy object or raise an error
            return None # Or some sensible default

        mahal_dist_tseq = err_gauss_tseq.map(get_mahal)

        dof = err_gauss_tseq.values[0].ndim
        dofs = [dof] * len(err_gauss_tseq)

        lower, upper = chi2_interval(alpha, dof)
        median = chi2.mean(dof)
        
        low_med_upp_tseq = TimeSequence()
        for t in err_gauss_tseq.times:
            low_med_upp_tseq.insert(t, (lower, median, upper))

        n = len(mahal_dist_tseq)
        mahal_dists = mahal_dist_tseq.values_as_array()

        above_median = np.sum(mahal_dists > median) / n
        in_interval = np.sum((mahal_dists > lower) & (mahal_dists < upper)) / n

        a = np.mean(mahal_dists)
        adof = n * dof
        a_lower, a_upper = chi2_interval(alpha, adof)
        aconf = (a_lower / n, a_upper / n)

        return ConsistencyData(mahal_dist_tseq, low_med_upp_tseq,
                               above_median, in_interval, alpha,
                               [dof], a, adof, aconf)
