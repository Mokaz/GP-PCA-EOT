import numpy as np
from pathlib import Path
from src.utils.tools import ssa, cart2pol, pol2cart, fourier_transform

# # TODO: Find better name than theta_0
class Extent:
    def __init__(self, parameters, dAngle):
        # angles for polar representation
        angles = np.arange(-np.pi, np.pi, dAngle)
        self.angles = angles

        # radii for polar representation
        nAngles = angles.size

        if parameters["type"] == "box":
            L = parameters["L"]
            W = parameters["W"]

            theta_0 = np.arctan2(W, L)
            abs_angles = np.abs(ssa(angles))
            radii = np.zeros(abs_angles.shape)

            for idx, angle in enumerate(abs_angles):
                if angle <= theta_0:
                    radii[idx] = L / 2 / np.cos(angle)
                elif angle <= np.pi - theta_0:
                    radii[idx] = W / 2 / np.cos(np.pi/2 - angle)
                else:
                    radii[idx] = L / 2 / np.cos(np.pi - angle)

        elif parameters["type"] == "ellipse":
            L = parameters["L"]
            W = parameters["W"]

            x_interpol = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_interpol = y_interpol = (W / 2) * np.sqrt(1 - ((2 / L) * x_interpol)**2)

            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)

            radii = np.interp(np.abs(ssa(angles)), angles_interpol, r_interpol, period=2*np.pi)

        elif parameters["type"] == "box_elliptic_sides":
            L = parameters["L"]
            W = parameters["W"]
            S = parameters["S"]

            abs_angles = np.abs(ssa(angles))

            x_ellipse = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_ellipse = S/2 + ((W - S) / 2) * np.sqrt(1 - ((2 / L) * x_ellipse)**2)

            angles_interpol, r_interpol = cart2pol(x_ellipse, y_ellipse)

            theta_corner = np.arctan2(S, L)
            radii = np.zeros(abs_angles.shape)

            for idx, angle in enumerate(abs_angles):
                if angle <= theta_corner:
                    radii[idx] = (L / 2) / np.cos(angle)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                else:
                    radii[idx] = (L / 2) / np.cos(np.pi - angle)
        

        elif parameters["type"] == "box_parabolic_bow_and_stern":
            L = parameters["L"]
            W = parameters["W"]
            P = parameters["P"]

            abs_angles = np.abs(ssa(angles))

            x_interpol = np.linspace((L / 2) - P, L / 2, num=nAngles, endpoint=True)
            y_interpol = (-L**2 * W + 4*L*P*W) / (8 * P**2) + ((L*W - 2*P*W) * x_interpol) / (2*P**2) - (W * x_interpol**2) / (2 * P**2)  
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)

            theta_corner = np.arctan2(W, L - 2*P)
            radii = np.zeros(abs_angles.shape)

            for idx, angle in enumerate(abs_angles):
                if angle <= theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = (W / 2) / np.cos(angle - np.pi/2)
                else:
                    radii[idx] = np.interp(angle, np.pi - angles_interpol, r_interpol, period=2*np.pi)

        elif parameters["type"] == "elliptic_bow_and_stern":
            L = parameters["L"]
            W = parameters["W"]
            P = parameters["P"]

            abs_angles = np.abs(ssa(angles))

            y_ellipse = np.linspace(-W / 2, W / 2, num=nAngles, endpoint=True)
            x_ellipse = (L / 2) - P + P * np.sqrt(1 - ((2 / W) * y_ellipse)**2)

            angles_interpol, r_interpol = cart2pol(x_ellipse, y_ellipse)

            theta_corner = np.arctan2(W, L - 2*P)
            radii = np.zeros(abs_angles.shape)

            for idx, angle in enumerate(abs_angles):
                if angle <= theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = (W / 2) / np.cos(angle - np.pi/2)
                else:
                    radii[idx] = np.interp(angle, np.pi - angles_interpol, r_interpol, period=2*np.pi)

        else:
            raise Exception(f'No valid type: {parameters["type"]}')
        
        self.radii = radii.reshape((-1, ))
        cart_x, cart_y = pol2cart(self.angles, self.radii)
        cart_x_closed = np.concatenate((cart_x, cart_x[0]), axis=None).reshape((1, -1))
        cart_y_closed = np.concatenate((cart_y, cart_y[0]), axis=None).reshape((1, -1))
        self.cartesian = np.concatenate((cart_x_closed, cart_y_closed), axis=0)
        self.norm_cartesian = np.concatenate((cart_x / L, cart_y / W), axis=0).reshape((2, -1))

        # get cartesian normalized to in width and length
        norm_angles, norm_radii = cart2pol(cart_x / L, cart_y / W)

        self.norm_radii = norm_radii
        self.norm_angles = norm_angles

        self.L = L
        self.W = W

# class PCAExtentModel:
#     def __init__(self, shape: Extent, N_pca: int = 4):
#         # calculate PCA parameters for the extent

#         self.L = shape.L
#         self.W = shape.W

#         PCA_parameters = np.load(Path('data/input_parameters/FourierPCAParameters_scaled.npz'))
#         self.fourier_coeff_mean = PCA_parameters['mean']
#         self.M = PCA_parameters['eigenvectors'][:, :N_pca].real

#         # Find PCA scaling factor
#         self.scaling_factor = 1 / np.linalg.norm(self.M[:,0])

#         self.extent_fourier = fourier_transform(shape.norm_angles, shape.norm_radii, num_coeff=64, symmetry=True)
#         pca_params = self.M.T @ (self.extent_fourier - self.fourier_coeff_mean) * self.scaling_factor**2
        
#         self.pca_extent = self.fourier_coeff_mean + self.M @ pca_params
#         self.pca_params = pca_params.ravel()


