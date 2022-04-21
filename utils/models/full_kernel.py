import numpy as np

from .utils import structure_sum, dot, power, detach, normalize
import utils.models.operations as ops


class FullKernelGap:
    def __init__(self, zeta, regularizer):
        self.zeta = zeta
        self.regularizer = regularizer

        self.training_points = None

        self.weights = None

    def fit(self, ps, energies, forces=None):
        self.training_points = normalize(detach(ps))

        k_nn_per_structure = self._compute_kernel(ps).block()

        kernel = k_nn_per_structure.values
        kernel[np.diag_indices_from(kernel)] += self.regularizer[0] / energies.std()

        self.offset = energies.mean()
        Y = energies - self.offset
        Y = Y.reshape(-1, 1)

        if forces is not None:
            raise ValueError(
                "Training full kernel model with forces is not implemented"
            )

        self.weights = ops.linalg_solve(kernel, Y, detach=True)

    def predict(self, ps, with_forces=False):
        if self.weights is None:
            raise Exception("call fit first")

        k_per_structure = self._compute_kernel(ps).block()
        kernel = k_per_structure.values
        energies = kernel @ self.weights + self.offset

        if with_forces:
            kernel_gradient = k_per_structure.gradient("positions")

            forces = -kernel_gradient.data @ self.weights
            forces = forces.reshape(-1, 3)
        else:
            forces = None

        return energies, forces

    def _compute_kernel(self, ps):
        return structure_sum(
            power(dot(ps, self.training_points), zeta=self.zeta), sum_properties=True
        )
