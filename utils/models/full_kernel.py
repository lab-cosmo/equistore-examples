import numpy as np

from .utils import invariant_block_to_2d_array
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

        k_nn_per_structure = self._compute_kernel(ps)

        kernel = invariant_block_to_2d_array(k_nn_per_structure.block())
        kernel[np.diag_indices_from(kernel)] += self.regularizer[0] / energies.std()

        self.offset = energies.mean()
        Y = energies - self.offset
        Y = Y.reshape(-1, 1)

        if forces is not None:
            kernel_samples, kernel_grad = k_nn_per_structure.block().gradient(
                "positions"
            )

            energy_grad = -forces.reshape(kernel_grad.shape[0], 1)
            Y = ops.vstack([Y, energy_grad])

            # TODO: regularize the kernel grad
            kernel_grad = kernel_grad.reshape(-1, kernel.shape[1])

            # TODO: this assume the atoms are in the same order in kernel_grad &
            # forces
            kernel = ops.vstack([kernel, kernel_grad])

        self.weights = ops.linalg_solve(kernel, Y, detach=True)

    def predict(self, ps, with_forces=False):
        if self.weights is None:
            raise Exception("call fit first")

        k_per_structure = self._compute_kernel(ps)

        block = k_per_structure.block()
        kernel = invariant_block_to_2d_array(block)

        energies = kernel @ self.weights + self.offset

        if with_forces:
            _, kernel_grad = block.gradient("positions")
            kernel_grad = kernel_grad.reshape(-1, self.weights.shape[0])

            forces = -kernel_grad @ self.weights
            forces = forces.reshape(-1, 3)
        else:
            forces = None

        return energies, forces

    def _compute_kernel(self, ps):
        return structure_sum(
            power(dot(ps, self.training_points), zeta=self.zeta), sum_features=True
        )
