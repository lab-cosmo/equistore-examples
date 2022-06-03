import numpy as np
import torch

import skcosmo.sample_selection
from equistore import Labels, TensorBlock, TensorMap

import utils.models.operations as ops

from .utils import detach, dot, normalize, power, structure_sum


def _select_support_points_for_block(block: TensorBlock, n_select: int):
    assert len(block.components) == 0

    fps = skcosmo.sample_selection.FPS(
        n_to_select=n_select,
        # stop before selecting identical points
        score_threshold=1e-12,
        score_threshold_type="relative",
    )

    array = block.values
    if isinstance(array, torch.Tensor):
        array = array.detach()
    fps.fit_transform(array)

    return TensorBlock(
        values=array[fps.selected_idx_],
        samples=block.samples[fps.selected_idx_],
        components=block.components,
        properties=block.properties,
    )


def select_support_points(ps, n_select):
    if isinstance(n_select, int):
        block = _select_support_points_for_block(ps.block(), n_select)

        return TensorMap(Labels.single(), [block])

    else:
        blocks = []
        for key, value in n_select.items():
            block = _select_support_points_for_block(
                ps.block(**key.as_dict()),
                n_select=value,
            )
            blocks.append(block)

        return TensorMap(ps.keys, blocks)


class CombinedModel:
    def __init__(
        self, do_normalize, support_points, zeta, regularizer, ratios, jitter=1e-13
    ):
        self.zeta = zeta
        self.jitter = jitter
        self.regularizer = regularizer
        self.ratios = ratios

        assert self.ratios[0] == 1
        assert len(self.ratios) == 2
        self.support_points = normalize(detach(support_points))

        self.do_normalize = do_normalize

        self.weights = None
        self.baseline = None

    def fit(self, ps, linear_features, energies, forces=None):
        # Create the kernels from the power spectrum
        k_mm = self._compute_kernel(self.support_points)
        K_MM = []
        for _, k_mm_block in k_mm:
            assert len(k_mm_block.components) == 0
            K_MM.append(k_mm_block.values)

        K_MM = ops.block_diag(*K_MM)
        K_MM[np.diag_indices_from(K_MM)] += self.jitter

        k_nm = self._compute_kernel(ps)
        k_nm = structure_sum(k_nm, sum_properties=False)

        # TODO: make sure this create an array in the same order as
        # `scipy.linalg.block_diag` above
        if len(k_nm.keys.names) != 0:
            names = list(k_nm.keys.names)
            k_nm.keys_to_properties(names)

        k_nm = k_nm.block()
        assert len(k_nm.components) == 0
        K_NM = k_nm.values

        # Subtract the mean of the energies for the learning
        self.baseline = 0  # energies.mean()
        Y = energies.reshape(-1, 1) - self.baseline

        # The regularizor is subtracted at this point for easier
        # generalizations later on which would allow structure-wise
        # tuning of the regularization.
        delta = energies.std()
        structures = np.unique(k_nm.samples["structure"])
        n_atoms_per_structure = []
        for structure in structures:
            n_atoms = np.sum(k_nm.samples["structure"] == structure)
            n_atoms_per_structure.append(float(n_atoms))

        energy_regularizer = (
            ops.sqrt(ops.array_like(energies, n_atoms_per_structure))
            * self.regularizer[0]
            / delta
        )
        Y /= energy_regularizer[:, None]
        K_NM[:] /= energy_regularizer[:, None]

        # Define the kernel and appropriately normalized target
        # properties.
        if forces is not None:
            # Define the kernel for the gradients.
            k_nm_gradient = k_nm.gradient("positions")
            k_nm_grad = k_nm_gradient.data.reshape(
                3 * k_nm_gradient.data.shape[0], k_nm_gradient.data.shape[2]
            )

            # As for the energies, the regularization is applied to
            # both the target property as well as the K_NM part.
            forces_regularizer = self.regularizer[1] / delta
            k_nm_grad[:] /= forces_regularizer
            energy_grad = -forces.reshape(-1, 1)
            energy_grad[:] /= forces_regularizer

            # Stack the energy and force parts for both the features
            # and target properties
            Y = ops.vstack([Y, energy_grad])
            K_NM = ops.vstack([K_NM, k_nm_grad])

        ###
        # Linear Model part
        ###
        linear_features_per_structure = structure_sum(linear_features)

        # Initial checks normalizations
        if self.do_normalize:
            linear_features_per_structure = normalize(linear_features_per_structure)

        block = linear_features_per_structure.block()
        assert len(block.components) == 0

        # Normalize features as for the kernel part (see comments above)
        # Note that this is different from the "linear model" implementation
        # currently found in equistore
        X = block.values
        X[:] /= energy_regularizer[:, None]

        # Similarly to the sparse kernel part, merge the kernels
        if forces is not None:
            gradient = block.gradient("positions")
            X_grad = gradient.data.reshape(3 * len(gradient.samples), X.shape[1])
            X_grad[:] /= forces_regularizer

            X = ops.vstack([X, X_grad])

        ###
        # Combination Part
        ###
        X_combined = np.hstack([K_NM, self.ratios[1] * X])
        num_features = X.shape[1]
        Regularization_combined = ops.block_diag(K_MM, np.eye(num_features))

        # Solve linear system of equation to obtain least squares solution
        A_to_invert = X_combined.T @ X_combined + Regularization_combined
        Y = X_combined.T @ Y
        self.weights = ops.linalg_solve(A_to_invert, Y, detach=True)

        # Store the shapes of the arrays to recover which components
        # of the weight vector correspond to which model
        self.model_parameters = []
        self.model_parameters.append(K_NM.shape[1])
        self.model_parameters.append(X.shape[1])

    def predict(self, ps, linear_features, with_forces=False):
        if self.weights is None:
            raise Exception("call fit first")

        # Evaluate the sparse kernel part
        k_per_atom = self._compute_kernel(ps)
        k_per_structure = structure_sum(k_per_atom, sum_properties=False)
        # TODO: make sure this create an array in the same order as block_diag above
        if len(k_per_structure.keys.names) != 0:
            names = list(k_per_structure.keys.names)
            k_per_structure.keys_to_properties(names)

        assert len(k_per_structure.block().components) == 0
        kernel = k_per_structure.block().values

        n_kernel = self.model_parameters[0]
        energies_1 = kernel @ self.weights[:n_kernel]

        # Linear Model part
        linear_features_per_structure = structure_sum(linear_features)
        if self.do_normalize:
            linear_features_per_structure = normalize(linear_features_per_structure)
        block = linear_features_per_structure.block()
        assert len(block.components) == 0
        X = block.values

        n_linear = self.model_parameters[1]
        energies_2 = self.ratios[1] * X @ self.weights[-n_linear:]

        # Combine
        energies = energies_1 + energies_2 + self.baseline

        # Forces
        if with_forces:
            # Kernel part
            kernel_gradient = k_per_structure.block().gradient("positions")
            forces_1 = -kernel_gradient.data @ self.weights[:n_kernel]

            # Linear part
            gradient = block.gradient("positions")
            X_grad = gradient.data.reshape(-1, 3, self.weights[-n_linear:].shape[0])
            forces_2 = -X_grad @ self.weights[-n_linear:]

            # Combine
            forces = forces_1 + forces_2
            forces = forces.reshape(-1, 3)
        else:
            forces = None

        return energies, forces

    def _compute_kernel(self, ps):
        return power(dot(ps, self.support_points, do_normalize=True), zeta=self.zeta)
