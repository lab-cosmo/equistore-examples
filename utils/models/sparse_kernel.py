import numpy as np
import torch

import skcosmo.sample_selection

from aml_storage import Descriptor, Labels, Block

from .utils import invariant_block_to_2d_array, array_2d_to_invariant
from .utils import structure_sum, dot, power, normalize, detach
import utils.models.operations as ops


def _select_support_points_for_block(block: Block, n_select: int):
    fps = skcosmo.sample_selection.FPS(n_to_select=n_select)

    array = invariant_block_to_2d_array(block)
    if isinstance(array, torch.Tensor):
        array = array.detach()
    fps.fit_transform(array)

    return Block(
        values=array_2d_to_invariant(array)[fps.selected_idx_],
        samples=block.samples[fps.selected_idx_],
        components=Labels.single(),
        features=block.features,
    )


def select_support_points(ps, n_select):
    if isinstance(n_select, int):
        block = _select_support_points_for_block(ps.block(), n_select)

        return Descriptor(Labels.single(), [block])

    else:
        blocks = []
        for key, value in n_select.items():
            block = _select_support_points_for_block(
                ps.block(**key.as_dict()),
                n_select=value,
            )
            blocks.append(block)

        return Descriptor(ps.sparse, blocks)


class SparseKernelGap:
    def __init__(self, support_points, zeta, regularizer, jitter=1e-13):
        self.zeta = zeta
        self.jitter = jitter
        self.regularizer = regularizer
        self.support_points = normalize(detach(support_points))

        self.weights = None
        self.baseline = None

    def fit(self, ps, energies, forces=None):
        k_mm = self._compute_kernel(self.support_points)
        K_MM = []
        for _, k_mm_block in k_mm:
            K_MM.append(invariant_block_to_2d_array(k_mm_block))

        K_MM = ops.block_diag(*K_MM)
        K_MM[np.diag_indices_from(K_MM)] += self.jitter

        k_nm = self._compute_kernel(ps)
        k_nm = structure_sum(k_nm, sum_features=False)

        # TODO: make sure this create an array in the same order as
        # `scipy.linalg.block_diag` above
        if len(k_nm.sparse.names) != 0:
            names = list(k_nm.sparse.names)
            k_nm.sparse_to_features(names)

        k_nm = k_nm.block()
        K_NM = invariant_block_to_2d_array(k_nm)

        self.baseline = energies.mean()

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

        K_NM[:] /= energy_regularizer[:, None]

        Y = (energies.reshape(-1, 1) - self.baseline) / energy_regularizer[:, None]

        if forces is not None:
            _, k_nm_grad = k_nm.gradient("positions")
            k_nm_grad = k_nm_grad.reshape(k_nm_grad.shape[0], k_nm_grad.shape[2])

            forces_regularizer = self.regularizer[1] / delta
            k_nm_grad[:] /= forces_regularizer

            energy_grad = -forces.reshape(-1, 1)
            energy_grad[:] /= forces_regularizer

            Y = ops.vstack([Y, energy_grad])
            K_NM = ops.vstack([K_NM, k_nm_grad])

        K = K_MM + K_NM.T @ K_NM
        Y = K_NM.T @ Y

        self.weights = ops.linalg_solve(K, Y, detach=True)

    def predict(self, ps, with_forces=False):
        if self.weights is None:
            raise Exception("call fit first")

        k_per_atom = self._compute_kernel(ps)
        k_per_structure = structure_sum(k_per_atom, sum_features=False)
        # TODO: make sure this create an array in the same order as block_diag above
        if len(k_per_structure.sparse.names) != 0:
            names = list(k_per_structure.sparse.names)
            k_per_structure.sparse_to_features(names)

        kernel = invariant_block_to_2d_array(k_per_structure.block())

        energies = kernel @ self.weights + self.baseline

        if with_forces:
            _, kernel_grad = k_per_structure.block().gradient("positions")
            kernel_grad = kernel_grad.reshape(-1, self.weights.shape[0])

            forces = -kernel_grad @ self.weights
            forces = forces.reshape(-1, 3)
        else:
            forces = None

        return energies, forces

    def _compute_kernel(self, ps):
        return power(dot(ps, self.support_points, do_normalize=True), zeta=self.zeta)
