import numpy as np
import torch
from aml_storage import Block, Labels, Descriptor

import utils.models.operations as ops


def invariant_block_to_2d_array(block):
    assert len(block.values.shape) == 3
    assert block.values.shape[1] == 1
    return block.values.reshape(block.values.shape[0], -1)


def array_2d_to_invariant(array):
    assert len(array.shape) == 2

    return array.reshape(array.shape[0], 1, -1)


def normalize(descriptor):
    blocks = []
    for sparse, block in descriptor:
        # only deal with invariants for now
        assert block.values.shape[1] == 1

        values = block.values.reshape(-1, block.values.shape[2])
        norm = ops.norm(values, axis=1)
        values = values / norm[:, None]

        new_block = Block(
            values=values.reshape(-1, 1, values.shape[1]),
            samples=block.samples,
            components=block.components,
            features=block.features,
        )

        if block.has_gradient("positions"):
            gradient_samples, gradient = block.gradient("positions")

            gradient = gradient.reshape(-1, gradient.shape[2])
            gradient = gradient / norm[gradient_samples["sample"], None]

            # gradient of x_i = X_i / N_i is given by
            # 1 / N_i \grad X_i - x_i [x_i @ 1 / N_i \grad X_i]
            for sample_i, (sample, _, _, _) in enumerate(gradient_samples):
                dot = gradient[sample_i] @ values[sample].T
                gradient[sample_i, :] -= dot * values[sample, :]

            new_block.add_gradient(
                "positions",
                gradient_samples,
                gradient.reshape(-1, 1, gradient.shape[1]),
            )

        blocks.append(new_block)

    return Descriptor(descriptor.sparse, blocks)


def dot(lhs_descriptor, rhs_descriptor, do_normalize=True):
    assert len(lhs_descriptor.sparse) == len(rhs_descriptor.sparse)
    if len(lhs_descriptor.sparse) != 0:
        assert np.all(lhs_descriptor.sparse == rhs_descriptor.sparse)

    if do_normalize:
        lhs_descriptor = normalize(lhs_descriptor)

    blocks = []
    for sparse, lhs_block in lhs_descriptor:
        rhs_block = rhs_descriptor.block(sparse)
        assert np.all(lhs_block.features == rhs_block.features)

        # only deal with invariants for now
        assert lhs_block.values.shape[1] == 1
        assert rhs_block.values.shape[1] == 1

        samples = lhs_block.samples
        features = rhs_block.samples

        rhs = invariant_block_to_2d_array(rhs_block)
        lhs = invariant_block_to_2d_array(lhs_block)

        block = Block(
            values=array_2d_to_invariant(lhs @ rhs.T),
            samples=samples,
            components=Labels.single(),
            features=features,
        )

        if lhs_block.has_gradient("positions"):
            gradient_samples, gradient = lhs_block.gradient("positions")

            gradient_data = gradient.reshape(-1, gradient.shape[2]) @ rhs.T

            block.add_gradient(
                "positions",
                gradient_samples,
                gradient_data.reshape(-1, 1, gradient_data.shape[1]),
            )

        if rhs_block.has_gradient("positions"):
            print("ignoring gradients of kernel support points")

        blocks.append(block)

    return Descriptor(lhs_descriptor.sparse, blocks)


def power(descriptor, zeta):
    assert zeta >= 1

    blocks = []
    for _, block in descriptor:
        new_block = Block(
            ops.float_power(block.values, zeta),
            block.samples,
            block.components,
            block.features,
        )

        if block.has_gradient("positions"):
            gradient_samples, gradient = block.gradient("positions")

            if zeta > 1:
                gradient_data = zeta * ops.float_power(
                    block.values[gradient_samples["sample"]], zeta - 1
                )
                gradient_data *= gradient
            else:
                assert zeta == 1
                gradient_data = gradient

            new_block.add_gradient("positions", gradient_samples, gradient_data)

        blocks.append(new_block)

    return Descriptor(descriptor.sparse, blocks)


def structure_sum(descriptor, sum_features=False):
    blocks = []
    for _, block in descriptor:

        # no lambda kernels for now
        assert block.values.shape[1] == 1

        structures = np.unique(block.samples["structure"])

        if sum_features:
            ref_structures = np.unique(block.features["structure"])
            features = Labels(["structure"], ref_structures.reshape(-1, 1))
        else:
            features = block.features

        result = ops.zeros_like(block.values, (len(structures), 1, features.shape[0]))

        if block.has_gradient("positions"):
            do_gradients = True
            gradient_samples, gradient = block.gradient("positions")

            assert np.all(np.unique(gradient_samples["structure"]) == structures)

            gradient_data = []
            new_gradient_samples = []
            grad_atoms_by_structure = []  # TODO: bad name
            for structure_i, s1 in enumerate(structures):
                mask = gradient_samples["structure"] == s1
                atoms = np.unique(gradient_samples[mask]["atom"])

                gradient_data.append(
                    ops.zeros_like(
                        gradient,
                        (3 * len(atoms), 1, features.shape[0]),
                    )
                )

                new_gradient_samples.append(
                    np.array(
                        [
                            [structure_i, s1, atom, spatial]
                            for atom in atoms
                            for spatial in range(3)
                        ],
                        dtype=np.int32,
                    )
                )

                grad_atoms_by_structure.append(
                    {atom: i for i, atom in enumerate(atoms)}
                )

            new_gradient_samples = Labels(
                names=["sample", "structure", "atom", "spatial"],
                values=np.concatenate(new_gradient_samples),
            )

        else:
            do_gradients = False

        if sum_features:
            for structure_i, s1 in enumerate(structures):
                s1_idx = block.samples["structure"] == s1

                for structure_j, s2 in enumerate(ref_structures):
                    s2_idx = block.features["structure"] == s2
                    result[structure_i, 0, structure_j] = ops.sum(
                        block.values[s1_idx, 0, :][:, s2_idx]
                    )

                    if do_gradients:
                        idx = np.where(gradient_samples["structure"] == s1)[0]
                        for sample_i in idx:
                            grad_sample = gradient_samples[sample_i]
                            atom_i = grad_atoms_by_structure[structure_i][
                                grad_sample["atom"]
                            ]
                            index = 3 * atom_i + grad_sample["spatial"]

                            gradient_data[structure_i][
                                index, 0, structure_j
                            ] += ops.sum(gradient[sample_i, 0, s2_idx])
        else:
            for structure_i, s1 in enumerate(structures):
                s1_idx = block.samples["structure"] == s1

                result[structure_i, 0, :] = ops.sum(block.values[s1_idx, 0, :], axis=0)

                if do_gradients:
                    for sample_i in np.where(gradient_samples["structure"] == s1)[0]:
                        grad_sample = gradient_samples[sample_i]

                        atom_i = grad_atoms_by_structure[structure_i][
                            grad_sample["atom"]
                        ]
                        index = 3 * atom_i + grad_sample["spatial"]

                        gradient_data[structure_i][index, 0, :] += gradient[
                            sample_i, 0, :
                        ]

        new_block = Block(
            values=result,
            samples=Labels(["structure"], structures.reshape(-1, 1)),
            components=Labels.single(),
            features=features,
        )

        if do_gradients:
            gradient = []
            for d in gradient_data:
                gradient.append(d.reshape(-1, features.shape[0]))

            gradient_data = ops.vstack(gradient).reshape(-1, 1, features.shape[0])
            new_block.add_gradient("positions", new_gradient_samples, gradient_data)

        blocks.append(new_block)

    return Descriptor(sparse=descriptor.sparse, blocks=blocks)


def detach(descriptor):
    if isinstance(descriptor.block(0).values, torch.Tensor):
        blocks = []
        for _, block in descriptor:
            blocks.append(
                Block(
                    values=block.values.detach(),
                    samples=block.samples,
                    components=block.components,
                    features=block.features,
                )
            )
        descriptor = Descriptor(descriptor.sparse, blocks)

    return descriptor
