import numpy as np
from math import sqrt

from aml_storage import Descriptor, Labels, Block

import utils.models.operations as ops


def compute_power_spectrum(spherical_expansion):
    assert spherical_expansion.sparse.names == (
        "spherical_harmonics_l",
        "center_species",
        "neighbor_species",
    )

    blocks = []
    sparse = []

    for (l1, cs1, ns1), spx_1 in spherical_expansion:
        for (l2, cs2, ns2), spx_2 in spherical_expansion:
            if l1 != l2 or cs1 != cs2:
                continue

            # with the same central species, we should have the same samples
            assert np.all(spx_1.samples == spx_2.samples)

            # TODO: explain (symmetry w.r.t. neighbor species exchange)
            if ns1 > ns2:
                continue
            elif ns1 == ns2:
                factor = 1.0 / sqrt(2 * l1 + 1)
            else:
                factor = sqrt(2) / sqrt(2 * l1 + 1)

            features = Labels(
                names=[f"{name}_1" for name in spx_1.features.names]
                + [f"{name}_2" for name in spx_2.features.names],
                values=np.array(
                    [
                        features_1.tolist() + features_2.tolist()
                        for features_1 in spx_1.features
                        for features_2 in spx_2.features
                    ],
                    dtype=np.int32,
                ),
            )

            data = factor * ops.einsum("ima, imb -> iab", spx_1.values, spx_2.values)

            block = Block(
                values=data.reshape(data.shape[0], 1, -1),
                samples=spx_1.samples,
                components=Labels.single(),
                features=features,
            )

            n_features = block.values.shape[2]

            if spx_1.has_gradient("positions"):
                gradients_samples_1, gradient_1 = spx_1.gradient("positions")
                gradients_samples_2, gradient_2 = spx_2.gradient("positions")

                gradients_samples = np.unique(
                    np.concatenate([gradients_samples_1, gradients_samples_2])
                )
                gradients_samples = gradients_samples.view(np.int32).reshape(-1, 4)

                gradients_samples = Labels(
                    names=gradients_samples_1.names, values=gradients_samples
                )

                gradients_sample_mapping = {
                    tuple(sample): i for i, sample in enumerate(gradients_samples)
                }

                gradient_data = ops.zeros_like(
                    gradient_1, (gradients_samples.shape[0], 1, n_features)
                )

                gradient_data_1 = factor * ops.einsum(
                    "ima, imb -> iab",
                    gradient_1,
                    spx_2.values[gradients_samples_1["sample"], :, :],
                ).reshape(gradients_samples_1.shape[0], -1)

                for sample, row in zip(gradients_samples_1, gradient_data_1):
                    new_row = gradients_sample_mapping[tuple(sample)]
                    gradient_data[new_row, 0, :] += row

                gradient_data_2 = factor * ops.einsum(
                    "ima, imb -> iab",
                    spx_1.values[gradients_samples_2["sample"], :, :],
                    gradient_2,
                ).reshape(gradients_samples_2.shape[0], -1)

                for sample, row in zip(gradients_samples_2, gradient_data_2):
                    new_row = gradients_sample_mapping[tuple(sample)]
                    gradient_data[new_row, 0, :] += row

                block.add_gradient("positions", gradients_samples, gradient_data)

            sparse.append((l1, cs1, ns1, ns2))
            blocks.append(block)

    sparse = Labels(
        names=[
            "spherical_harmonics_l",
            "center_species",
            "neighbor_species_1",
            "neighbor_species_2",
        ],
        values=np.array(sparse, dtype=np.int32),
    )
    descriptor = Descriptor(sparse, blocks)
    descriptor.sparse_to_features("spherical_harmonics_l")
    return descriptor
