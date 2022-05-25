from typing import List
import copy
import numpy as np

import ase
from pylode import DensityProjectionCalculator

from equistore import Labels, TensorBlock, TensorMap


class PyLODESphericalExpansion:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(
        self, frames: List[ase.Atoms], show_progress: bool = False
    ) -> TensorMap:
        # Step 1: compute spherical expansion with pylode
        hypers = copy.deepcopy(self._hypers)
        
        if not isinstance(frames,list):
            frames = [frames]

        global_species = list(
            map(int, np.unique(np.concatenate([f.numbers for f in frames])))
        )

        calculator = DensityProjectionCalculator(**hypers)
        calculator.transform(frames, show_progress=show_progress)
        data = calculator.features
        info = calculator.representation_info

        # Step 2: move data around to follow the storage convention
        sparse = Labels(
            names=["spherical_harmonics_l", "center_species", "neighbor_species"],
            values=np.array(
                [
                    [l, center_species, neighbor_species]
                    for l in range(hypers["max_angular"] + 1)
                    for center_species in global_species
                    for neighbor_species in global_species
                ],
                dtype=np.int32,
            ),
        )

        properties = Labels(
            names=["n"],
            values=np.array([[n] for n in range(hypers["max_radial"])], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(hypers["max_angular"] + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        blocks = []
        for sparse_i, (l, center_species, neighbor_species) in enumerate(sparse):
            neighbor_species_i = global_species.index(neighbor_species)
            center_species_mask = np.where(info[:, 2] == center_species)[0]
            block_data = data[center_species_mask, neighbor_species_i, :, lm_slices[l]]
            block_data = block_data.swapaxes(1, 2)

            samples = Labels(
                names=["structure", "center"],
                values=np.copy(info[center_species_mask, :2]).astype(np.int32),
            )
            spherical_component = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            block_gradients = None
            gradient_samples = None
            if hypers["compute_gradients"]:
                gradients = calculator.feature_gradients
                grad_info = calculator.gradients_info

                gradient_samples = []
                block_gradients = []
                for sample_i, (structure, center) in enumerate(samples):
                    gradient_mask = np.logical_and(
                        np.logical_and(
                            grad_info[:, 0] == structure,
                            grad_info[:, 1] == center,
                        ),
                        grad_info[:, 4] == neighbor_species,
                    )
                    for grad_index in np.where(gradient_mask)[0]:
                        block_gradient = gradients[
                            grad_index, :, neighbor_species_i, :, lm_slices[l]
                        ]
                        block_gradient = block_gradient.swapaxes(1, 2)
                        if np.linalg.norm(block_gradient) == 0:
                            continue

                        block_gradients.append(block_gradient[None, :, :, :])

                        structure = grad_info[grad_index, 0]
                        neighbor = grad_info[grad_index, 2]

                        gradient_samples.append((sample_i, structure, neighbor))

                if len(gradient_samples) != 0:
                    block_gradients = np.vstack(block_gradients)
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom"],
                        values=np.vstack(gradient_samples).astype(np.int32),
                    )
                else:
                    block_gradients = np.zeros(
                        (0, 3, spherical_component.shape[0], properties.shape[0])
                    )
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom"],
                        values=np.zeros((0, 3), dtype=np.int32),
                    )

            block = TensorBlock(
                values=np.copy(block_data),
                samples=samples,
                components=[spherical_component],
                properties=properties,
            )

            spatial_component = Labels(
                names=["gradient_direction"],
                values=np.array([[0], [1], [2]], dtype=np.int32),
            )
            gradient_components = [spatial_component, spherical_component]

            if block_gradients is not None:
                block.add_gradient(
                    "positions", block_gradients, gradient_samples, gradient_components
                )

            blocks.append(block)

        return TensorMap(sparse, blocks)
