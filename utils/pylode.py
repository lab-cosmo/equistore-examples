from typing import List
import copy
import numpy as np

import warnings

import ase
from pylode import DensityProjectionCalculator as SphericalExpansion

from aml_storage import Labels, Block, Descriptor


class PyLODESphericalExpansion:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames: List[ase.Atoms]) -> Descriptor:
        # Step 1: compute spherical expansion with pylode
        hypers = copy.deepcopy(self._hypers)
        global_species = list(
            map(int, np.unique(np.concatenate([f.numbers for f in frames])))
        )

        hypers["global_species"] = global_species
        hypers["expansion_by_species_method"] = "user defined"

        calculator = SphericalExpansion(**hypers)
        calculator.transform(frames)
        data = calculator.features
        gradients = calculator.gradients
        info = calculator.representation_info
        grad_info = calculator.gradients_info

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

        features = Labels(
            names=["n"],
            values=np.array([[n] for n in range(hypers["max_radial"])], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(hypers["max_angular"] + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        data = data.reshape(
            data.shape[0], len(global_species), hypers["max_radial"], -1
        )
        gradients = gradients.reshape(
            gradients.shape[0], len(global_species), hypers["max_radial"], -1
        )

        global_to_per_structure_atom_id = []
        for frame in frames:
            for i in range(len(frame)):
                global_to_per_structure_atom_id.append(i)

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
            components = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            block_gradients = None
            gradient_samples = None
            if hypers["compute_gradients"]:
                warnings.warn(
                    "numpy/forward gradients are currently broken with librascal,"
                    "please use rascaline instead"
                )
                # the code below is missing some of the gradient sample, making
                # the forces not match the energy in a finite difference test

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
                        start = 3 * grad_index
                        stop = 3 * grad_index + 3
                        block_gradient = gradients[
                            start:stop, neighbor_species_i, :, lm_slices[l]
                        ]
                        block_gradient = block_gradient.swapaxes(1, 2)
                        if np.linalg.norm(block_gradient) == 0:
                            continue

                        block_gradients.append(block_gradient)

                        structure = grad_info[grad_index, 0]
                        neighbor = global_to_per_structure_atom_id[
                            grad_info[grad_index, 2]
                        ]
                        gradient_samples.append((sample_i, structure, neighbor, 0))
                        gradient_samples.append((sample_i, structure, neighbor, 1))
                        gradient_samples.append((sample_i, structure, neighbor, 2))

                if len(gradient_samples) != 0:
                    block_gradients = np.concatenate(block_gradients)
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom", "spatial"],
                        values=np.vstack(gradient_samples).astype(np.int32),
                    )
                else:
                    block_gradients = np.zeros(
                        (0, components.shape[0], features.shape[0])
                    )
                    gradient_samples = Labels(
                        names=["sample", "structure", "atom", "spatial"],
                        values=np.zeros((0, 4), dtype=np.int32),
                    )

            # reset atom index (librascal uses a global atom index)
            samples = Labels(
                names=["structure", "center"],
                values=np.array(
                    [
                        (structure, global_to_per_structure_atom_id[center])
                        for structure, center in samples
                    ],
                    dtype=np.int32,
                ),
            )

            block = Block(
                values=np.copy(block_data),
                samples=samples,
                components=components,
                features=features,
            )

            if block_gradients is not None:
                block.add_gradient("positions", gradient_samples, block_gradients)

            blocks.append(block)

        return Descriptor(sparse, blocks)
