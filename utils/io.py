import numpy as np
import h5py

from aml_storage import Labels, Block, Descriptor


def write(path, descriptor, dtype="default"):
    path += '.h5' * (not path.endswith('.h5'))
    with h5py.File(path, mode="w", track_order=True) as file:
        _write_labels(file, "sparse", descriptor.sparse)

        for i, (_, block) in enumerate(descriptor):
            _write_block(file, f"block-{i}", block)


def _write_labels(h5_group, name, labels):
    dataset = h5_group.create_dataset(
        name,
        data=labels.view(dtype="int32").reshape(len(labels), -1),
    )

    dataset.attrs["names"] = labels.names


def _write_block(root, name, block):
    group = root.create_group(name)

    group.create_dataset(
        "values",
        data=block.values,
    )
    _write_labels(group, "samples", block.samples)
    _write_labels(group, "components", block.components)
    _write_labels(group, "features", block.features)

    if len(block.gradients_list()) == 0:
        return

    gradients = group.create_group("gradients")
    for parameter in block.gradients_list():
        gradient_sample, gradient_data = block.gradient(parameter)

        group = gradients.create_group(parameter)

        group.create_dataset(
            "data",
            data=gradient_data,
        )
        _write_labels(group, "samples", gradient_sample)
        _write_labels(group, "components", block.components)
        _write_labels(group, "features", block.features)


def read(path):
    with h5py.File(path, mode="r") as file:
        sparse = file["sparse"]
        sparse = Labels(sparse.attrs["names"], np.array(sparse))

        blocks = []
        for i in range(len(sparse)):
            h5_block = file[f"block-{i}"]

            values = np.array(h5_block["values"])
            samples = h5_block["samples"]
            samples = Labels(samples.attrs["names"], np.array(samples))

            components = h5_block["components"]
            components = Labels(components.attrs["names"], np.array(components))

            features = h5_block["features"]
            features = Labels(features.attrs["names"], np.array(features))

            block = Block(values, samples, components, features)

            if "gradients" in h5_block:
                gradients = h5_block["gradients"]
                for parameter, grad_block in gradients.items():
                    data = np.array(grad_block["data"])

                    samples = grad_block["samples"]
                    samples = Labels(samples.attrs["names"], np.array(samples))

                    # skip components & features for now

                    block.add_gradient(parameter, samples, data)

            blocks.append(block)

    return Descriptor(sparse, blocks)
