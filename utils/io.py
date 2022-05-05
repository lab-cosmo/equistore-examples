import numpy as np
import h5py

from equistore import Labels, TensorBlock, TensorMap


def write(path, descriptor, dtype="default"):
    if not path.endswith(".h5"):
        path += ".h5"

    with h5py.File(path, mode="w", track_order=True) as file:
        _write_labels(file, "keys", descriptor.keys)

        blocks = file.create_group("blocks")
        for i, (_, block) in enumerate(descriptor):
            _write_block(blocks, str(i), block)


def _write_labels(h5_group, name, labels):
    if len(labels) == 0:
        data = np.zeros((0, len(labels.names)), dtype=np.int32)
    else:
        data = labels.view(dtype="int32").reshape(len(labels), -1)

    dataset = h5_group.create_dataset(name, data=data)

    dataset.attrs["names"] = labels.names


def _write_block(root, name, block):
    group = root.create_group(name)

    group.create_dataset(
        "values",
        data=block.values,
    )
    _write_labels(group, "samples", block.samples)
    components = block.components
    if components:
        component_group = group.create_group("components")
        for i, component in enumerate(components):
            _write_labels(component_group, str(i), component)
    _write_labels(group, "properties", block.properties)

    if len(block.gradients_list()) == 0:
        return

    gradients = group.create_group("gradients")
    for parameter in block.gradients_list():
        gradient = block.gradient(parameter)

        group = gradients.create_group(parameter)

        group.create_dataset(
            "data",
            data=gradient.data,
        )
        _write_labels(group, "samples", gradient.samples)
        components = gradient.components
        if components:
            component_group = group.create_group("components")
            for i, component in enumerate(components):
                _write_labels(component_group, str(i), component)
        _write_labels(group, "properties", gradient.properties)


def read(path):
    with h5py.File(path, mode="r") as file:
        keys = file["keys"]
        keys = Labels(keys.attrs["names"], np.array(keys))

        blocks = []
        for i in range(len(keys)):
            h5_block = file[f"blocks/{i}"]

            values = np.array(h5_block["values"])
            samples = h5_block["samples"]
            samples = Labels(samples.attrs["names"], np.array(samples))

            components = []
            if "components" in h5_block:
                for i in range(len(h5_block["components"])):
                    component = h5_block[f"components/{i}"]
                    components.append(
                        Labels(component.attrs["names"], np.array(component))
                    )

            properties = h5_block["properties"]
            properties = Labels(properties.attrs["names"], np.array(properties))

            block = TensorBlock(values, samples, components, properties)

            if "gradients" in h5_block:
                gradients = h5_block["gradients"]
                for parameter, grad_block in gradients.items():
                    data = np.array(grad_block["data"])

                    samples = grad_block["samples"]
                    samples = Labels(samples.attrs["names"], np.array(samples))

                    components = []
                    if "components" in grad_block:
                        for i in range(len(grad_block["components"])):
                            component = grad_block[f"components/{i}"]
                            components.append(
                                Labels(component.attrs["names"], np.array(component))
                            )

                    # skip components & properties for now

                    block.add_gradient(parameter, data, samples, components)

            blocks.append(block)

    return TensorMap(keys, blocks)


################################################################################


def _tensor_map_to_dict(tensor_map):
    result = {}
    result["keys"] = tensor_map.keys

    for block_i, (_, block) in enumerate(tensor_map):
        prefix = f"blocks/{block_i}/values"
        result[f"{prefix}/data"] = block.values
        result[f"{prefix}/samples"] = block.samples
        for i, component in enumerate(block.components):
            result[f"{prefix}/components/{i}"] = component
        result[f"{prefix}/properties"] = block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            prefix = f"blocks/{block_i}/gradient/{parameter}"
            result[f"{prefix}/data"] = gradient.data
            result[f"{prefix}/samples"] = gradient.samples
            for i, component in enumerate(gradient.components):
                result[f"{prefix}/components/{i}"] = component

    return result


def write_npz(path, tensor_map):
    assert path.endswith(".npz")

    all_entries = _tensor_map_to_dict(tensor_map)

    np.savez(path, **all_entries)


def _labels_from_npz(data):
    names = data.dtype.names
    return Labels(names=names, values=data.view(dtype=np.int32).reshape(-1, len(names)))


def read_npz(path):
    dictionary = np.load("spx.npz")

    keys = _labels_from_npz(dictionary["keys"])
    blocks = []

    gradient_parameters = []
    for block_i in range(len(keys)):
        prefix = f"blocks/{block_i}/values"
        data = dictionary[f"{prefix}/data"]

        samples = _labels_from_npz(dictionary[f"{prefix}/samples"])
        components = []
        for i in range(len(data.shape) - 2):
            components.append(_labels_from_npz(dictionary[f"{prefix}/components/{i}"]))

        properties = _labels_from_npz(dictionary[f"{prefix}/properties"])

        block = TensorBlock(data, samples, components, properties)

        if block_i == 0:
            prefix = f"blocks/{block_i}/gradient/"
            for name in dictionary.keys():
                if name.startswith(prefix) and name.endswith("/data"):
                    gradient_parameters.append(name[len(prefix) : -len("/data")])

        for parameter in gradient_parameters:
            prefix = f"blocks/{block_i}/gradient/{parameter}"
            data = dictionary[f"{prefix}/data"]

            samples = _labels_from_npz(dictionary[f"{prefix}/samples"])
            components = []
            for i in range(len(data.shape) - 2):
                components.append(
                    _labels_from_npz(dictionary[f"{prefix}/components/{i}"])
                )

            block.add_gradient(parameter, data, samples, components)

        blocks.append(block)

    return TensorMap(keys, blocks)
