import numpy as np
from scipy.io import netcdf_file
import itertools
import zipfile
import io

from aml_storage import Labels, Block, Descriptor


def write(path, descriptor, dtype="default"):
    # The format used here is a collection of netcdf file, using classic netcdf
    # (v3) and 64-bit offsets, grouped together in a non compressed zip file.
    #
    # Each block is stored in a separate netcdf file & the sparse labels are
    # stored in another one
    file = zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED)

    # write the sparse labels in a sparse.nc file
    buffer = io.BytesIO()
    nc_file = netcdf_file(buffer, mode="w", version=2)
    names_size = names_size = max(
        len(name.encode("ascii")) for name in descriptor.sparse.names
    )
    nc_file.createDimension("names_length", names_size)
    _write_labels(nc_file, "sparse", descriptor.sparse, names_size)
    nc_file.flush()

    file.writestr("sparse.nc", data=buffer.getbuffer().tobytes())

    # and the blocks in block-{i}.nc
    for i in range(len(descriptor.sparse)):
        block = descriptor.block(i)

        buffer = io.BytesIO()
        nc_file = netcdf_file(buffer, mode="w", version=2)
        _write_block(nc_file, block, dtype)
        nc_file.flush()

        file.writestr(f"block-{i}.nc", data=buffer.getbuffer().tobytes())

    file.close()


def _write_labels(file, name, labels, names_size):
    count = len(labels)
    size = len(labels[0])

    file.createDimension(name, count)
    file.createDimension(f"{name}_variables", size)

    samples = file.createVariable(name, "int32", [name, f"{name}_variables"])
    samples[:] = labels.view(dtype=np.int32).reshape(count, size)

    names = file.createVariable(
        f"{name}_variables", "B", [f"{name}_variables", "names_length"]
    )

    for name in labels.names:
        assert len(name.encode("ascii")) <= names_size

    data = "".join(n.ljust(names_size) for n in labels.names)
    data = np.array(list(data.encode("ascii")), dtype="i1")
    names[:] = data.reshape(len(labels.names), names_size)


def _write_block(file, block, dtype):
    names_size = max(
        len(name.encode("ascii"))
        for name in itertools.chain(
            block.samples.names,
            block.components.names,
            block.features.names,
        )
    )
    # TODO: include gradient samples in names_size

    file.createDimension("names_length", names_size)

    _write_labels(file, "samples", block.samples, names_size)
    _write_labels(file, "components", block.components, names_size)
    _write_labels(file, "features", block.features, names_size)

    if dtype == "default":
        dtype = block.values.dtype

    values = file.createVariable("values", dtype, ["samples", "components", "features"])
    values[:] = block.values

    # TODO: we need to be able to list gradient?
    if block.has_gradient("positions"):
        gradient_samples, gradient = block.gradient("positions")

        parameter = "positions"
        _write_labels(
            file, f"gradient_{parameter}_samples", gradient_samples, names_size
        )

        values = file.createVariable(
            f"gradient_{parameter}",
            dtype,
            [f"gradient_{parameter}_samples", "components", "features"],
        )
        values[:] = gradient


def read(path):
    file = zipfile.ZipFile(path, mode="r")

    with file.open("sparse.nc") as sparse_fd:
        nc_file = netcdf_file(sparse_fd, mode="r")
        sparse = _read_labels(nc_file, "sparse")

    blocks = []
    for i in range(len(sparse)):
        with file.open(f"block-{i}.nc") as block_fd:
            nc_file = netcdf_file(block_fd, mode="r")

            samples = _read_labels(nc_file, "samples")
            components = _read_labels(nc_file, "components")
            features = _read_labels(nc_file, "features")

            values = nc_file.variables["values"][:].astype(np.float64)
            block = Block(values, samples, components, features)

            parameters = []
            for variable in nc_file.variables.keys():
                if variable.endswith("_samples"):
                    continue
                if variable.endswith("_samples_variables"):
                    continue

                if variable.startswith("gradient_"):
                    parameters.append(variable[9:])

            for parameter in parameters:
                samples = _read_labels(nc_file, f"gradient_{parameter}_samples")
                values = nc_file.variables[f"gradient_{parameter}"][:].astype(
                    np.float64
                )
                block.add_gradient(parameter, samples, values)

            blocks.append(block)

    return Descriptor(sparse, blocks)


def _read_labels(file, name):
    names = []
    for i in range(file.dimensions[f"{name}_variables"]):
        variable = file.variables[f"{name}_variables"][i, :].tolist()
        variable = map(lambda u: u.decode("ascii"), variable)
        variable = "".join(variable)
        names.append(variable.strip())

    return Labels(names, file.variables[name][:].astype(np.int32))
