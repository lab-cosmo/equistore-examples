import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap


def get_tensorblock_name_structure(tensorblock: equistore.block.TensorBlock) -> dict:
    """
    Takes a TensorBlock object and returns a dict showing the
    structure of the Label names. An example is shown below for
    a descriptor where 'cell' and 'positions' gradients have been
    calculated:

    {
    "samples": ("structure", "center"),
    "components": [("spherical_harmonics_m",)],
    "properties": ("n",),
    "gradients": {
        "cell": {
            "samples": ("sample",),
            "components": [
                ("direction_1",),
                ("direction_2",),
                ("spherical_harmonics_m",),
            ],
            "properties": ("n",),
        },
        "positions": {
            "samples": ("sample", "structure", "atom"),
            "components": [("direction",), ("spherical_harmonics_m",)],
            "properties": ("n",),
        },
    },
    }

    Parameters
    ----------
    tensorblock : equistore.block.TensorBlock
        The input TensorBlock object whose Label name structure will be returned

    Returns
    -------
    dict
        The name structure of the input TensorBlock.
    """
    names = {
        "samples": tensorblock.samples.names,
        "components": [c.names for c in tensorblock.components],
        "properties": tensorblock.properties.names,
    }
    if type(tensorblock) == equistore.block.TensorBlock:
        names.update(
            {
                "gradients": {
                    parameter: get_tensorblock_name_structure(gradient_block)
                    for parameter, gradient_block in tensorblock.gradients()
                },
            }
        )
    return names


def get_tensormap_name_structure(tensormap: equistore.tensor.TensorMap) -> dict:
    """
    Takes a TensorMap object and returns a dict showing the
    structure of the Label names. An example is shown below for
    a descriptor where 'cell' and 'positions' gradients have been
    calculated:

    {
    "keys": ("spherical_harmonics_l", "species_center", "species_neighbor"),
    "samples": ("structure", "center"),
    "components": [("spherical_harmonics_m",)],
    "properties": ("n",),
    "gradients": {
        "cell": {
            "samples": ("sample",),
            "components": [
                ("direction_1",),
                ("direction_2",),
                ("spherical_harmonics_m",),
            ],
            "properties": ("n",),
        },
        "positions": {
            "samples": ("sample", "structure", "atom"),
            "components": [("direction",), ("spherical_harmonics_m",)],
            "properties": ("n",),
        },
    },
    }

    Parameters
    ----------
    tensormap : equistore.tensor.TensorMap
        The input TensorMap object whose Label name structure will be returned

    Returns
    -------
    dict
        The name structure of the input TensorMap.
    """
    names = {"keys": tensormap.keys.names}
    names.update(get_tensorblock_name_structure(tensormap.block(0)))
    return names


def rename_tensorblock(
    tensorblock: equistore.block.TensorBlock, new_names: dict
) -> equistore.block.TensorBlock:
    """
    Takes a TensorBlock and creates a new TensorBlock with the
    same data values, but but different Label names for the:

        - samples, components and properties of the TensorBlock,
        - samples, components, and properties of each of the
          TensorBlock's Gradient TensorBlocks,

    according to the desired new names stored in the new_names dict.

    In order to pass a correctly structured naming dictionary, first
    call the function get_tensorblock_name_structure() on your TensorBlock
    to see its name structure, then edit the name values accordingly.

    Note: the names of the properties of Gradient TensorBlocks must
    be the same as those of the parent TensorBlock.

    Note: for N components names for the TensorBlock, the final N names of
    the components of the Gradient TensorBlock should be equivalent.

    Note: this function does not techically 'rename' the TensorBlock, but
    instead creates a new TensorBlock with updated names.

    Parameters
    ----------
    tensorblock : equistore.block.TensorBlock
        The input TensorBlock whose data values will be copied, but Labels
        renamed.
    new_names : dict
        A dictionary containing the desired names for the new TensorBlock.
        This must have the correct structure; use the
        get_tensorblock_name_structure() function on your TensorMap first
        and edit the names to the desired ones.

    Returns
    -------
    equistore.block.TensorBlock
        A new TensorBlock object with data values copied from the old
        TensorBlock, but will new Label names.

    """
    # The Labels names of the properties of each Gradient TensorBlock should be the same
    # as the Labels Check that the user hasn't tried to define different Gradient
    # TensorBlock and parent TensorBlock properties names.
    props = list(new_names["properties"])
    grad_props_list = [grad["properties"] for grad in new_names["gradients"].values()]
    assert np.all(
        [np.array_equal(props, grad_props) for grad_props in grad_props_list]
    ), (
        "The names of TensorBlock properties Labels should be the same as those for the Gradient TensorBlocks. "
        + "You have passed "
        + str(props)
        + " as the names of the properties of the TensorBlock, and "
        + str(grad_props_list)
        + " as the names of the properties for the Gradient TensorBlocks."
    )

    # Where there are N number of components names for a given TensorBlock,
    # the final N components names of the associated Gradient TensorBlocks
    # should be equivalent. Check that the user hasn't tried to define different
    # Gradient TensorBlock and parent TensorBlock components names.
    comps = list(new_names["components"])
    N_comps = len(comps)
    grad_comps_list = [
        grad["components"][-N_comps:] for grad in new_names["gradients"].values()
    ]
    assert np.all(
        [np.array_equal(comps, grad_comps) for grad_comps in grad_comps_list]
    ), (
        "Where there are N number of names of TensorBlock components, the final N names of the Gradient TensorBlock components should match. "
        + "You have passed "
        + str(comps)
        + " as the names of the properties of the TensorBlock, and "
        + str(grad_comps_list)
        + " as the names of the properties for each of the Gradient TensorBlocks."
    )

    # Define new block. Copy block values but create new samples, components,
    # properties names
    new_block = TensorBlock(
        values=tensorblock.values,
        samples=Labels(
            names=new_names["samples"], values=tensorblock.samples.asarray()
        ),
        components=[
            Labels(names=c, values=tensorblock.components[i].asarray())
            for i, c in enumerate(new_names["components"])
        ],
        properties=Labels(
            names=new_names["properties"], values=tensorblock.properties.asarray()
        ),
    )
    # Add Gradient TensorBlocks to the new parent TensorBlock.
    # Properties names get inherited from the parent TensorBlock.
    gblocks = {param: gblock for param, gblock in list(tensorblock.gradients())}
    for param, names in new_names["gradients"].items():
        new_block.add_gradient(
            parameter=param,
            data=gblocks[param].data,
            samples=Labels(
                names=names["samples"],
                values=gblocks[param].samples.asarray(),
            ),
            components=[
                Labels(names=c, values=gblocks[param].components[i].asarray())
                for i, c in enumerate(names["components"])
            ],
        )
    return new_block


def rename_tensormap(
    tensormap: equistore.tensor.TensorMap, new_names: dict
) -> equistore.tensor.TensorMap:
    """
    Takes a TensorMap and creates a new TensorMap with the same data
    values, but different Labels names for the:

        - keys of the TensorMap,
        - samples, components and properties of each TensorBlock,
        - samples, components, and properties of each Gradient
          TensorBlock of each TensorBlock,

    according to the desired new names stored in the new_names dict.

    In order to pass a correctly structured naming dictionary, first
    call the function get_tensormap_name_structure() on your TensorMap
    to see its name structure, then edit the name values accordingly.

    Note: the names of the properties of Gradient TensorBlocks must
    be the same as those of the TensorBlocks.

    Note: this function does not techically 'rename' the TensorBlock, but
    instead creates a new TensorBlock with updated names.

    Parameters
    ----------
    tensormap : equistore.tensor.TensorMap
        The input TensorMap whose data values will be copied, but Labels
        renamed.
    new_names : dict
        A dictionary containing the desired names for the new TensorMap.
        This must have the correct structure; use the
        get_tensormap_name_structure() function on your TensorMap first
        and edit the names to the desired ones.

    Returns
    -------
    equistore.tensor.TensorMap
        A new TensorMap object with data values copied from the old
        TensorMap, but will new Label names.
    """
    return TensorMap(
        keys=Labels(names=new_names["keys"], values=tensormap.keys.asarray()),
        blocks=[rename_tensorblock(block, new_names) for block in tensormap.blocks()],
    )
