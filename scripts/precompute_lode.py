#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute LODE descriptor for given dataset.

Calculations may take along time and memory. Be careful!
"""
import argparse
import logging
import os
import sys
from typing import Dict, List

import ase
import ase.io

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import equistore.io
from utils.pylode import (
    PyLODESphericalExpansion,
    PyLODESphericalExpansionRealspace,
)

# Print all log message to the console
logging.basicConfig(level=logging.DEBUG)


def precompute_lode(frames: List[ase.Atoms],
                    hypers: Dict,
                    realspace: bool = False,
                    show_progress: bool = False):
    """Calculate LODE descripor from an ase list of atoms.

    Parameters
    ----------
    frames : List[ase.Atoms]
        List of datasets for calculating features
    hypers : Dict
        Dictionary containng the LODE hyperparameters.
        See `pylode.DensityProjectionCalculator` for details.
    realspace : bool
        Use realspace implementation for calculating features.
    show_progress : bool
        Show a progress bar

    Returns
    -------
    descriptor : aml_storage.descriptor.Descriptor
        The feature descriptor
    """
    if realspace:
        calculator = PyLODESphericalExpansionRealspace(hypers)
    else:
        calculator = PyLODESphericalExpansion(hypers)

    descriptor = calculator.compute(frames=frames, show_progress=show_progress)

    return descriptor


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-input_file",
        dest="input_file",
        type=str,
        help="Trajectory for constructing features.",
        required=True,
    )
    parser.add_argument(
        "-index",
        dest="index",
        type=str,
        help="slicing string for trjectory slicing",
        default=":",
    )
    parser.add_argument(
        "-realspace",
        dest="realspace",
        action="store_true",
        help="Use the realspace implementation for calculating features.",
    )
    parser.add_argument(
        "-max_radial",
        dest="max_radial",
        type=int,
        help="Number of radial functions",
        default=6,
    )
    parser.add_argument(
        "-max_angular",
        dest="max_angular",
        type=int,
        help="Number of angular functions",
        default=6,
    )
    parser.add_argument(
        "-radial_basis_radius",
        dest="radial_basis_radius",
        type=float,
        help="Environment cutoff (Å)",
        default=4.0,
    )
    parser.add_argument(
        "-cutoff_radius",
        dest="cutoff_radius",
        type=float,
        help="Spherical real space cutoff to use for atomic environments (Å). "
             "If `None` the same cutoff_radius as radial_basis_radius is "
             "used. Only applies if realspace flag is set.",
        default=None,
    )
    parser.add_argument(
        "-smearing",
        dest="smearing",
        type=float,
        help="Smearing of the Gaussain (Å)."
        "Note that computational cost scales "
        "cubically with 1/smearing.",
        default=0.3,
    )
    parser.add_argument(
        "-kcut",
        dest="kcut",
        type=float,
        help="Cutoff for the kspace sum if. "
        "If `None` it is set to `1.2 * π / smearing`"
        "which is a reasonable estimate for many systems."
        "The option only has an effect without the `-realspace` flag.",
        default=None,
    )
    parser.add_argument(
        "-radial_basis",
        dest="radial_basis",
        type=str,
        default="monomial",
        const="monomial",
        nargs="?",
        choices=["monomial", "GTO", "GTO_primitive", "GTO_analytical"],
        help="The radial basis. Currently implemented "
        "are 'GTO_analytical', 'GTO_primitive', 'GTO', 'monomial'.",
    )
    parser.add_argument(
        "-potential_exponent",
        dest="potential_exponent",
        type=int,
        const=1,
        nargs="?",
        choices=list(range(7)),
        default=1,
        help="potential exponent: "
        "i.e. p=0 uses Gaussian densities, "
        "p=1 is LODE using 1/r (Coulomb) densities"
        "p=6 is LODE using 1/r^6 for dispersion",
    )
    parser.add_argument(
        "-subtract_center_contribution",
        dest="subtract_center_contribution",
        action="store_true",
        help="Subtract contribution from the central atom.",
    )
    parser.add_argument(
        "-compute_gradients",
        dest="compute_gradients",
        action="store_true",
        help="Compute gradients",
    )
    parser.add_argument(
        "-outfile",
        dest="outfile",
        type=str,
        help="Output filename for the feature matrix.",
        default="precomputed_lode",
    )

    args = parser.parse_args()
    frames = ase.io.read(
        args.__dict__.pop("input_file"), index=args.__dict__.pop("index")
    )

    # Remove paramaters only apply to real or kspace implementation.
    realspace = args.__dict__.pop("realspace")
    if realspace:
        args.__dict__.pop("kcut")
    else:
        args.__dict__.pop("cutoff_radius")
    
    equistore.io.save(
        args.__dict__.pop("outfile"),
        precompute_lode(frames,
                        args.__dict__,
                        realspace=realspace,
                        show_progress=True),
        use_numpy=True
    )


if __name__ == "__main__":
    main()
