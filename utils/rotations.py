import numpy as np
from scipy.spatial.transform import Rotation
from sympy.physics.wigner import wigner_d

def wigner_d_matrix(l, alpha, beta, gamma):
    """Computes a Wigner D matrix
     D^l_{mm'}(alpha, beta, gamma)
    from sympy and converts it to numerical values.
    (alpha, beta, gamma) are Euler angles (radians, ZYZ convention) and l the irrep.
    """
    return np.complex128(wigner_d(l, alpha, beta, gamma))

def rotation_matrix(alpha, beta, gamma):
    """A Cartesian rotation matrix in the appropriate convention
    (ZYZ, implicit rotations) to be consistent with the common Wigner D definition.
    (alpha, beta, gamma) are Euler angles (radians)."""
    return Rotation.from_euler("ZYZ", [alpha, beta, gamma]).as_matrix()