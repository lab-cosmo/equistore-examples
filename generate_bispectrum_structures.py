import numpy as np 
import ase 

def points_on_circle(r,phi):
    return np.array([[r * np.cos(p), r * np.sin(p)] for p in phi])

def rotate_vector_2d(psi, cartesian_pos):
    """
    Rotate points in cartesian_pos (a vector of (x,y)) by angle psi in 2D
    """
    matrix = np.zeros((2,2))
    matrix[0,0] = np.cos(psi)
    matrix[0,1] = -np.sin(psi)
    matrix[1,0] = np.sin(psi)
    matrix[1,1] = np.cos(psi)

    return np.einsum("ij, nj-> ni", matrix, cartesian_pos)

def generate_nu3_degen_structs(r,phi,psi,z1,z2, center_species='C', ring_species='H', z2species='O'):
    from ase import Atoms
    structs = []
    n = len(phi)
    layer1 = points_on_circle(r,phi)    
    layer2 = rotate_vector_2d(psi, layer1)
    for idx_str in range(2):
        positions = np.zeros((2*n+2,3))
        positions[1:n+1,:2] = layer1
        positions[1:n+1, 2] = z1
        positions[n+1:1+2*n,:2] = layer2
        positions[n+1:1+2*n, 2] = -z1
        #add z1 to layer1
        #add -z1 to layer2
        if idx_str ==0:
            positions[2*n+1, 2] = z2
        else:
            positions[2*n+1, 2] = -z2

        atom_string = center_species + ring_species*(2*n) + z2species
        atoms = Atoms(atom_string, positions=positions, cell=np.eye(3)*10)

        structs.append(atoms)

    return structs

def single_partial_derivative(parameters, diff_param_names=['r', 'phi', 'psi', 'z1', 'z2'], diff_param=None, delta=1e-6):
    #parameters is a list of [r,phi, psi, z1,z2, sp1, sp2, sp3]
    #diff_param cane be r, phi, psi, z1, z2
    #as a sanity check, we will compute the derivative for both the +,- structures 
    if diff_param is None or isinstance('z2', list): 
        raise ValueError("please specify a single variable name along which to compute partial derivative")
        
    delta_parameters = parameters.copy()
    diff_idx = diff_param_names.index(diff_param)
    delta_parameters[diff_idx]+=delta
    coords_plus, coords_minus = generate_nu3_degen_structs(*parameters )
    delta_coords_plus, delta_coords_minus = generate_nu3_degen_structs(*delta_parameters)
    
    partial_derivative_plus = (delta_coords_plus.positions - coords_plus.positions)/delta
    partial_derivative_minus = (delta_coords_minus.positions - coords_minus.positions)/delta
    return partial_derivative_plus, partial_derivative_minus

def derivatives_finite_differences(parameters, diff_param = None, delta=1e-6):
    # use the single partial derivative across multiple parameters if necessary 
    # using f(x+deltax, y+deltay) = f(x,y) + partial f/x deltax + partial f/y delta y 
    if diff_param is None or not isinstance(diff_param, list): 
        raise ValueError("please specify variable name(s) along which to compute partial derivative, single param should also be passed as list")
    
    for p in diff_param: 
        der_plus, der_minus = single_partial_derivative(parameters, diff_param_names=['r', 'phi', 'psi', 'z1', 'z2'], diff_param=p, delta=delta)
    
    return None
