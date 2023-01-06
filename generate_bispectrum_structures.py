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
    #     positions[0] = [0,0,0] #central atom
        positions[1:n+1,:2] = layer1
        positions[1:n+1, 2] = z1
#         print(positions,'\n')
        positions[n+1:1+2*n,:2] = layer2
        positions[n+1:1+2*n, 2] = -z1
#         print(positions,'\n')
        #add z1 to layer1
        #add -z1 to layer2
        if idx_str ==0:
            positions[2*n+1, 2] = z2
        else:
            positions[2*n+1, 2] = -z2
#         print(positions,'\n')

        atom_string = center_species + ring_species*(2*n) + z2species
        atoms = Atoms(atom_string, positions=positions, cell=np.eye(3)*10)

        structs.append(atoms)

    return structs


