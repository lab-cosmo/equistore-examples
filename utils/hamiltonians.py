from .builder import DescriptorBuilder
from .clebsh_gordan import ClebschGordanReal
import numpy as np

###########  I/O UTILITIES ##############
def fix_pyscf_l1(dense, frame, orbs):
    """ pyscf stores l=1 terms in a xyz order, corresponding to (m=0, 1, -1).
        this converts into a canonical form where m is sorted as (-1, 0,1) """
    idx = []
    iorb = 0
    atoms = list(frame.numbers)
    for atype in atoms:
        cur=()
        for ia, a in enumerate(orbs[atype]):
            n,l,m = a
            if (n,l) != cur:
                if l == 1:
                    idx += [iorb+1, iorb+2, iorb]
                else:
                    idx += range(iorb, iorb+2*l+1)
                iorb += 2*l+1
                cur = (n,l)
    return dense[idx][:,idx]

############ matrix/block manipulations ###############

def _components_idx(l):
    """ just a mini-utility function to get the m=-l..l indices """
    return np.arange(-l,l+1, dtype=np.int32).reshape(2*l+1,1)

def _components_idx_2d(li, lj):
    """ indexing the entries in a 2d (l_i, l_j) block of the hamiltonian
    in the uncoupled basis """
    return np.array(np.meshgrid(_components_idx(li), _components_idx(lj)), dtype = np.int32).T.reshape(-1,2)

def _orbs_offsets(orbs):
    """ offsets for the orbital subblocks within an atom block of the Hamiltonian matrix """
    orbs_tot = {}
    orbs_offset = {}
    for k in orbs:
        ko = 0
        for n,l,m in orbs[k]:
            if m != -l:
                continue
            orbs_offset[(k,n,l)] = ko
            ko+=2*l+1
        orbs_tot[k] = ko
    return orbs_tot, orbs_offset

def _atom_blocks_idx(frames, orbs_tot):
    """ position of the hamiltonian subblocks for each atom in each frame """
    atom_blocks_idx = {}
    for A, f in enumerate(frames):
        ki = 0
        for i, ai in enumerate(f.numbers):
            kj = 0
            for j, aj in enumerate(f.numbers):
                atom_blocks_idx[(A, i, j)] = (ki, kj)
                kj += orbs_tot[aj]
            ki += orbs_tot[ai]
    return atom_blocks_idx

def dense_to_blocks(dense, frames, orbs):
    """
    Converts a list of dense matrices `dense` corresponding to the single-particle Hamiltonians for the structures
    described by `frames`, and using the orbitals described in the dictionary `orbs` into a TensorMap storage format.

    The label convention is as follows: 

    The keys that label the blocks are ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"].
    block_type: 0 -> diagonal blocks, atom i=j
                2 -> different species block, stores only when n_i,l_i and n_j,l_j are lexicographically sorted
                1,-1 -> same specie, off-diagonal. store separately symmetric (1) and anti-symmetric (-1) term
    a_{i,j}: chemical species (atomic number) of the two atoms
    n_{i,j}: radial channel
    l_{i,j}: angular momentum
    """

    block_builder = DescriptorBuilder(["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"], ["structure", "atom_i", "atom_j"], [["m1"], ["m2"]], ["value"])
    orbs_tot, _ = _orbs_offsets(orbs)
    for A in range(len(frames)):
        frame = frames[A]
        ham = dense[A]
        ki_base = 0
        for i, ai in enumerate(frame.numbers):
            kj_base = 0
            for j, aj in enumerate(frame.numbers):
                if i==j:
                    block_type = 0  # diagonal
                elif ai==aj:
                    if i>j:
                        kj_base+=orbs_tot[aj]
                        continue
                    block_type = 1  # same-species
                else:
                    if ai>aj: # only sorted element types
                        kj_base += orbs_tot[aj]
                        continue
                    block_type = 2  # different species
                block_data = ham[ki_base:ki_base+orbs_tot[ai], kj_base:kj_base+orbs_tot[aj]]
                #print(block_data, block_data.shape)
                if block_type == 1:
                    block_data_plus = (block_data + block_data.T) *1/np.sqrt(2)
                    block_data_minus = (block_data - block_data.T) *1/np.sqrt(2)
                ki_offset = 0
                for ni, li, mi in orbs[ai]:
                    if mi != -li: # picks the beginning of each (n,l) block and skips the other orbitals
                        continue
                    kj_offset = 0
                    for nj, lj, mj in orbs[aj]:
                        if mj != -lj: # picks the beginning of each (n,l) block and skips the other orbitals
                            continue                    
                        if (ai==aj and (ni>nj or (ni==nj and li>lj))): 
                            kj_offset += 2*lj+1
                            continue
                        block_idx = (block_type, ai, ni, li, aj, nj, lj)
                        if block_idx not in block_builder.blocks:
                            block = block_builder.add_block(sparse=block_idx, properties=np.asarray([[0]], dtype=np.int32),
                                            components=[_components_idx(li), _components_idx(lj)] )
                            
                            if block_type == 1:
                                block_asym = block_builder.add_block(sparse=(-1,)+block_idx[1:], properties=np.asarray([[0]], dtype=np.int32), 
                                            components=[_components_idx(li), _components_idx(lj)])
                        else:
                            block = block_builder.blocks[block_idx]
                            if block_type == 1:
                                block_asym = block_builder.blocks[(-1,)+block_idx[1:]]
                        
                        islice = slice(ki_offset,ki_offset+2*li+1)
                        jslice = slice(kj_offset,kj_offset+2*lj+1)
                        
                        if block_type == 1:
                            block.add_samples(labels=[(A,i,j)],data=block_data_plus[islice, jslice].reshape((1,2*li+1,2*lj+1,1)) )
                            block_asym.add_samples(labels=[(A,i,j)], 
                                                   data=block_data_minus[islice, jslice].reshape((1,2*li+1,2*lj+1,1)) )
                        else:
                            block.add_samples(labels=[(A,i,j)], data=block_data[islice, jslice].reshape((1,2*li+1,2*lj+1,1)) )
                        
                        kj_offset += 2*lj+1
                    ki_offset += 2*li+1
                kj_base+=orbs_tot[aj]

            ki_base+=orbs_tot[ai]
    return block_builder.build()    


def blocks_to_dense(blocks, frames, orbs):
    """
    Converts a TensorMap containing matrix blocks in the uncoupled basis, `blocks` into dense matrices.
    Needs `frames` and `orbs` to reconstruct matrices in the correct order. See `dense_to_blocks` to understant
    the different types of blocks.
    """

    orbs_tot, orbs_offset =  _orbs_offsets(orbs)
    
    atom_blocks_idx = _atom_blocks_idx(frames, orbs_tot)
    
    # init storage for the dense hamiltonians
    dense = []        
    for f in frames:
        norbs = 0
        for ai in f.numbers:
            norbs += orbs_tot[ai]
        ham = np.zeros((norbs, norbs), dtype=np.float64)
        dense.append(ham)

    # loops over block types
    for idx, block in blocks:
        cur_A = -1
        block_type, ai, ni, li, aj, nj, lj = tuple(idx)

        # offset of the orbital block within the pair block in the matrix
        ki_offset = orbs_offset[(ai,ni,li)]
        kj_offset = orbs_offset[(aj,nj,lj)]

        # loops over samples (structure, i, j)
        for (A,i,j), block_data in zip(block.samples, block.values):        
            if A != cur_A:
                ham = dense[A]
                cur_A = A
            # coordinates of the atom block in the matrix
            ki_base, kj_base = atom_blocks_idx[(A,i,j)]
            islice = slice(ki_base+ki_offset, ki_base+ki_offset+2*li+1)
            jslice = slice(kj_base+kj_offset, kj_base+kj_offset+2*lj+1)

            # print(i, ni, li, ki_base, ki_offset)
            if block_type == 0:
                ham[islice, jslice] = block_data[:,:,0].reshape(2*li+1,2*lj+1)
                if ki_offset != kj_offset:
                    ham[jslice, islice] = block_data[:,:,0].reshape(2*li+1,2*lj+1).T
            elif block_type == 2:
                ham[islice, jslice] = block_data[:,:,0].reshape(2*li+1,2*lj+1)
                ham[jslice, islice] = block_data[:,:,0].reshape(2*li+1,2*lj+1).T            
            elif block_type == 1:
                ham[islice, jslice] += np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1)  / np.sqrt(2), dtype=np.float64)
                ham[jslice, islice] += np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1).T / np.sqrt(2), dtype=np.float64)
                if ki_offset != kj_offset:
                    islice = slice(ki_base+kj_offset, ki_base+kj_offset+2*lj+1)
                    jslice = slice(kj_base+ki_offset, kj_base+ki_offset+2*li+1)
                    ham[islice, jslice] += np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1).T  / np.sqrt(2), dtype=np.float64)
                    ham[jslice, islice] += np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1)  / np.sqrt(2), dtype=np.float64)
            elif block_type == -1:
                ham[islice, jslice] += np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1) / np.sqrt(2), dtype=np.float64)
                ham[jslice, islice] += np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1).T / np.sqrt(2), dtype=np.float64)
                if ki_offset != kj_offset:
                    islice = slice(ki_base+kj_offset, ki_base+kj_offset+2*lj+1)
                    jslice = slice(kj_base+ki_offset, kj_base+ki_offset+2*li+1)
                    ham[islice, jslice] -= np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1).T  / np.sqrt(2), dtype=np.float64)
                    ham[jslice, islice] -= np.asarray(block_data[:,:,0].reshape(2*li+1,2*lj+1)  / np.sqrt(2), dtype=np.float64)
    return dense

def couple_blocks(blocks, cg=None):
    if cg is None:
        lmax = max(blocks.sparse["li"]+blocks.sparse["lj"])
        cg = ClebschGordanReal(lmax)

    block_builder = DescriptorBuilder(["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j", "L"], ["structure", "atom_i", "atom_j"], [["M"]], ["value"])
    for idx, block in blocks:
        block_type, ai, ni, li, aj, nj, lj = tuple(idx)
        decoupled = np.moveaxis(np.asarray(block.values, dtype=np.float64),-1,-2).reshape((len(block.samples), len(block.properties), 2*li+1, 2*lj+1))
        coupled = cg.couple(decoupled)[(li,lj)]
        for L in coupled:
            block_idx = tuple(idx) + (L,)
            # skip blocks that are zero because of symmetry
            if ai==aj and ni==nj and li==lj:
                parity = (-1)**(li+lj+L)
                if (parity == -1 and block_type in (0,1)) or (parity==1 and block_type == -1):
                    continue
            new_block = block_builder.add_block(sparse=block_idx, properties=np.asarray([[0]], dtype=np.int32), 
                            components=[_components_idx(L).reshape(-1,1)] )
            new_block.add_samples(labels=block.samples.view(dtype=np.int32).reshape(block.samples.shape[0],-1), 
                            data=np.moveaxis(coupled[L], -1, -2 ) )

    return block_builder.build()

def decouple_blocks(blocks, cg=None):
    if cg is None:
        lmax = max(blocks.sparse["L"])
        cg = ClebschGordanReal(lmax)

    block_builder = DescriptorBuilder(["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"], ["structure", "atom_i", "atom_j"], [["m1"], ["m2"]], ["value"])
    for idx, block in blocks:
        block_type, ai, ni, li, aj, nj, lj, L = tuple(idx)
        block_idx = (block_type, ai, ni, li, aj, nj, lj)
        if block_idx in block_builder.blocks:
            continue        
        coupled = {}
        for L in range(np.abs(li-lj), li+lj+1):
            bidx = blocks.keys.position(block_idx+(L,))
            if bidx is not None:
                coupled[L] = np.moveaxis(blocks.block(bidx).values, -1, -2) 
        decoupled = cg.decouple( {(li,lj):coupled})
        
        new_block = block_builder.add_block(sparse=block_idx, properties=np.asarray([[0]], dtype=np.int32), 
                            components=[_components_idx(li), _components_idx(lj)] )
        new_block.add_samples(labels=block.samples.view(dtype=np.int32).reshape(block.samples.shape[0],-1),
                            data=np.moveaxis(decoupled, 1, -1))
    return block_builder.build()    