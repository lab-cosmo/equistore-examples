from .builder import DescriptorBuilder
import numpy as np

###########  I/O UTILITIES ##############
def fix_pyscf_l1(fock, frame, orbs):
    """ pyscf stores l=1 terms in a xyz order, corresponding to (m=0, 1, -1).
        this converts into a canonical form where m is sorted as (-1, 0,1) """
    idx = []
    iorb = 0;
    atoms = list(frame.symbols)
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
    return fock[idx][:,idx]

############ matrix/block manipulations ###############

def _components_idx(li, lj):
    return np.array(np.meshgrid(np.arange(-li,li+1), np.arange(-lj,lj+1)), dtype = np.int32).T.reshape(-1,2)

def dense_to_blocks(dense, frames, orbs):
    ki, kj = 0, 0
    ham_builder = DescriptorBuilder(["block_type", "a1", "n1", "l1", "a2", "n2", "l2"], ["structure", "atom_i", "atom_j", "ki_base", "kj_base"], ["m1", "m2"], ["hamiltonian"])
    for A in range(len(frames)):
        frame = frames[A]
        ham = dense[A]
        ki = 0
        for i, ai in enumerate(frame.numbers):
            ki_base = ki # pointer at where the i-atom block starts
            for ni, li, mi in orbs[ai]:
                kj = 0
                if mi != -li:
                    continue
                for j, aj in enumerate(frame.numbers):
                    if i<j: # operate only on the lower-triangular block
                        continue                    
                    if i==j:
                        block_type = 0  # diagonal
                    elif ai==aj:
                        block_type = 1  # same-species
                    else:
                        block_type = 2  # different species
                    kj_base = kj # pointer at where the j-atom block starts
                    for nj, lj, mj in orbs[aj]:                
                        if mj != -lj:
                            continue
                        block_idx = (block_type, ai, ni, li, aj, nj, lj)
                        if block_idx not in ham_builder.blocks:
                            block = ham_builder.add_block(sparse=block_idx, features=np.asarray([[0]], dtype=np.int32), 
                                components=_components_idx(li,lj) )

                            if block_type == 1:
                                block_asym = ham_builder.add_block(sparse=(-1,)+block_idx[1:], features=np.asarray([[0]], dtype=np.int32), 
                                components=_components_idx(li,lj) )
                        else:                        
                            block = ham_builder.blocks[block_idx]
                            if block_type == 1:
                                block_asym = ham_builder.blocks[(-1,)+block_idx[1:]]
                        
                        block_data_ij = np.asarray(ham[ki:ki+2*li+1, kj:kj+2*lj+1])

                        if block_type == 1:
                            kj_offset = kj-kj_base
                            ki_offset = ki-ki_base
                            block_data_ji = np.asarray(ham[kj_base+ki_offset:kj_base+ki_offset+2*li+1, ki_base+kj_offset:ki_base+kj_offset+2*lj+1])                        
                            block.add_samples(labels=[(A,i,j, ki_base, kj_base)], data=(block_data_ij+block_data_ji).reshape((1,-1,1))/np.sqrt(2) )
                            block_asym.add_samples(labels=[(A,i,j,ki_base, kj_base)], data=(block_data_ij-block_data_ji).reshape((1,-1,1))/np.sqrt(2) )
                        else:
                            block.add_samples(labels=[(A,i,j,ki_base, kj_base)], data=block_data_ij.reshape((1,-1,1)))                    
                        kj += 2*lj+1
                ki += 2*li+1
    return ham_builder.build()

def blocks_to_dense(blocks, frames, orbs):
    dense = []    
    for f in frames:
        norbs = 0
        for ai in f.numbers:        
            norbs += len(orbs[ai])
        ham = np.zeros((norbs, norbs), dtype=np.float64)
        dense.append(ham)

    for idx, block in blocks:
        cur_A = -1
        block_type, ai, ni, li, aj, nj, lj = tuple(idx)
        ki_offset = 0
        for no, lo, mo in orbs[ai]:        
            if no == ni and lo == li:
                break
            ki_offset += 1
        kj_offset = 0
        for no, lo, mo in orbs[aj]:        
            if no == nj and lo == lj:
                break
            kj_offset += 1
        for (A,i,j,ki_base,kj_base), block_data in zip(block.samples, block.values):
            if A != cur_A:
                ham = dense[A]
                cur_A = A
            if block_type == 0:
                ham[ki_base+ki_offset:ki_base+ki_offset+2*li+1, kj_base+kj_offset:kj_base+kj_offset+2*lj+1] = block_data[:,0].reshape(2*li+1,2*lj+1)
            elif block_type == 2:
                ham[ki_base+ki_offset:ki_base+ki_offset+2*li+1, kj_base+kj_offset:kj_base+kj_offset+2*lj+1] = block_data[:,0].reshape(2*li+1,2*lj+1)
                ham[kj_base+kj_offset:kj_base+kj_offset+2*lj+1, ki_base+ki_offset:ki_base+ki_offset+2*li+1] = block_data[:,0].reshape(2*li+1,2*lj+1).T
            elif block_type == 1:
                ham[ki_base+ki_offset:ki_base+ki_offset+2*li+1, kj_base+kj_offset:kj_base+kj_offset+2*lj+1] += np.asarray(block_data[:,0].reshape(2*li+1,2*lj+1)  / np.sqrt(2), dtype=np.float64)
                ham[kj_base+ki_offset:kj_base+ki_offset+2*li+1, ki_base+kj_offset:ki_base+kj_offset+2*lj+1] += np.asarray(block_data[:,0].reshape(2*li+1,2*lj+1) / np.sqrt(2), dtype=np.float64)
            elif block_type == -1:
                ham[ki_base+ki_offset:ki_base+ki_offset+2*li+1, kj_base+kj_offset:kj_base+kj_offset+2*lj+1] += np.asarray(block_data[:,0].reshape(2*li+1,2*lj+1) / np.sqrt(2), dtype=np.float64)
                ham[kj_base+ki_offset:kj_base+ki_offset+2*li+1, ki_base+kj_offset:ki_base+kj_offset+2*lj+1] -= np.asarray(block_data[:,0].reshape(2*li+1,2*lj+1) / np.sqrt(2), dtype=np.float64)
    return dense