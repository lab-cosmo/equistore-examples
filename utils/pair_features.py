import numpy as np
import json
from equistore import Labels, TensorBlock, TensorMap
from utils.builder import DescriptorBuilder
import ase.io
from itertools import product
from utils.clebsh_gordan import ClebschGordanReal
from utils.hamiltonians import fix_pyscf_l1, dense_to_blocks, blocks_to_dense, couple_blocks, decouple_blocks
import matplotlib.pyplot as plt
from utils.librascal import  RascalSphericalExpansion, RascalPairExpansion
from utils.hamiltonians import *
import copy
from rascal.representations import SphericalExpansion


def get_lm_slice(hypers):
    lm_slices = []
    start = 0
    for l in range(hypers["max_angular"] + 1):
        stop = start + 2 * l + 1
        lm_slices.append(slice(start, stop))
        start = stop
    return lm_slices

def rho0ij_builder(hypers, frames):
    if hypers["compute_gradients"]:
        raise Exception("Pair expansion with gradient is not implemented")
    len_frames=[len(f) for f in frames]
    max_atoms = max(len_frames)
    actual_global_species = list(
    map(int, np.unique(np.concatenate([f.numbers for f in frames]))))
    calculator = SphericalExpansion(**hypers)
    manager = calculator.transform(frames)
    info = manager.get_representation_info()
    global_species = list(range(max_atoms))
    
    hypers_ij= copy.deepcopy(hypers)
    hypers_ij["global_species"] = global_species
    hypers_ij["expansion_by_species_method"] = "user defined"
    lm_slices = get_lm_slice(hypers)

    ijframes = []
    for f in frames:
        ijf = f.copy()
        ijf.numbers = global_species[:len(f)]
        ijframes.append(ijf)

    calculator = SphericalExpansion(**hypers_ij)
    gij_expansion=[]
    for ijf in ijframes:
        gij_expansion.append(calculator.transform(ijf).get_features(calculator).reshape(len(ijf), max_atoms, hypers_ij["max_radial"], -1))
#     gij_expansion=calculator.transform(ijframes).get_features(calculator).reshape(len(ijframes), max_atoms, max_atoms, hypers_ij["max_radial"], -1) #TODO: change for differet len
#     print(gij_expansion.shape)
    
    feat_builder= DescriptorBuilder(["block_type", "L", "nu", "sigma",  "species_i", "species_j"], ["structure", "center_i", "center_j"], [["mu"]], ["n"])

    pair_loc=[]
    lmax = hypers["max_angular"]
    for sp_i in actual_global_species:
        for sp_j in actual_global_species:
            center_species_mask = np.where(info[:, 2] == sp_i)[0]
            neighbor_species_mask = np.where(info[:, 2] == sp_j)[0]
            for i, (struct_i, atom_i) in enumerate(info[center_species_mask[:]][...,:-1]):
                for j, (struct_j, atom_j) in enumerate(info[neighbor_species_mask[:]][...,:-1]):
                    ifr=struct_i

                    if not (struct_i==struct_j):
                        continue
                    if atom_i==atom_j:
                        block_type = 0  # diagonal
                    elif sp_i==sp_j:
                        block_type = 1  # same-species
                    elif sp_j > sp_i:
                        block_type = 2  # different species
                    else:
                        continue

                    if [struct_i, atom_j, atom_i] not in pair_loc:
                        pair_loc.append([struct_i, atom_i, atom_j])
                    
                    for l in range(lmax+1):
#                         print(block_type, ifr, l)
                        block_idx=(block_type, l, 0, 1, sp_i, sp_j)
                        if block_idx not in feat_builder.blocks:
                            TensorBlock = feat_builder.add_block(
                                sparse=block_idx, 
                                properties=np.asarray([list(range(hypers["max_radial"]))], dtype=np.int32).T, 
                                components= [np.asarray([list(range(-l, l+1))], dtype=np.int32 ).T] 
                            )

                            if block_type == 1:
                                block_asym = feat_builder.add_block(
                                    sparse=(-1,)+block_idx[1:], 
                                    properties=np.asarray([list(range(hypers["max_radial"]))], dtype=np.int32).T,
                                    components= [np.asarray([list(range(-l, l+1))], dtype=np.int32 ).T]
                                )
                            
                        else:                        
                            TensorBlock = feat_builder.blocks[block_idx]
                            if block_type == 1:
                                block_asym = feat_builder.blocks[(-1,)+block_idx[1:]]

                        block_data =gij_expansion[struct_i][atom_i%len_frames[ifr], atom_j%len_frames[ifr], :, lm_slices[l]].T #TODO: change for not water
                        #POSSIBLE replacement:(atom_i-sum(info[:struct_i,:].axis=1))%len(info[np.where(info[:,:,0]==struct_i)])
                        if block_type == 1:
                            if (atom_i%len_frames[ifr])<=(atom_j%len_frames[ifr]):
                                block_data_ji = gij_expansion[struct_i][atom_j%len_frames[ifr], atom_i%len_frames[ifr], :, lm_slices[l]].T                  
                                TensorBlock.add_samples(labels=np.asarray([[struct_i, atom_i%len_frames[ifr], atom_j%len_frames[ifr]]], dtype=np.int32), data=(block_data+block_data_ji).reshape((1,-1,block_data.shape[1]))/np.sqrt(2) )
                                block_asym.add_samples(labels=np.asarray([[struct_i, atom_i%len_frames[ifr], atom_j%len_frames[ifr]]], dtype=np.int32), data=(block_data-block_data_ji).reshape((1,-1,block_data.shape[1]))/np.sqrt(2) )

                        else:
#                             print(block_data.shape, [struct_i, atom_i%len_frames[ifr], atom_j%len_frames[ifr]])
                            TensorBlock.add_samples(labels=np.asarray([[struct_i, atom_i%len_frames[ifr], atom_j%len_frames[ifr]]], dtype=np.int32), data=block_data.reshape(1, -1, block_data.shape[1]))
    #                     
    return feat_builder.build()

def tensor_g_rho_nu(rho0ij, rhoinu, hypers, cg, property_names=None):
    """ rho_ij^{nu+1} = <q|rho_i^{nu+1}; kh> <n| rho_ij^{0}; lm> cg 
    feature_names is a tuple of the form <n_rho0ij, l_rho0ij, n's in rhoi, k|
    Make sure you have transferred the species label to the feature (sparse_to_properties)
    """
    nu_ij=rho0ij.keys["nu"][0]
    
    # rhoinu = acdc_nu1; 
    property_names=None
    nu_ij=rho0ij.keys["nu"][0] #should be 0
    nu_rho= rhoinu.keys["nu"][0]
    NU = nu_rho+nu_ij

    lmax= hypers["max_angular"]

    if cg is None:
        cg = ClebschGordanReal(lmax)

    sparse_labels=copy.deepcopy(rho0ij.keys)
    sparse_labels["nu"] = NU

    new_sparse_labels = []
    for i in sparse_labels:
        new_sparse_labels.append(tuple(i))
        i[3]*=-1
        if i not in new_sparse_labels:
            new_sparse_labels.append(tuple(i))

    X_blocks = {new_sparse_labels[i]:[] for i in range(len(new_sparse_labels))}
    X_idx = {new_sparse_labels[i]:[] for i in range(len(new_sparse_labels))}
    X_samples= {new_sparse_labels[i]:[] for i in range(len(new_sparse_labels))}


    if property_names is None:
        property_names = (
            tuple(n + "_a" for n in rho0ij.block(0).properties.names)
            + ("l_" + str(NU),)
            + ("q_" + str(NU),) #for n in rhoinu.block(0).features.names)
            + ("k_" + str(NU),)
            )

    for index_a, block_a in rho0ij:
        block_type = index_a["block_type"]
        lam_a = index_a["L"]
        sigma_a = index_a["sigma"]
        sp_i, sp_j = index_a["species_i"], index_a["species_j"]
        for index_b, block_b in rhoinu:
            lam_b = index_b["L"]
            sigma_b = index_b["sigma"]
            rho_sp_i= index_b["species_i"]
            if not(sp_i== rho_sp_i):
                continue
            for L in range(np.abs(lam_a - lam_b),  min(lam_a + lam_b, lmax)+1):
                S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                block_idx=(block_type, L, NU, S, sp_i, sp_j)
                sel_feats=[]
                for n_a in range(len(block_a.properties)):
                    f_a = tuple(block_a.properties[n_a]) #values of n_a'th feature in block_a.features
                    for n_b in range(len(block_b.properties)):
                        f_b = tuple(block_b.properties[n_b]) #values of n_b'th feature in block_b.features
                        IDX = f_a  + (lam_a,)+ (''.join(map(str, f_b)),) + (lam_b,)
                        #IDX = f_a  + (lam_a,)+f_b + (lam_b,)
                        sel_feats.append([n_a, n_b])
                        X_idx[block_idx].append(IDX)

                sel_feats = np.asarray(sel_feats, dtype=int)
                if len(sel_feats) == 0:
                    print(IDX, L, "len_feats 0")
                    continue

    #             #REMEMBER- values.shape = (nstruct*nat fitting the block criteria, 2*l+1, featsize)
                if block_type==0:
                    if not(sp_i== rho_sp_i):
                        continue
                    one_shot_blocks = cg.combine_einsum(
                     block_a.values[:, :, sel_feats[:, 0]],  #sel_feats[:,0]= n_a
                     block_b.values[:, :, sel_feats[:, 1]],  #sel_feats[:,1]= n_b*nspecies
                    L,
                    combination_string="iq,iq->iq",
                ) 

                    samples = block_a.samples
                    
                else:
                    xx=[]
                    yy=[]
                    for samplea in block_a.samples:
                        centeri = samplea['center_i']
                        centerj = samplea['center_j']
                        stra = samplea['structure']
                        idxb= block_b.samples[np.where(block_b.samples['center']==centeri)]
                        samples_idxb=block_b.samples[np.where(np.in1d(block_b.samples,idxb))[0]]
                        xx.append(samples_idxb[np.where(samples_idxb['structure']==stra)[0]])
                        if (block_type==1 or block_type==-1):
                            idxbj= block_b.samples[np.where(block_b.samples['center']==centerj)]
                            samples_idxbj=block_b.samples[np.where(np.in1d(block_b.samples,idxbj))[0]]
                            yy.append(samples_idxbj[np.where(samples_idxbj['structure']==stra)[0]])
                    #     yy.append()
                    ixx=[np.where(i==block_b.samples)[0][0] for i in xx]
                    iyy=[np.where(i==block_b.samples)[0][0] for i in yy]
                    rhoinuvalues = block_b.values[ixx]
                    rhojnuvalues = block_b.values[iyy]

                    if (block_type==1 or block_type==-1): 
                        if not(sp_i== rho_sp_i):
                            continue
                        one_shot_blocks_ij = cg.combine_einsum(
                         block_a.values[:, :, sel_feats[:, 0]], 
                         rhoinuvalues[:, :, sel_feats[:, 1]],
                        L,
                        combination_string="iq,iq->iq",
                    )

                        one_shot_blocks_ji = cg.combine_einsum(
                         block_a.values[:, :, sel_feats[:, 0]], 
                         rhojnuvalues[:, :, sel_feats[:, 1]],
                        L,
                        combination_string="iq,iq->iq",
                    )
                        samples = block_a.samples
                        
                        if block_type==1:
                            one_shot_blocks = (one_shot_blocks_ij+one_shot_blocks_ji)/np.sqrt(2)
                        elif block_type==-1:
                            one_shot_blocks = (one_shot_blocks_ij-one_shot_blocks_ji)/np.sqrt(2)


#                    elif block_type==-1:
#                        if not(sp_i== rho_sp_i):
#                            continue
#                        one_shot_blocks_ij = cg.combine_einsum(
#                         block_a.values[:, :, sel_feats[:, 0]], 
#                         rhoinuvalues[:, :, sel_feats[:, 1]],
#                        L,
#                        combination_string="iq,iq->iq",
#                    )
#
#                        one_shot_blocks_ji = cg.combine_einsum(
#                         block_a.values[:, :, sel_feats[:, 0]], 
#                         rhojnuvalues[:, :, sel_feats[:, 1]],
#                        L,
#                        combination_string="iq,iq->iq",
#                    )
#                        samples = block_a.samples
#                        one_shot_blocks = (one_shot_blocks_ij-one_shot_blocks_ji)/np.sqrt(2)

                    elif block_type==2:
                        #TODO: recheck this 
        #                     if sp_i<rho_sp_i:
        #                         continue
                        if not(sp_i== rho_sp_i):
                            continue
                        
                        #print(sp_i, rho_sp_i, sp_j, block_a.values.shape, block_b.values.shape)
                        one_shot_blocks = cg.combine_einsum(
                         block_a.values[:, :, sel_feats[:, 0]], 
                         rhoinuvalues[:, :, sel_feats[:, 1]],
                        L,
                        combination_string="iq,iq->iq",
                    )
                        samples = block_a.samples

                for Q in range(len(sel_feats)):
                    (n_a, n_b) = sel_feats[Q]
                    n=block_b.properties[n_b]
                    IDX = (n_a,)  + (lam_a,)+(''.join(map(str, n)),) + (lam_b,)
                    newblock = one_shot_blocks[:, :, Q]
   
                    X_blocks[block_idx].append(newblock)
                    X_samples[block_idx].append(samples)
    nonzero_idx = []

    nonzero_blocks = []
    for block_idx in X_blocks:
        block_type, L, NU, S, sp_i, sp_j = block_idx
        # create blocks
        if len(X_blocks[block_idx]) == 0:
            #print(block_idx, "skipped")
            continue  # skips empty blocks

        nonzero_idx.append(block_idx)
        block_data = np.moveaxis(np.asarray(X_blocks[block_idx]), 0, -1) 
        block_samples = X_samples[block_idx][0]
        newblock = TensorBlock(
            values=block_data,
            samples=block_samples,
            components=[Labels(
                ["mu"], np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1)
            )],
            properties=Labels(property_names, np.asarray(X_idx[block_idx], dtype=np.int32)),
        )

        nonzero_blocks.append(newblock)
        print(block_idx, 'done')

    X = TensorMap(
        Labels(rho0ij.keys.names, np.asarray(nonzero_idx, dtype=np.int32)), nonzero_blocks
    )


    return X

