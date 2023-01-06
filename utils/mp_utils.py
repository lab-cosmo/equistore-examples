import numpy as np
from equistore import Labels, TensorBlock, TensorMap
from itertools import product
from utils.acdc_mini import acdc_standardize_keys, cg_increment, cg_combine, _remove_suffix
from utils.clebsh_gordan import ClebschGordanReal
from utils.librascal import  RascalSphericalExpansion, RascalPairExpansion
from rascal.representations import SphericalExpansion

def contract_three_center_property(Yii1i2, numpy = True):
    if len(Yii1i2.keys.dtype)>4: #i.e. not a standard acdc tensormap 
        Yii1i2.keys_to_properties('species_neighbor_b')
        Yii1i2.keys_to_properties('species_neighbor_a')
    contracted_Yii1i2_blocks=[]

    property_names = Yii1i2.property_names
    for key,block in Yii1i2:
        contract_values=[]
        contract_properties=[]
        contract_samples=list(product(np.unique(block.samples['structure']), np.unique(block.samples['center'])))
        for isample, sample in enumerate(contract_samples):
            sample_idx = [idx for idx, tup in enumerate(block.samples) if tup[0]==sample[0] and tup[1]==sample[1]]
            contract_values.append(block.values[sample_idx].sum(axis=0)) # sum i_1, i_2
#             print(key, block.values[sample_idx].sum(axis=0).shape,sample,sample_idx)
#         print(key, len(contract_values), torch.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1).shape)
        if numpy: 
#             print("contract 3 center", np.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1).shape)
            new_block = TensorBlock(values = np.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1),
                                        samples = Labels(['structure', 'center'], np.asarray(contract_samples, np.int32)), 
                                         components = block.components,
                                         properties= Labels(list(property_names), block.properties.asarray())
                                )
        else:
#             print("contract 3 center", torch.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1).shape)
            
            new_block = TensorBlock(values = torch.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1),
                                #eshape(len(contract_samples),contract_values[0].shape[0] ,-1),
                                        samples = Labels(['structure', 'center'], np.asarray(contract_samples, np.int32)), 
                                         components = block.components,
                                         properties= Labels(list(property_names), block.properties.asarray())
                                )

        contracted_Yii1i2_blocks.append(new_block)

    contracted_Yii1i2 = TensorMap(Yii1i2.keys, contracted_Yii1i2_blocks)
    return contracted_Yii1i2


def atom_to_structure(X, numpy = True):
    # go from rhoi to structure 
    #ASSUMING - homogeneous dataset - all frames have same molecule
    struct_blocks=[]
    ele = np.unique(X.keys['species_center'])
    final_keys = list(set([tuple(list(x)[:-1]) for x in X.keys]))
    for key, block in X:
        contract_values=[]
        contract_properties=[]
        contract_samples = np.asarray(np.unique(block.samples['structure'])).reshape(-1,1)
        for isample, sample in enumerate(contract_samples):
            sample_idx = [idx for idx, tup in enumerate(block.samples) if tup[0]==sample]
            contract_values.append(block.values[sample_idx].sum(axis=0))
#             print(key, sample_idx)
        if numpy: 
            new_block = TensorBlock(values = np.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1),
                                samples = Labels(['structure'], np.asarray(contract_samples, np.int32)), 
                                components = block.components,
                                properties= block.properties
                                    )
        else:
#             print("structure", torch.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1).shape)
            new_block = TensorBlock(values = torch.vstack(contract_values).reshape(len(contract_samples),contract_values[0].shape[0] ,-1),
                                samples = Labels(['structure'], np.asarray(contract_samples, np.int32)), 
                                components = block.components,
                                properties= block.properties
                                    )
        struct_blocks.append(new_block)

    struct_X = TensorMap(X.keys, struct_blocks)


    all_structures = np.unique(np.concatenate([block.samples for (k,block) in struct_X]))
    values = torch.zeros((len(all_structures), 1, len(struct_X.block(0).properties)))
    for species in ele: 
        values += struct_X.block(species_center = species).values

    final_block = TensorBlock(values = values,
                        samples = Labels(['structure'], np.asarray(all_structures, np.int32).reshape(-1,1)), 
                        components = [Labels(['spherical_component_m'], np.asarray([[0]], np.int32))],
                        properties= struct_X.block(0).properties
               )
    struct_X = TensorMap(Labels(['feat'], np.asarray([[0]], np.int32)), [final_block])
    return struct_X

def compare_with_rho2i(contracted_rhoii1i2, rho2i):
    assert len(rho2i) == len(contracted_rhoii1i2)
    for rho2_k, rho2_b in rho2i:
        contracted_block = contracted_rhoii1i2.block(order_nu = rho2_k[0], inversion_sigma = rho2_k[1], spherical_harmonics_l = rho2_k[2], species_center = rho2_k[3])
        idx = []
        cidx = []
        for i,p in enumerate(rho2_b.properties):
            find_p = (p['species_neighbor_a'], p['species_neighbor_b'], p['n_1_a'], p['k_2'], p['n_1_b'], p['l_2'])
        #     print(p, find_p)
            for ip,cp in enumerate(contracted_block.properties):
                if tuple(find_p) == tuple(cp) :
                    idx.append(i)
                    cidx.append(ip)
                    break
        print(rho2_k, np.linalg.norm(rho2_b.values[:,:,idx] - contracted_block.values[:,:,cidx]))

        
def relabel_key_contract(tensormap):
    """ Relabel the key to contract with other_keys_match, for ACDC - 'species_center' gets renamed to 'species_contract'
    while for N-center ACDC 'specoes_neighbor' gets renamed to 'species_contract'  """
    new_tensor_blocks = []
    new_tensor_keys = [] 
    for k,b in tensormap:
        key = tuple(k)
        block = TensorBlock(values=b.values,
                        samples=b.samples,
                        components=b.components,
                        properties=b.properties
                           )
        new_tensor_blocks.append(block)
        new_tensor_keys.append(key)
    if 'species_neighbor' in tensormap.keys.dtype.names:
        #Relabel neighbor species as species_contract to be the channel to contract |rho_j> |g_ij>
        new_tensor_keys = Labels(('order_nu', 'inversion_sigma', 'spherical_harmonics_l', 'species_center','species_contract'), np.asarray(new_tensor_keys, dtype=np.int32))
    else:
        #Relabel center species as species_contract to be the channel to contract |rho_j>  
        new_tensor_keys = Labels(('order_nu', 'inversion_sigma', 'spherical_harmonics_l','species_contract'), np.asarray(new_tensor_keys, dtype=np.int32))

    new_tensormap = TensorMap(new_tensor_keys, new_tensor_blocks)
    return new_tensormap



def contract_rho_ij(rhoijp, elements, property_names=None):
    """ contract the doubly decorated pair feature rhoijp = |rho_{ij}^{[nu, nu']}> to return  |rho_{i}^{[nu <- nu']}> """
    rhoMPi_keys = list(set([tuple(list(x)[:-1]) for x in rhoijp.keys]))
    rhoMPi_blocks = []
    if property_names==None:
        property_names = rhoijp.property_names

    for key in rhoMPi_keys:
        contract_blocks=[]
        contract_properties=[]
        contract_samples=[]#rho1i.block(rho1i.blocks_matching(species_center=key[-1])[0]).samples #samples for corres key

        for ele in elements:
            blockidx = rhoijp.blocks_matching(species_contract= ele)
            sel_blocks = [rhoijp.block(i) for i in blockidx if key==tuple(list(rhoijp.keys[i])[:-1])]
            if not len(sel_blocks):
#                 print(key, ele, "skipped")
                continue
            assert len(sel_blocks)==1 #sel_blocks is the corresponding rho11 block with the same key and species_contract = ele
            block = sel_blocks[0]
            filter_idx = list(zip(block.samples['structure'], block.samples['center']))
    #             #len(block.samples)==len(filter_idx)
            struct, center = np.unique(block.samples['structure']), np.unique(block.samples['center'])
            possible_block_samples = list(product(struct,center))

            block_samples=[]
            ij_samples=[]
            block_values = []

            for isample, sample in enumerate(possible_block_samples):
                sample_idx = [idx for idx, tup in enumerate(filter_idx) if tup[0] ==sample[0] and tup[1] == sample[1]]
                if len(sample_idx)==0:
                    continue
    #             #print(key, ele, sample, block.samples[sample_idx])
                block_samples.append(sample)
                ij_samples.append(block.samples[sample_idx])
                block_values.append(block.values[sample_idx].sum(axis=0)) #sum j belonging to ele, 
                #block_values has as many entries as samples satisfying (key, ele) so in general we have a ragged list
                #of contract_blocks

            contract_blocks.append(block_values)
            contract_samples.append(block_samples)
            contract_properties.append(block.properties.asarray())

        all_block_samples= sorted(list(set().union(*contract_samples))) 
#         print('nsamples',len(all_block_samples) )
        all_block_values = np.zeros(((len(all_block_samples),)+ block.values.shape[1:]+(len(contract_blocks),)))
        for ib, bb in enumerate(contract_samples):
            nzidx=[i for i in range(len(all_block_samples)) if all_block_samples[i] in bb]
#             print(elements[ib],key, bb, all_block_samples)
            all_block_values[nzidx,:,:,ib] = contract_blocks[ib]

        new_block = TensorBlock(values = all_block_values.reshape(all_block_values.shape[0],all_block_values.shape[1] ,-1),
                                        samples = Labels(['structure', 'center'], np.asarray(all_block_samples, np.int32)), 
                                         components = block.components,
                                         properties= Labels(list(property_names), np.asarray(np.vstack(contract_properties),np.int32))
                                         )

        rhoMPi_blocks.append(new_block)
    rhoMPi = TensorMap(Labels(['order_nu','inversion_sigma','spherical_harmonics_l','species_center'],np.asarray(rhoMPi_keys,                         dtype=np.int32)), rhoMPi_blocks)

    return rhoMPi


def flatten(x):
    #works for tuples of the form ((a,b,c), d) to (a,b,c,d)
    if isinstance(x, tuple):
        return tuple(x[0])+(x[1],)
    elif isinstance(x, list):
        #list of tuples
        flat_list_tuples=[]
        for aa in x:
            flat_list_tuples.append(flatten(aa))
        return flat_list_tuples

def cg_combine(
    x_a,
    x_b,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
    filter_sigma = [-1,1],
    other_keys_match=None,
    mp=False
):
    """
    modified cg_combine from acdc_mini.py to add the MP contraction, that contracts over NOT the center but the neighbor yielding |rho_j> |g_ij>, can be merged   
    """

    # determines the cutoff in the new features
    lmax_a = max(x_a.keys["spherical_harmonics_l"])
    lmax_b = max(x_b.keys["spherical_harmonics_l"])
    if lcut is None:
        lcut = lmax_a + lmax_b

    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut) 

    similar=True
    if "neighbor" in x_b.sample_names: #and "neighbor" not in x_a.sample_names:
        #similar only when combining two rho1i's (not rho1i with gij or |r_ij> with |r_ik>)
        similar = False

    other_keys_a = tuple(name for name in x_a.keys.names if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"] )
    other_keys_b = tuple(name for name in x_b.keys.names if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"] )
    if mp: 

        if other_keys_match is None:
            OTHER_KEYS = [ k+"_a" for k in other_keys_a ] + [ k+"_b" for k in other_keys_b ]
        else:     
            OTHER_KEYS =  [ 
                k+("_a" if k in other_keys_b else "") for k in other_keys_a if k not in other_keys_match ] + [
                k+("_b" if k in other_keys_a else "") for k in other_keys_b if k not in other_keys_match ]  +other_keys_match    
    else: 
        if other_keys_match is None:
            OTHER_KEYS = [ k+"_a" for k in other_keys_a ] + [ k+"_b" for k in other_keys_b ]
        else:     
            OTHER_KEYS = other_keys_match + [ 
                k+("_a" if k in other_keys_b else "") for k in other_keys_a if k not in other_keys_match ] + [
                k+("_b" if k in other_keys_a else "") for k in other_keys_b if k not in other_keys_match ]  

    if x_a.block(0).has_gradient("positions"):
        grad_components = x_a.block(0).gradient("positions").components
    else:
        grad_components = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        NU = x_a.keys[0]["order_nu"] + x_b.keys[0]["order_nu"]
        feature_names = (
            tuple(n + "_a" for n in x_a.property_names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.property_names)
            + ("l_" + str(NU),)
        )

    X_idx = {}
    X_blocks = {}
    X_samples = {}
    X_grad_samples = {}
    X_grads = {}

    for index_a, block_a in x_a:
        lam_a = index_a["spherical_harmonics_l"]
        sigma_a = index_a["inversion_sigma"]
        order_a = index_a["order_nu"]                
        properties_a = block_a.properties  # pre-extract this block as accessing a c property has a non-zero cost
        samples_a = block_a.samples
        for index_b, block_b in x_b:
            lam_b = index_b["spherical_harmonics_l"]
            sigma_b = index_b["inversion_sigma"]
            order_b = index_b["order_nu"]       
            properties_b = block_b.properties
            samples_b = block_b.samples
            samples_final = samples_b
            b_slice = list(range(len(samples_b)))
            if similar and lam_b<lam_a:
                continue

            if other_keys_match is None:            
                OTHERS = tuple( index_a[name] for name in other_keys_a ) + tuple( index_b[name] for name in other_keys_b )
            else:
                OTHERS = tuple(index_a[k] for k in other_keys_match if index_a[k]==index_b[k])
                if len(OTHERS)<len(other_keys_match):
                    continue
                # adds non-matching keys to build outer product
                if mp: 

                    OTHERS = tuple(index_a[k] for k in other_keys_a if k not in other_keys_match) + OTHERS 
                    OTHERS = tuple(index_b[k] for k in other_keys_b if k not in other_keys_match) + OTHERS 
                else: 
                    OTHERS = OTHERS + tuple(index_a[k] for k in other_keys_a if k not in other_keys_match)
                    OTHERS = OTHERS + tuple(index_b[k] for k in other_keys_b if k not in other_keys_match)

            if mp: 
                if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                    center_slice = []
                    smp_a, smp_b = 0, 0
                    while smp_b < samples_b.shape[0]:               
                        #print(index_b, samples_b[smp_b][["structure", "center", "neighbor"]], index_a, samples_a[smp_a])
                        idx= [idx for idx, tup in enumerate(samples_a) if tup[0] ==samples_b[smp_b]["structure"] and tup[1] == samples_b[smp_b]["neighbor"] ][0]
                        center_slice.append(idx)
                        smp_b+=1
                    center_slice = np.asarray(center_slice)
    #                     print(index_a, index_b, center_slice,  block_a.samples, block_b.samples)
                else: 
                    center_slice = slice(None)
            else:
                if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:

                    neighbor_slice = []
                    smp_a, smp_b = 0, 0
                    while smp_b < samples_b.shape[0]:                    
                        if samples_b[smp_b][["structure","center"]] != samples_a[smp_a]:
                            if(smp_a+1 < samples_a.shape[0]):
                                smp_a+=1
                        neighbor_slice.append(smp_a)
                        smp_b+=1
                    neighbor_slice = np.asarray(neighbor_slice)
                    print(index_a, index_b, neighbor_slice,  block_a.samples[neighbor_slice], block_b.samples)

                elif "neighbor" in samples_b.names and "neighbor" in samples_a.names:
                    #taking tensor products of gij and gik
                    neighbor_slice = []
                    b_slice = []
                    samples_final = []
                    smp_a, smp_b = 0, 0
                    """
                    while smp_b < samples_b.shape[0]:
                        idx= [idx for idx, tup in enumerate(samples_a) if tup[0] ==samples_b[smp_b]["structure"] and tup[1] == samples_b[smp_b]["center"]]
                        neighbor_slice.extend(idx)
                        b_slice.extend([smp_b]*len(idx))
                        samples_final.extend(flatten(list(product([samples_b[smp_b]],block_a.samples.asarray()[idx][:,-1]))))
                        smp_b+=1
                    """                    
                    sc_b = (-1,-1)
                    while smp_b < samples_b.shape[0]:    
                        # checks if structure index needs updating
                        if samples_b[smp_b]["center"] != sc_b[1] or samples_b[smp_b]["structure"] != sc_b[0]:  
                            # checks if structure index needs updating
                            sc_b = samples_b[smp_b][["structure", "center"]]
                            idx = np.where( (samples_b[smp_b]["structure"] == samples_a["structure"] ) & 
                                           (samples_b[smp_b]["center"] == samples_a["center"] ))[0]
                            smp_a_idx = samples_a["neighbor"][idx].view(np.int32)
                        neighbor_slice.extend(idx)    
                        b_slice.extend([smp_b]*len(idx))
                        #samples_final.extend(flatten(list(product([samples_b[smp_b]],smp_a_idx))))
                        #samples_final.extend(np.hstack([[tuple(samples_b[smp_b-1])]*8, smp_a_idx[:,np.newaxis] ]) )
                        samples_final.extend([tuple(samples_b[smp_b]) + (idx,) for idx in smp_a_idx])
                        smp_b+=1
                    neighbor_slice = np.asarray(neighbor_slice)
    #                 print(index_a, index_b, neighbor_slice)#,  block_a.samples[neighbor_slice], block_b.samples)
                    samples_final = Labels(["structure", "center", "neighbor_1", "neighbor_2"], np.asarray(samples_final, dtype=np.int32))                        
                elif "neighbor_1" in samples_b.names: 
                    # combining three center feature with rho_{i i1 i2}
                    neighbor_slice = []
                    b_slice = []
                    smp_a, smp_b = 0, 0
                    """
                    while smp_b < samples_b.shape[0]:
                        idx= [idx for idx, tup in enumerate(samples_a) if tup[0] ==samples_b[smp_b]["structure"] and tup[1] == samples_b[smp_b]["center"]]
                        neighbor_slice.extend(idx)
                        b_slice.extend([smp_b]*len(idx))
                        smp_b+=1
                    """
                    sc_b = (-1,-1)
                    while smp_b < samples_b.shape[0]:    
                        # checks if structure index needs updating
                        if samples_b[smp_b]["center"] != sc_b[1] or samples_b[smp_b]["structure"] != sc_b[0]:  
                            # checks if structure index needs updating
                            sc_b = samples_b[smp_b][["structure", "center"]]
                            idx = np.where( (samples_b[smp_b]["structure"] == samples_a["structure"] ) & 
                                           (samples_b[smp_b]["center"] == samples_a["center"] ))[0]
                        neighbor_slice.extend(idx)    
                        b_slice.extend([smp_b]*len(idx))
                        smp_b+=1
                    neighbor_slice = np.asarray(neighbor_slice)
#                     print(samples_b[b_slice], samples_a[neighbor_slice])
                
                else:
                    neighbor_slice = slice(None) 

            # determines the properties that are in the select list  
            sel_feats = []
            sel_idx = []
            sel_feats = np.indices((len(properties_a), len(properties_b))).reshape(2,-1).T

            prop_ids_a = []
            prop_ids_b = []
            for n_a, f_a in enumerate(properties_a):
                prop_ids_a.append( tuple(f_a) + (lam_a,))
            for n_b, f_b in enumerate(properties_b):
                prop_ids_b.append( tuple(f_b) + (lam_b,))
            prop_ids_a = np.asarray(prop_ids_a) 
            prop_ids_b = np.asarray(prop_ids_b)
            sel_idx = np.hstack([prop_ids_a[sel_feats[:,0]],prop_ids_b[sel_feats[:,1]] ])    #creating a tensor product          
            if len(sel_feats) == 0:
                continue            
            # loops over all permissible output blocks. note that blocks will
            # be filled from different la, lb
            for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                # determines parity of the block
                S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                if not S in filter_sigma:
                    continue
                NU = order_a + order_b                
                KEY = (NU, S, L,) + OTHERS
                if not KEY in X_idx:
                    X_idx[KEY] = []
                    X_blocks[KEY] = []
                    X_samples[KEY] = samples_final
                    if grad_components is not None:
                        X_grads[KEY] = []  
                        X_grad_samples[KEY] = block_b.gradient("positions").samples

                # builds all products in one go
                if mp:
                    if isinstance(center_slice,slice) or  len(center_slice):
                        one_shot_blocks = clebsch_gordan.combine_einsum(
                            block_a.values[center_slice][:, :, sel_feats[:, 0]],
                            block_b.values[:, :, sel_feats[:, 1]],
                            L,
                            combination_string="iq,iq->iq",
                        )

                        if grad_components is not None: 
                            raise ValueError("grads not implemented with MP") 
                    else:
                        one_shot_blocks = []

                else: 
                    if isinstance(neighbor_slice,slice) or  len(neighbor_slice) :
                        one_shot_blocks = clebsch_gordan.combine_einsum(
                        block_a.values[neighbor_slice][:, :, sel_feats[:, 0]],
                        block_b.values[b_slice][:, :, sel_feats[:, 1]],
                        L,
                        combination_string="iq,iq->iq",
                    )

                        if grad_components is not None:
                            grad_a = block_a.gradient("positions")
                            grad_b = block_b.gradient("positions")
                            grad_a_data = np.swapaxes(grad_a.data, 1,2)
                            grad_b_data = np.swapaxes(grad_b.data, 1,2)
                            one_shot_grads = clebsch_gordan.combine_einsum(
                                block_a.values[grad_a.samples["sample"]][neighbor_slice, :, sel_feats[:, 0]],
                                grad_b_data[b_slice][..., sel_feats[:, 1]],
                                L=L,
                                combination_string="iq,iaq->iaq",
                            ) + clebsch_gordan.combine_einsum(
                                block_b.values[grad_b.samples["sample"]][b_slice][:, :, sel_feats[:, 1]],
                                grad_a_data[neighbor_slice, ..., sel_feats[:, 0]],
                                L=L,
                                combination_string="iq,iaq->iaq",
                            )
                    else:
                        one_shot_blocks = []



                # now loop over the selected features to build the blocks

                X_idx[KEY].append(sel_idx)
                if len(one_shot_blocks):
                    X_blocks[KEY].append(one_shot_blocks)
                if grad_components is not None:
                    X_grads[KEY].append(one_shot_grads)

    # turns data into sparse storage format (and dumps any empty block in the process)
    nz_idx = []
    nz_blk = []
    for KEY in X_blocks:
        L = KEY[2]
        # create blocks
        if len(X_blocks[KEY]) == 0:
            continue  # skips empty blocks
        nz_idx.append(KEY)
    #         print(KEY, X_samples[KEY], len(X_blocks[KEY]) , X_blocks[KEY][0])
        block_data = np.concatenate(X_blocks[KEY], axis=-1)
        sph_components = Labels(
                ["spherical_harmonics_m"], np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1)
            )
        newblock = TensorBlock(
            values=block_data,
            samples=X_samples[KEY],
            components=[sph_components],
            properties=Labels(feature_names, np.asarray(np.vstack(X_idx[KEY]), dtype=np.int32)),
        )

        nz_blk.append(newblock)
    X = TensorMap(
        Labels(["order_nu", "inversion_sigma", "spherical_harmonics_l"] + OTHER_KEYS, np.asarray(nz_idx, dtype=np.int32)), nz_blk
    )
    return X

def cg_increment(
    x_nu, x_1, clebsch_gordan=None, lcut=None, filter_sigma=[-1,1],
    other_keys_match=None, mp=False
):
    """Specialized version of the CG product to perform iterations with nu=1 features"""

    nu = x_nu.keys["order_nu"][0]

    feature_roots = _remove_suffix(x_1.block(0).properties.names)

    if nu == 1:
        feature_names = tuple(root+"_1" for root in feature_roots) + ("l_1",
                        ) + tuple(root+"_2" for root in feature_roots) + ("l_2",)
    else:
        feature_names = tuple(x_nu.block(0).properties.names) + (
            "k_" + str(nu + 1),) + tuple(root+"_"+str(nu+1) for root in feature_roots) + (
            "l_" + str(nu + 1),)

    return cg_combine(
        x_nu,
        x_1,
        feature_names=feature_names,
        clebsch_gordan=clebsch_gordan,
        lcut=lcut,
        filter_sigma=filter_sigma,
        other_keys_match=other_keys_match, mp=mp
    )

