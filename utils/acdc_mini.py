import numpy as np
import re
from equistore import Labels, TensorBlock, TensorMap
from .clebsh_gordan import ClebschGordanReal


def _remove_suffix(names, new_suffix=""):
    suffix = re.compile("_[0-9]?$")
    rname = []
    for name in names:
        match = suffix.search(name)
        if match is None:
            rname.append(name+new_suffix)
        else:
            rname.append(name[:match.start()]+new_suffix)
    return rname


def acdc_standardize_keys(descriptor):
    """ Standardize the naming scheme of density expansion coefficient blocks (nu=1) """

    key_names = np.array(descriptor.keys.names)
    if not "spherical_harmonics_l" in key_names:
        raise ValueError("Descriptor missing spherical harmonics channel key `spherical_harmonics_l`")
    if "species_atom_1" in key_names:
        key_names[np.where(key_names=="species_atom_1")[0]] = "species_center"
    if "species_atom_2" in key_names:
        key_names[np.where(key_names=="species_atom_2")[0]] = "species_neighbor"
    key_names = tuple(key_names)
    blocks = []
    keys = []
    for key, block in descriptor:
        key = tuple(key)
        if not "inversion_sigma" in key_names:
            key = (1,) + key
        if not "order_nu" in key_names:
            key = (1,) + key
        keys.append(key)
        property_names = _remove_suffix(block.properties.names, "_1")
        sample_names = [ "center" if b == "first_atom" else ("neighbor" if b == "second_atom" else b) for b in block.samples.names ]
        # drops pair_id which we don't need in this application
        if "pair_id" in sample_names:
            new_samples = Labels( [n for n in sample_names if n!="pair_id"],
                                 np.asarray(block.samples.view(dtype = np.int32)).reshape(-1,len(sample_names))[:,[i for i in range(len(block.samples.names)) if block.samples.names[i] != "pair_id"]])
        else:
            new_samples = Labels(sample_names, np.asarray(block.samples.view(dtype = np.int32)).reshape(-1,len(sample_names)) )
        blocks.append(
            TensorBlock( values=block.values, samples=new_samples,
                        components=block.components, 
                         properties=Labels(property_names, np.asarray(block.properties.view(dtype = np.int32)).reshape(-1,len(property_names)) )
            ) )
    
    if not "inversion_sigma" in key_names:
        key_names = ("inversion_sigma",) + key_names
    if not "order_nu" in key_names:
        key_names = ( "order_nu",) + key_names
    
    return TensorMap(keys = Labels(names=key_names, values=np.asarray(keys, dtype=np.int32)),
                     blocks = blocks
                    )       

def cg_combine(
    x_a,
    x_b,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
    filter_sigma=[-1,1],
    other_keys_match=None,    
):
    """
    Performs a CG product of two sets of equivariants. Only requirement is that
    sparse indices are labeled as ("inversion_sigma", "spherical_harmonics_l", "order_nu"). The automatically-determined
    naming of output features can be overridden by giving a list of "feature_names".
    By defaults, all other key labels are combined in an "outer product" mode, i.e. if there is a key-side 
    neighbor_species in both x_a and x_b, the returned keys will have two neighbor_species labels, 
    corresponding to the parent features. By providing a list `other_keys_match` of keys that should match, these are 
    not outer-producted, but combined together. for instance, passing `["species center"]` means that the keys with the
    same species center will be combined together, but yield a single key with the same species_center in the results.
    """

    # determines the cutoff in the new features
    lmax_a = max(x_a.keys["spherical_harmonics_l"])
    lmax_b = max(x_b.keys["spherical_harmonics_l"])
    if lcut is None:
        lcut = lmax_a + lmax_b

    # creates a CG object, if needed
    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut)

    other_keys_a = tuple(name for name in x_a.keys.names if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"] )
    other_keys_b = tuple(name for name in x_b.keys.names if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"] )
    
    if other_keys_match is None:
        OTHER_KEYS = [ k+"_a" for k in other_keys_a ] + [ k+"_b" for k in other_keys_b ]
    else:     
        OTHER_KEYS = other_keys_match + [ 
            k+("_a" if k in other_keys_b else "") for k in other_keys_a if k not in other_keys_match ] + [
            k+("_b" if k in other_keys_a else "") for k in other_keys_b if k not in other_keys_match ]     
    
    # we assume grad components are all the same
    if x_a.block(0).has_gradient("positions"):
        grad_components = x_a.block(0).gradient("positions").components
    else:
        grad_components = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        NU = x_a.keys[0]["order_nu"] + x_b.keys[0]["order_nu"]
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).properties.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.block(0).properties.names)
            + ("l_" + str(NU),)
        )

    X_idx = {}
    X_blocks = {}
    X_samples = {}
    X_grad_samples = {}
    X_grads = {}

    # loops over sparse blocks of x_a
    for index_a, block_a in x_a:
        lam_a = index_a["spherical_harmonics_l"]
        sigma_a = index_a["inversion_sigma"]
        order_a = index_a["order_nu"]                
        properties_a = block_a.properties  # pre-extract this block as accessing a c property has a non-zero cost
        samples_a = block_a.samples
        
        # and x_b
        for index_b, block_b in x_b:
            lam_b = index_b["spherical_harmonics_l"]
            sigma_b = index_b["inversion_sigma"]
            order_b = index_b["order_nu"]       
            properties_b = block_b.properties
            samples_b = block_b.samples
            
            
            if other_keys_match is None:            
                OTHERS = tuple( index_a[name] for name in other_keys_a ) + tuple( index_b[name] for name in other_keys_b )
            else:
                OTHERS = tuple(index_a[k] for k in other_keys_match if index_a[k]==index_b[k])
                # skip combinations without matching key
                if len(OTHERS)<len(other_keys_match):
                    continue
                # adds non-matching keys to build outer product
                OTHERS = OTHERS + tuple(index_a[k] for k in other_keys_a if k not in other_keys_match)
                OTHERS = OTHERS + tuple(index_b[k] for k in other_keys_b if k not in other_keys_match)
                                
            if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                # we hard-code a combination method where b can be a pair descriptor. this needs some work to be general and robust
                # note also that this assumes that structure, center are ordered in the same way in the centred and neighbor descriptors
                neighbor_slice = []
                smp_a, smp_b = 0, 0
                while smp_b < samples_b.shape[0]:                    
                    if samples_b[smp_b][["structure","center"]] != samples_a[smp_a]:
                        smp_a+=1
                    neighbor_slice.append(smp_a)
                    smp_b+=1
                neighbor_slice = np.asarray(neighbor_slice)
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
            sel_idx = np.hstack([prop_ids_a[sel_feats[:,0]],prop_ids_b[sel_feats[:,1]] ])            
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
                    X_samples[KEY] = block_b.samples
                    if grad_components is not None:
                        X_grads[KEY] = []  
                        X_grad_samples[KEY] = block_b.gradient("positions").samples
                                
                # builds all products in one go
                one_shot_blocks = clebsch_gordan.combine_einsum(
                    block_a.values[neighbor_slice][:, :, sel_feats[:, 0]],
                    block_b.values[:, :, sel_feats[:, 1]],
                    L,
                    combination_string="iq,iq->iq",
                )
                # do gradients, if they are present...
                if grad_components is not None:
                    grad_a = block_a.gradient("positions")
                    grad_b = block_b.gradient("positions")
                    grad_a_data = np.swapaxes(grad_a.data, 1,2)
                    grad_b_data = np.swapaxes(grad_b.data, 1,2)
                    one_shot_grads = clebsch_gordan.combine_einsum(
                        block_a.values[grad_a.samples["sample"]][neighbor_slice, :, sel_feats[:, 0]],
                        grad_b_data[..., sel_feats[:, 1]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    ) + clebsch_gordan.combine_einsum(
                        block_b.values[grad_b.samples["sample"]][:, :, sel_feats[:, 1]],
                        grad_a_data[neighbor_slice, ..., sel_feats[:, 0]],
                        L=L,
                        combination_string="iq,iaq->iaq",
                    )

                # now loop over the selected features to build the blocks
                
                X_idx[KEY].append(sel_idx)
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
        block_data = np.concatenate(X_blocks[KEY], axis=-1)
        sph_components = Labels(
                ["spherical_harmonics_m"], np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1)
            )
        newblock = TensorBlock(
            # feature index must be last
            values=block_data,
            samples=X_samples[KEY],
            components=[sph_components],
            properties=Labels(feature_names, np.asarray(np.vstack(X_idx[KEY]), dtype=np.int32)),
        )
        if grad_components is not None:
            grad_data = np.swapaxes(np.concatenate(X_grads[KEY], axis=-1), 2, 1)
            newblock.add_gradient("positions", data=grad_data, samples=X_grad_samples[KEY], components=[grad_components[0], sph_components] )
        nz_blk.append(newblock)
    X = TensorMap(
        Labels(["order_nu", "inversion_sigma", "spherical_harmonics_l"] + OTHER_KEYS, np.asarray(nz_idx, dtype=np.int32)), nz_blk
    )
    return X

def cg_increment(
    x_nu, x_1, clebsch_gordan=None, lcut=None, filter_sigma=[-1,1],
    other_keys_match=None,
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
        lcut=lcut, filter_sigma=filter_sigma,
        other_keys_match=other_keys_match
    )    

def _matrix_sqrt(MMT):
    eva, eve = np.linalg.eigh(MMT)
    return (eve * np.sqrt(eva)) @ eve.T

def compress_features(x, w=None, threshold=None):
    new_blocks = []
    new_idxs = []
    new_A = []
    for index, block in x:
        nfeats = block.values.shape[-1]
        L = index['spherical_harmonics_l']

        # makes a copy of the features
        X = block.values.reshape(-1, nfeats).copy()
        selection = []
        while len(selection) < nfeats:
            norm = (X**2).sum(axis=0)
            sel_idx = norm.argmax()
            
            if norm[sel_idx] / (2 * L + 1) / X.shape[0] < threshold:
                break
            sel_x = X[:, sel_idx] / np.sqrt(norm[sel_idx])
            selection.append(sel_idx)
            
            # orthogonalize
            X -= sel_x.reshape(-1, 1) @ (sel_x @ X).reshape(1, -1)
        selection.sort()
        nsel = len(selection)
        if nsel == 0:
            continue
        new_idxs.append(tuple(index))
        
        Xt = block.values.reshape(-1, nfeats)[:, selection].copy()
        if w is not None:
            for i, s in enumerate(selection):
                Xt[:, i] /= w.block(index).values[0, 0, s]
        
        W = np.linalg.pinv(Xt) @ block.values.reshape(-1, nfeats)
        WW = W @ W.T
        A = _matrix_sqrt(WW)
        Xt = Xt @ A
        new_blocks.append(
            TensorBlock(
                values=Xt.reshape(block.values.shape[:2] + (-1,)),
                samples=block.samples,
                components=block.components,
                properties=block.properties[selection],
            )
        )
        new_A.append(
            TensorBlock(
                values=A.T.reshape(1, nsel, nsel),
                components=[Labels(
                    ["q_comp"], np.arange(nsel, dtype=np.int32).reshape(-1, 1)
                )],
                samples=Labels(["dummy"], np.zeros(shape=(1, 1), dtype=np.int32)),
                properties=block.properties[selection],
            )
        )
        new_sparse = Labels(x.keys.names, np.asarray(new_idxs, dtype=np.int32))
        
    return TensorMap(new_sparse, new_blocks), TensorMap(new_sparse, new_A)
