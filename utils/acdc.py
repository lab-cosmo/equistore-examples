import numpy as np
from aml_storage import Labels, Block, Descriptor

from utils.builder import DescriptorBuilder
from utils.clebsh_gordan import ClebschGordanReal

###############################################################
# General utils
###############################################################


# TODO: these will be deprecated when a dict interface for Descriptors will be ready
def _mdesc_2_mdict(mdesc):
    """Converts a descriptors of weights to a dictionary"""
    mdict = {}
    for index, block in mdesc:
        mdict[tuple(index)] = {
            tuple(f): w for f, w in zip(block.features, block.values[0, 0])
        }
    return mdict


def _mdict_2_mdesc(mdict, sparse_names, feature_names):
    """Converts a dictionary of weights to a descriptor"""
    return Descriptor(
        sparse=Labels(
            names=sparse_names, values=np.asarray(list(mdict.keys()), dtype=np.int32)
        ),
        blocks=[
            Block(
                values=np.asarray(list(b.values())).reshape(1, 1, -1),
                components=Labels(["dummy"], np.zeros(shape=(1, 1), dtype=np.int32)),
                samples=Labels(["dummy"], np.zeros(shape=(1, 1), dtype=np.int32)),
                features=Labels(
                    feature_names, np.asarray(list(b.keys()), dtype=np.int32)
                ),
            )
            for b in mdict.values()
        ],
    )


def features_norm(x):
    """Computes the sample-wise norm of a set of features."""

    norm = np.zeros(len(x.block(0).samples))
    for index, block in x:
        norm += (block.values**2).sum(axis=1).sum(axis=1)
    return norm


def features_count(x):
    return np.sum([len(b.features) * (2 * i["lam"] + 1) for i, b in x])


#################################################################
# Basic CG iteration routines
#################################################################
def full_product_indices(m_a, m_b, feature_names=None):
    """
    Enumerates the indices of features that result from a product of ACDC equivariants.
    Sparse indices should be labeled as ("sigma", "lam", "nu").
    The arguments should correspond to the list of feature weights (multiplicities), to
    account for selection and duplicate removal.
    """

    # determines the cutoff in the new features
    lmaw_a = max(m_a.sparse["lam"])
    lmaw_b = max(m_b.sparse["lam"])
    lcut = lmaw_a + lmaw_b + 1

    # assumes uniform nu
    nu_a = m_a.sparse["nu"][0]
    nu_b = m_b.sparse["nu"][0]

    # block indexes for the incremented features
    NU = nu_a + nu_b
    M_dict = {(S, L, NU): {} for L in range(lcut + 1) for S in [-1, 1]}

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        feature_names = (
            tuple(n + "_a" for n in m_a.block(0).features.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in m_b.block(0).features.names)
            + ("l_" + str(NU),)
        )

    # computes the indices that must be present, as well as their weights
    for index_a, block_a in m_a:
        lam_a = index_a["lam"]
        sigma_a = index_a["sigma"]
        for index_b, block_b in m_b:
            lam_b = index_b["lam"]
            sigma_b = index_b["sigma"]
            for f_a, w_a in zip(block_a.features, block_a.values[0, 0]):
                f_a = tuple(f_a)
                for f_b, w_b in zip(block_b.features, block_b.values[0, 0]):
                    f_b = tuple(f_b)
                    W = w_a * w_b
                    IDX = f_a + (lam_a,) + f_b + (lam_b,)
                    for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                        S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                        M_dict[(S, L, NU)][IDX] = W

    # removes empty blocks
    for k in list(M_dict):
        if len(M_dict[k]) == 0:
            M_dict.pop(k)

    # casts the weights into a Descriptor format
    return _mdict_2_mdesc(M_dict, m_a.sparse.names, feature_names)


def cg_combine(
    x_a,
    x_b,
    m_a=None,
    m_b=None,
    M=None,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
):
    """
    Performs a CG product of two sets of equivariants. Only requirement is that
    sparse indices are labeled as ("sigma", "lam", "nu"). The automatically-determined
    naming of output features can be overridden by giving a list of "feature_names".
    A dictionary of select_features (organized in the same blocks as the sparse indices,
    each containing a dictionary of the feature indices and an associated multiplicity)
    can also be specified to filter the features that should be selected.
    """

    # determines the cutoff in the new features
    lmax_a = max(x_a.sparse["lam"])
    lmax_b = max(x_b.sparse["lam"])
    if lcut is None:
        lcut = lmax_a + lmax_b

    # creates a CG object, if needed
    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut)

    # assumes uniform nu in the descriptors
    nu_a = x_a.sparse["nu"][0]
    nu_b = x_b.sparse["nu"][0]

    # block indexes for the incremented features
    NU = nu_a + nu_b
    X_blocks = {(S, L, NU): [] for L in range(lcut + 1) for S in [-1, 1]}
    X_idx = {(S, L, NU): [] for L in range(lcut + 1) for S in [-1, 1]}

    # NB : assumes the samples are matching between different blocks. we could add some kind of
    # validation, at least on size if not on content
    samples_a = x_a.block(0).samples
    samples_b = x_b.block(0).samples
    if samples_a.shape != samples_b.shape:
        raise Exception("Mixed-samples combination not implemented")
    else:
        samples = samples_a        
    if x_a.block(0).has_gradient("positions"):
        grad_samples, _ = x_a.block(0).gradient("positions")
        X_grads = {(S, L, NU): [] for L in range(lcut + 1) for S in [-1, 1]}
    else:
        grad_samples = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).features.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.block(0).features.names)
            + ("l_" + str(NU),)
        )

    # it's much easier (and faster) to manipulate these as dictionary of dictionaries
    if M is not None:
        M_dict = _mdesc_2_mdict(M)
        weights_are_matrix = M.block(0).values.shape[1] > 1
    else:
        weights_are_matrix = False
    if m_a is not None:
        m_a_dict = _mdesc_2_mdict(m_a)
    if m_b is not None:
        m_b_dict = _mdesc_2_mdict(m_b)

    # loops over sparse blocks of x_a
    for index_a, block_a in x_a:
        lam_a = index_a["lam"]
        sigma_a = index_a["sigma"]

        if m_a is not None:
            w_block_a = m_a_dict[tuple(index_a)]
        # and x_b
        for index_b, block_b in x_b:
            lam_b = index_b["lam"]
            sigma_b = index_b["sigma"]
            if m_b is not None:
                w_block_b = m_b_dict[tuple(index_b)]
            # loops over all permissible output blocks. note that blocks will
            # be filled from different la, lb
            for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                # determines parity of the block
                S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                if M is not None:
                    if (S, L, NU) not in M.sparse:
                        continue
                    W_oth_features = M.block(sigma=S, lam=L, nu=NU).features
                    W_block = M_dict[(S, L, NU)]
                sel_feats = []
                sel_weights = []
                # determines the features that are in the select list
                for n_a in range(len(block_a.features)):
                    f_a = tuple(block_a.features[n_a])
                    w_a = 1.0 if m_a is None else w_block_a[f_a]
                    for n_b in range(len(block_b.features)):
                        f_b = tuple(block_b.features[n_b])
                        w_b = 1.0 if m_b is None else w_block_b[f_b]

                        # the index is assembled consistently with the scheme above
                        IDX = f_a + (lam_a,) + f_b + (lam_b,)

                        if M is None:
                            w_X = 1.0
                        else:
                            if IDX in W_oth_features:
                                if weights_are_matrix:
                                    w_X = 1.0
                                else:
                                    w_X = W_block[IDX]
                            else:
                                continue
                        sel_feats.append([n_a, n_b])
                        sel_weights.append(w_X / (w_a * w_b))
                        X_idx[(S, L, NU)].append(IDX)

                sel_feats = np.asarray(sel_feats, dtype=int)

                if len(sel_feats) == 0:
                    continue

                # builds all products in one go
                one_shot_blocks = clebsch_gordan.combine_einsum(
                    block_a.values[:, :, sel_feats[:, 0]],
                    block_b.values[:, :, sel_feats[:, 1]],
                    L,
                    combination_string="iq,iq->iq",
                )
                # do gradients, if they are present...
                if grad_samples is not None:
                    smp_a, grad_a = block_a.gradient("positions")
                    smp_b, grad_b = block_b.gradient("positions")
                    one_shot_grads = clebsch_gordan.combine_einsum(
                        block_a.values[smp_a["sample"]][:, :, sel_feats[:, 0]],
                        grad_b[:, :, sel_feats[:, 1]],
                        L=L,
                        combination_string="iq,iq->iq",
                    ) + clebsch_gordan.combine_einsum(
                        block_b.values[smp_b["sample"]][:, :, sel_feats[:, 1]],
                        grad_a[:, :, sel_feats[:, 0]],
                        L=L,
                        combination_string="iq,iq->iq",
                    )

                # now loop over the selected features to build the blocks
                for Q in range(len(sel_feats)):
                    # (n_a, n_b) = sel_feats[Q]
                    newblock = one_shot_blocks[:, :, Q] * sel_weights[Q]
                    X_blocks[(S, L, NU)].append(newblock)
                    if grad_samples is not None:
                        newgrad = one_shot_grads[:, :, Q] * sel_weights[Q]
                        X_grads[(S, L, NU)].append(newgrad)

    # turns data into sparse storage format (and dumps any empty block in the process)
    nz_idx = []
    nz_blk = []
    for SLN in X_blocks:
        S, L, NU = SLN
        # create blocks
        if len(X_blocks[SLN]) == 0:
            continue  # skips empty blocks
        nz_idx.append(SLN)
        block_data = np.moveaxis(np.asarray(X_blocks[SLN]), 0, -1)
        if weights_are_matrix:
            A = M.block(sigma=S, lam=L, nu=NU).values[0].T
            block_data = block_data @ A
        newblock = Block(
            # feature index must be last
            values=block_data,
            samples=samples,
            components=Labels(
                ["mu"], np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1)
            ),
            features=Labels(feature_names, np.asarray(X_idx[SLN], dtype=np.int32)),
        )
        if grad_samples is not None:
            grad_data = np.moveaxis(np.asarray(X_grads[SLN]), 0, -1)
            if weights_are_matrix:
                grad_data = grad_data @ A
            newblock.add_gradient("positions", grad_samples, grad_data)
        nz_blk.append(newblock)
    X = Descriptor(
        Labels(["sigma", "lam", "nu"], np.asarray(nz_idx, dtype=np.int32)), nz_blk
    )
    return X


def cg_combine_builder(
    x_a,
    x_b,
    m_a=None,
    m_b=None,
    M=None,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
):
    """
    Performs a CG product of two sets of equivariants. Only requirement is that
    sparse indices are labeled as ("sigma", "lam", "nu"). The automatically-determined
    naming of output features can be overridden by giving a list of "feature_names".
    A dictionary of select_features (organized in the same blocks as the sparse indices,
    each containing a dictionary of the feature indices and an associated multiplicity)
    can also be specified to filter the features that should be selected.
    """

    # determines the cutoff in the new features
    lmax_a = max(x_a.sparse["lam"])
    lmax_b = max(x_b.sparse["lam"])
    if lcut is None:
        lcut = lmax_a + lmax_b

    # creates a CG object, if needed
    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut)

    # assumes uniform nu in the descriptors
    nu_a = x_a.sparse["nu"][0]
    nu_b = x_b.sparse["nu"][0]

    # block indexes for the incremented features
    NU = nu_a + nu_b

    # NB : assumes the samples are matching. we could add some kind of
    # validation, at least on size if not on content
    samples = x_a.block(0).samples
    if x_a.block(0).has_gradient("positions"):
        grad_samples, _ = x_a.block(0).gradient("positions")
    else:
        grad_samples = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).features.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.block(0).features.names)
            + ("l_" + str(NU),)
        )

    builder = DescriptorBuilder(
        ["sigma", "lam", "nu"],
        sample_names=samples.names,
        component_names=["nu"],
        feature_names=feature_names,
    )

    # it's much easier (and faster) to manipulate these as dictionary of dictionaries
    if M is not None:
        weights_are_matrix = M.block(0).values.shape[1] > 1
    else:
        weights_are_matrix = False

    # loops over sparse blocks of x_a
    for index_a, block_a in x_a:
        lam_a = index_a["lam"]
        sigma_a = index_a["sigma"]

        if m_a is not None:
            w_block_a = m_a.block(index_a)
            w_block_a_values = w_block_a.values
            w_block_a_features = w_block_a.features
        # and x_b
        for index_b, block_b in x_b:
            lam_b = index_b["lam"]
            sigma_b = index_b["sigma"]
            if m_b is not None:
                w_block_b = m_b.block(index_b)
                w_block_b_values = w_block_b.values
                w_block_b_features = w_block_b.features

            # loops over all permissible output blocks. note that blocks will
            # be filled from different la, lb
            for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                # determines parity of the block
                S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                if M is not None:
                    if (S, L, NU) not in M.sparse:
                        continue
                    W_block_features = M.block(sigma=S, lam=L, nu=NU).features
                    W_block_values = M.block(sigma=S, lam=L, nu=NU).values
                if (S, L, NU) not in builder.blocks:
                    builder.add_block(
                        (S, L, NU),
                        samples=samples,
                        components=np.arange(-L, L + 1, dtype=np.int32).reshape(-1, 1),
                    )
                block = builder.blocks[(S, L, NU)]

                sel_feats = []
                sel_weights = []
                sel_idx = []
                # determines the features that are in the select list
                for n_a in range(len(block_a.features)):
                    f_a = tuple(block_a.features[n_a])
                    w_a = (
                        1.0
                        if m_a is None
                        else w_block_a_values[0, 0, w_block_a_features.position(f_a)]
                    )
                    for n_b in range(len(block_b.features)):
                        f_b = tuple(block_b.features[n_b])
                        w_b = (
                            1.0
                            if m_b is None
                            else w_block_b_values[
                                0, 0, w_block_b_features.position(f_b)
                            ]
                        )

                        # the index is assembled consistently with the scheme above
                        IDX = f_a + (lam_a,) + f_b + (lam_b,)

                        if M is None:
                            w_X = 1.0
                        else:
                            IDX_pos = W_block_features.position(IDX)
                            if IDX_pos is None:
                                continue
                            else:
                                if weights_are_matrix:
                                    w_X = 1.0
                                else:
                                    w_X = W_block_values[0, 0, IDX_pos]
                        sel_feats.append([n_a, n_b])
                        sel_weights.append(w_X / (w_a * w_b))
                        sel_idx.append(IDX)

                if len(sel_feats) == 0:
                    continue

                sel_feats = np.asarray(sel_feats, dtype=int)
                sel_weights = np.asarray(sel_weights)
                sel_idx = np.asarray(sel_idx)

                # builds all products in one go
                one_shot_blocks = clebsch_gordan.combine_einsum(
                    block_a.values[:, :, sel_feats[:, 0]],
                    block_b.values[:, :, sel_feats[:, 1]],
                    L,
                    combination_string="iq,iq->iq",
                )

                # do gradients, if they are present...
                if grad_samples is not None:
                    smp_a, grad_a = block_a.gradient("positions")
                    smp_b, grad_b = block_b.gradient("positions")
                    one_shot_grads = clebsch_gordan.combine_einsum(
                        block_a.values[smp_a["sample"]][:, :, sel_feats[:, 0]],
                        grad_b[:, :, sel_feats[:, 1]],
                        L=L,
                        combination_string="iq,iq->iq",
                    ) + clebsch_gordan.combine_einsum(
                        block_b.values[smp_b["sample"]][:, :, sel_feats[:, 1]],
                        grad_a[:, :, sel_feats[:, 0]],
                        L=L,
                        combination_string="iq,iq->iq",
                    )

                    block.add_features(
                        sel_idx,
                        one_shot_blocks * sel_weights,
                        one_shot_grads * sel_weights,
                    )
                else:
                    block.add_features(sel_idx, one_shot_blocks * sel_weights)

    X = builder.build()
    return X


def cg_increment_builder(
    x_nu, x_1, m_nu=None, m_1=None, M=None, clebsch_gordan=None, lcut=None
):
    """Specialized version of the CG product to perform iterations with nu=1 features"""
    nu = x_nu.sparse["nu"][0]
    if nu == 1:
        feature_names = ("n_1", "l_1", "n_2", "l_2")
    else:
        feature_names = tuple(x_nu.block(0).features.names) + (
            "k_" + str(nu + 1),
            "n_" + str(nu + 1),
            "l_" + str(nu + 1),
        )
    return cg_combine_builder(
        x_nu,
        x_1,
        m_nu,
        m_1,
        feature_names=feature_names,
        M=M,
        clebsch_gordan=clebsch_gordan,
        lcut=lcut,
    )


def cg_increment(
    x_nu, x_1, m_nu=None, m_1=None, M=None, clebsch_gordan=None, lcut=None
):
    """Specialized version of the CG product to perform iterations with nu=1 features"""
    nu = x_nu.sparse["nu"][0]
    if nu == 1:
        feature_names = ("n_1", "l_1", "n_2", "l_2")
    else:
        feature_names = tuple(x_nu.block(0).features.names) + (
            "k_" + str(nu + 1),
            "n_" + str(nu + 1),
            "l_" + str(nu + 1),
        )
    return cg_combine(
        x_nu,
        x_1,
        m_nu,
        m_1,
        feature_names=feature_names,
        M=M,
        clebsch_gordan=clebsch_gordan,
        lcut=lcut,
    )


########################################################################
#  Canonical-ordering routines
########################################################################


def _get_idx_nl(nu, idx, names):
    # extracts the n,l feature indices. assumes indices are labelled n_nu, l_nu
    nl_values = np.zeros((nu, 2), dtype=np.int32)
    for i, n in enumerate(names):
        w, k = n.split("_")
        k = int(k)
        if w == "n":
            nl_values[k - 1][0] = idx[i]
        elif w == "l":
            nl_values[k - 1][1] = idx[i]
    return nl_values


def _sort_idx_nl(nl_values):
    # sorts the n,l indices lexicographically (l first, then n)
    isort = np.lexsort(nl_values.T)
    return nl_values[isort]


def canonical_indices(m_nu, m_1, M=None):
    """
    Determines canonical (n,l, lexicographically sorted) indices, to implement
    the selection rule with lam_1<l2<l3 ... lam_nu [cf. Nigam et al. JCP 2020]
    This eliminates a large fraction of the linearly dependent equivariants.

    NB: this only works if we keep stacking the same density coefficients, which
    we assume are passed as the second argument. Also tracks multiplicity so
    norm should be conserved.
    """

    nu = m_nu.sparse["nu"][0]
    if nu == 1:
        names = ("n_1", "l_1", "n_2", "l_2")
    else:
        names = tuple(m_nu.block(0).features.names) + (
            "k_" + str(nu + 1),
            "n_" + str(nu + 1),
            "l_" + str(nu + 1),
        )
    if M is None:
        M = full_product_indices(m_nu, m_1)

    M_dict = {}
    for m_index, m_block in M:
        canonical_idx = {}
        canonical_counts = {}
        if len(m_block.features) == 0:
            continue
        for idx, W in zip(m_block.features, m_block.values[0, 0]):
            idx = tuple(idx)
            # gets only the n,l part of the nu features
            cidx = _get_idx_nl(nu + 1, idx, names)
            # gets the sorted version
            sidx = _sort_idx_nl(cidx)
            # converts to tuple
            cidx = tuple(cidx.flatten())
            sidx = tuple(sidx.flatten())
            # gets multiplicity (weight)

            if sidx not in canonical_counts:
                canonical_idx[sidx] = []
                canonical_counts[sidx] = 0

            canonical_counts[sidx] += W**2
            if cidx == sidx:
                canonical_idx[sidx].append(idx)

        MD = {}
        for nl in canonical_counts:
            if np.round(canonical_counts[nl]) % len(canonical_idx[nl]) > 1e-10:
                print(
                    "non integer count!", canonical_counts[nl], len(canonical_idx[nl])
                )
            for idx in canonical_idx[nl]:
                MD[idx] = np.sqrt(canonical_counts[nl] / len(canonical_idx[nl]))
        M_dict[tuple(m_index)] = MD

    # casts the weights into a Descriptor format
    return _mdict_2_mdesc(M_dict, M.sparse.names, names)


##############################################################################
# Numerical feature compression
##############################################################################

def threshold_indices(
    x_a,
    x_b,
    m_a=None,
    m_b=None,
    sel_threshold=0.0,
    l_threshold=None,
    M=None,
    feature_names=None,
):
    """
    Determines ACDC indices of the next iteration using a
    """

    # determines the cutoff in the new features
    lmaw_a = max(x_a.sparse["lam"])
    lmaw_b = max(x_b.sparse["lam"])
    if l_threshold is None:
        l_threshold = lmaw_a + lmaw_b + 1

    # assumes uniform nu
    nu_a = x_a.sparse["nu"][0]
    nu_b = x_b.sparse["nu"][0]
    NU = nu_a + nu_b

    if feature_names is None:
        feature_names = (
            tuple(n + "_a" for n in x_a.block(0).features.names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.block(0).features.names)
            + ("l_" + str(NU),)
        )

    # block indexes for the incremented features
    M_dict = {(S, L, NU): {} for L in range(l_threshold + 1) for S in [-1, 1]}

    nsamples = len(x_a.block(0).samples)
    thresh_norm = 0.0
    # computes the indices that must be present, as well as their weights
    for index_a, block_a in x_a:
        lam_a = index_a["lam"]
        sigma_a = index_a["sigma"]
        block_a_sz = (block_a.values**2).sum(axis=1)
        for index_b, block_b in x_b:
            lam_b = index_b["lam"]
            sigma_b = index_b["sigma"]
            # recomputing is a bit wasteful, but whatever. optimize later
            block_b_sz = (block_b.values**2).sum(axis=1)
            for i_a, f_a, bsz_a in zip(
                range(len(block_a.features)), block_a.features, block_a_sz.T
            ):
                f_a = tuple(f_a)
                if m_a is None:
                    w_a = 1.0
                else:
                    w_a = m_a.block(index_a).values[0, 0, i_a]
                for i_b, f_b, bsz_b in zip(
                    range(len(block_b.features)), block_b.features, block_b_sz.T
                ):
                    f_b = tuple(f_b)
                    if m_b is None:
                        w_b = 1.0
                    else:
                        w_b = m_b.block(index_b).values[0, 0, i_b]
                    W = w_a * w_b
                    IDX = f_a + (lam_a,) + f_b + (lam_b,)

                    # we consider the threshold based on the mean value of the
                    # features that will be generated. this is consistent with
                    # the way we select when compressing after the fact.
                    ab_norm = (bsz_a * bsz_b).sum() / nsamples
                    if ab_norm / ((2 * lam_a + 1) * (2 * lam_b + 1)) > sel_threshold:
                        for L in range(
                            np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, l_threshold)
                        ):
                            S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                            M_dict[(S, L, NU)][IDX] = W
                    else:
                        thresh_norm += ab_norm

    # removes empty blocks
    for k in list(M_dict):
        if len(M_dict[k]) == 0:
            M_dict.pop(k)

    # casts the weights into a Descriptor format
    return _mdict_2_mdesc(M_dict, x_a.sparse.names, feature_names), thresh_norm


def _matrix_sqrt(MMT):
    eva, eve = np.linalg.eigh(MMT)
    return (eve * np.sqrt(eva)) @ eve.T


def compress_features(x, w=None, threshold=None):
    new_blocks = []
    new_idxs = []
    new_A = []
    for index, block in x:
        nfeats = block.values.shape[-1]
        S, L, NU = tuple(index)

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
            Block(
                values=Xt.reshape(block.values.shape[:2] + (-1,)),
                samples=block.samples,
                components=block.components,
                features=block.features[selection],
            )
        )
        new_A.append(
            Block(
                values=A.T.reshape(1, nsel, nsel),
                components=Labels(
                    ["q_comp"], np.arange(nsel, dtype=np.int32).reshape(-1, 1)
                ),
                samples=Labels(["dummy"], np.zeros(shape=(1, 1), dtype=np.int32)),
                features=block.features[selection],
            )
        )
        new_sparse = Labels(x.sparse.names, np.asarray(new_idxs, dtype=np.int32))
    return Descriptor(new_sparse, new_blocks), Descriptor(new_sparse, new_A)
