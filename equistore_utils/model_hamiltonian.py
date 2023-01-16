import numpy as np
import sklearn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.base import RegressorMixin, BaseEstimator
from .hamiltonians import *
from time import time
import scipy
from equistore import Labels, TensorBlock, TensorMap


def get_feat_keys(fock_keys, sigma=None, order_nu=None):
    """Map COUPLED fock_block keys to corresponding feature key. take as extra input the sigma, nu value if required.
    sigma=0 returns all possible sigma values at given 'nu'"""
    blocktype, species1, n1, l1, species2, n2, l2, L = fock_keys
    if sigma is None:
        z = (l1+l2+L)%2
        sigma = 1 - 2*z
    feat = [(order_nu, sigma, L, species1, species2, blocktype)]
#     feat= (blocktype, L,sigma,species1, species2)
    feat = Labels(["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center", "species_neighbor", "block_type"], np.asarray(feat, dtype=np.int32))
    return feat

def get_feat_keys_from_uncoupled(block_keys, sigma=None, order_nu=None):
    """Map UNCOUPLED block keys to corresponding feature key. take as extra input the sigma, nu value if required.
    sigma=0 returns all possible sigma values at given 'nu'"""
    blocktype, species1, n1, l1, species2, n2, l2 = block_keys
    feat_blocktype = blocktype
    keys_L=[]
    for L in range(abs(l1-l2), l1+l2+1):
        if sigma is None:
            z = (l1+l2+L)%2
            inv_sigma = 1 - 2*z
        elif abs(sigma)==1:
            inv_sigma = sigma
        else: 
            raise("Please check sigma value, it should be +1 or -1")
        
        if blocktype == 1 and n1 == n2 and l1 == l2:
            feat_blocktype = inv_sigma
                 
        if inv_sigma == -1 and blocktype == 0 and n1 == n2 and l1 == l2:
            continue     
        
        keys_L.append([(order_nu, inv_sigma, L, species1, species2, feat_blocktype)])
    #     feat= (blocktype, L,sigma,species1, species2)
    feat = Labels(["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center", "species_neighbor", "block_type"], np.asarray(keys_L, dtype=np.int32).reshape(-1,6))
    return feat

class SASplitter:
    """ CV splitter that takes into account the presence of "L blocks"
    associated with symmetry-adapted regression. Basically, you can trick conventional
    regression schemes to work on symmetry-adapted data y^M_L(A_i) by having the (2L+1)
    angular channels "unrolled" into a flat array. Then however splitting of train/test
    or cross validation must not "cut" across the M block. This takes care of that.
    """
    def __init__(self, L, cv=2):
        self.L = L
        self.cv = cv
        self.n_splits = cv

    def split(self, X, y, groups=None):

        ntrain = X.shape[0]
        if ntrain % (2*self.L+1) != 0:
            raise ValueError("Size of training data is inconsistent with the L value")
        ntrain = ntrain // (2*self.L+1)
        nbatch = (2*self.L+1)*(ntrain//self.n_splits)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for n in range(self.n_splits):
            itest = idx[n*nbatch:(n+1)*nbatch]
            itrain = np.concatenate([idx[:n*nbatch], idx[(n+1)*nbatch:]])
            yield itrain, itest

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
    

class SARidge(Ridge):
    """ Symmetry-adapted ridge regression class """

    def __init__(self, L, alpha=1, alphas=None, cv=2, solver='auto',
                 fit_intercept=False, scoring='neg_root_mean_squared_error'):
        self.L = L
        # L>0 components have zero mean by symmetry
        if L>0:
            fit_intercept = False
        self.cv = SASplitter(L, cv)
        self.alphas = alphas
        self.cv_stats = None
        self.scoring = scoring
        self.solver = solver
        super(SARidge, self).__init__(alpha=alpha, fit_intercept=fit_intercept, solver=solver)

    def fit(self, Xm, Ym, X0=None):
        # this expects properties in the form [i, m] and features in the form [i, q, m]
        # in order to train a SA-GPR model the m indices have to be moved and merged with the i

        #Xm_flat = np.moveaxis(Xm, 2, 1).reshape((-1, Xm.shape[1]))
        Xm_flat = Xm.reshape((-1, Xm.shape[2]))
        Ym_flat = Ym.flatten()
        if self.alphas is not None:
            # determines alpha by grid search
            rcv = Ridge(fit_intercept=self.fit_intercept)
            gscv = GridSearchCV(rcv, dict(alpha=self.alphas), cv=self.cv, scoring=self.scoring)
            gscv.fit(Xm_flat, Ym_flat)
            self.cv_stats = gscv.cv_results_
            self.alpha = gscv.best_params_["alpha"]
            
        b=super(SARidge, self).fit(Xm_flat, Ym_flat)    
        return b.coef_, b.intercept_

    def predict(self, Xm, X0=None):

        Y = super(SARidge, self).predict(Xm.reshape((-1, Xm.shape[2])))
        return Y.reshape((-1, 2*self.L+1,1))
    
    
class Fock_regression():
    def __init__(self, *args, **kwargs):
        #self._orbs = orbs
        #self._frame = frame
        self._args = args
        self._kwargs = kwargs
        self.model_template = SARidge
        
    def fit(self, feats, fock_bc, slices=None, progress=None):
        self._models = {}
        self._cv_stats = {}
        self._model_weights = {}
        self._model_intercepts = {}
        blocks = []
        block_idx = []
        self.block_keys = fock_bc.keys
        
        for idx_fock, block_fock in fock_bc:
            block_type, ai, ni, li, aj, nj, lj, L = tuple(idx_fock)
            
            parity= (-1)**(li+lj+L)
            tgt = block_fock.values
             
            block_feats = feats.block(block_type=block_type, spherical_harmonics_l=L,inversion_sigma=parity, species_center=ai, species_neighbor=aj) 
            
            self._models[idx_fock] = self.model_template(L, *self._args, **self._kwargs)
            self._model_weights[idx_fock], self._model_intercepts[idx_fock] = self._models[idx_fock].fit(block_feats.values, tgt)
            self._cv_stats[idx_fock] = self._models[idx_fock].cv_stats
            
        return self._model_weights, self._model_intercepts


    def predict(self, feats, frame, orbs, progress=None):
        pred_blocks = []
        for idx in self._models.keys():
            block_type, ai, ni, li, aj, nj, lj, L = idx
            parity= (-1)**(li+lj+L)            
            block_feats = feats.block(block_type=block_type, spherical_harmonics_l=L,inversion_sigma=parity, species_center=ai, species_neighbor=aj) 
            block_data = self._models[idx].predict(block_feats.values)
            block_samples = block_feats.samples
                   
            newblock = TensorBlock(
            values=block_data,
                samples=block_samples,
                components=[Labels(
                    ["mu"], np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1)
                )],
                properties= Labels(["values"], np.asarray([[0]], dtype=np.int32))
            )

            pred_blocks.append(newblock) 
                
        pred_fock = TensorMap(self.block_keys, pred_blocks)
        pred_dense = blocks_to_dense(decouple_blocks(pred_fock), frame, orbs)
        pred_eigvals = np.linalg.eigvalsh(pred_dense) #These are in general not the eigenvalues we are targetting, unless the fock matrix is orthogonal we need to solve a generalized eigenvalue problem. (might need to add a paramter 'overlap' in the function which is 'None' when target is orthogonal, otheriwse takes corresponding overlap matrix.                         
        return pred_dense, pred_eigvals
