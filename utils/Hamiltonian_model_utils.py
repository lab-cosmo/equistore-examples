import numpy as np
import sklearn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.base import RegressorMixin, BaseEstimator
from utils.hamiltonians import _orbs_offsets, _atom_blocks_idx
from time import time
import scipy
from equistore import Labels, TensorBlock, TensorMap


def get_feat_keys(fock_keys, sigma=0, nu=1):
    """Map fock_block keys to corresponding feature key. take as extra input the sigma, nu value if required.
    sigma=0 returns all possible sigma values at given 'nu'"""
    blocktype, species1, n1, l1, species2, n2, l2, L = fock_keys 
    feat= (blocktype, L, nu,sigma,species1, species2)
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

        super(SARidge, self).fit(Xm_flat, Ym_flat)
    def predict(self, Xm, X0=None):

        Y = super(SARidge, self).predict(Xm.reshape((-1, Xm.shape[2])))
        return Y.reshape((-1, 2*self.L+1,1))
    
    
class Fock_regression():
    ## maybe we can make better use of TensorMaps in defining models
    def __init__(self, orbs, *args, **kwargs):
        self._orbs = orbs
        _, self._eldict = _orbs_offsets(orbs)
        self._args = args
        self._kwargs = kwargs
        self.model_template = SARidge
        self.block_keys =[]
        self.block_samples={}
        self.block_feats={}
        
    def fit(self, feats, fock_bc, slices=None, progress=None):
        self._models = {}
        self._cv_stats = {}
        blocks = []
        block_idx = []
        self.block_keys = fock_bc.keys
        
        for idx_fock, block_fock in fock_bc:
            block_type, ai, ni, li, aj, nj, lj, L = tuple(idx_fock)
            
            parity= (-1)**(li+lj+L)
            tgt = block_fock.values
            idx_feats = get_feat_keys(idx_fock, sigma=parity, nu=1)
            if idx_feats not in feats.keys:
                print(idx_feats)
                raise ValueError("feature key not found")
             
            self.block_feats[idx_fock]= feats.block(block_type=block_type, L=L, nu=1, sigma=parity, species_i=ai, species_j=aj) 
            
            self._models[idx_fock] = self.model_template(L, *self._args, **self._kwargs)
            self._models[idx_fock].fit(self.block_feats[idx_fock].values, tgt)
            self.block_samples[idx_fock]= block_fock.samples
            
            #print(self._models.keys
                    
                    
    def predict(self, feats, progress=None):
        pred_blocks = []
        for idx in self._models.keys():
            L = idx[-1]                    
            block_data = self._models[idx].predict(self.block_feats[idx].values)
            block_samples = self.block_samples[idx]
            print(idx, block_data.shape)        
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
                        
        return pred_fock                

                
                              