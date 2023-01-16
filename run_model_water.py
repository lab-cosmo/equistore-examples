import numpy as np
import torch
import json
import ase.io
from itertools import product
import matplotlib.pyplot as plt
# from rascal.representations import SphericalExpansion
import copy
from tqdm import tqdm
from ase.units import Hartree

from torch_hamiltonian_utils.torch_cg import ClebschGordanReal
from torch_hamiltonian_utils.torch_hamiltonians import fix_pyscf_l1, lowdin_orthogonalize, dense_to_blocks, blocks_to_dense, couple_blocks, decouple_blocks, hamiltonian_features
from torch_hamiltonian_utils.torch_builder import TensorBuilder

import equistore
from equistore import Labels, TensorBlock, TensorMap
from equistore_utils.librascal import  RascalSphericalExpansion, RascalPairExpansion
from equistore_utils.acdc_mini import acdc_standardize_keys, cg_increment, cg_combine
from equistore_utils.model_hamiltonian import get_feat_keys, get_feat_keys_from_uncoupled 

import importlib
torch.set_default_dtype(torch.float64)



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


def lowdin_orthogonalize(fock, s):
    """
    lowdin orthogonalization of a fock matrix computing the square root of the overlap matrix
    """
    eva, eve = np.linalg.eigh(s)
    sm12 = eve @ np.diag(1.0/np.sqrt(eva)) @ eve.T
    return sm12 @ fock @ sm12


frames1 = ase.io.read("data/water-hamiltonian/water_coords_1000.xyz", ":800")
#frames1 = ase.io.read("data/ethanol-hamiltonian/ethanol_4500.xyz",":500")
frames = frames1 #+ frames2
for f in frames:
    f.cell = [100,100,100]
    f.positions += 50
jorbs = json.loads(json.load(open('data/water-hamiltonian/water_orbs.json', "r")))
#jorbs = json.loads(json.load(open('data/water-hamiltonian/orbs_def2_water.json', "r")))
#jorbs = json.load(open('data/ethanol-hamiltonian/orbs_saph_ethanol.json','r'))
orbs = {}
zdic = {"O" : 8, "H":1, "C":6}
for k in jorbs:
    orbs[zdic[k]] = jorbs[k]

#focks = np.load("data/water-hamiltonian/water_fock.npy", allow_pickle=True)[:len(frames1)]
#overlap = np.load("data/water-hamiltonian/water_overlap.npy", allow_pickle=True)[:len(frames1)]

#orthogonal = []
#for i in range(len(focks)): 
#    focks[i] = fix_pyscf_l1(focks[i],frames[i], orbs)
#    overlap[i] = fix_pyscf_l1(overlap[i],frames[i], orbs)
#    orthogonal.append(lowdin_orthogonalize(focks[i], overlap[i]))

focks1 = np.load("data/water-hamiltonian/water_saph_orthogonal.npy", allow_pickle=True)[:len(frames1)]
#focks1 = np.load("data/ethanol-hamiltonian/ethanol_saph_orthogonal.npy", allow_pickle=True)[:len(frames1)]

focks = focks1


cg = ClebschGordanReal(7)

blocks = dense_to_blocks(focks, frames, orbs)
fock_bc = couple_blocks(blocks, cg)
# ## Feature computation

rascal_hypers = {
    "interaction_cutoff": 4.0,
    "cutoff_smooth_width": 0.5,
    "max_radial": 8,
    "max_angular": 6,
    "gaussian_sigma_constant" : 0.2,
    "gaussian_sigma_type": "Constant",
    "compute_gradients":  False,
}

spex = RascalSphericalExpansion(rascal_hypers)
rhoi = spex.compute(frames)
#
lmax = rascal_hypers["max_angular"]
pairs = RascalPairExpansion(rascal_hypers)
gij = pairs.compute(frames)
rho1i = acdc_standardize_keys(rhoi)
rho1i.keys_to_properties(['species_neighbor'])
gij =  acdc_standardize_keys(gij)


rho2i = cg_increment(rho1i, rho1i, lcut=2, other_keys_match=["species_center"], clebsch_gordan=cg)

#rho3i = cg_increment(rho2i, rho1i, lcut=2, other_keys_match=["species_center"], clebsch_gordan=cg)

rho1ij = cg_increment(rho1i, gij, lcut=2, other_keys_match=["species_center"], clebsch_gordan=cg)

#rho2ij = cg_increment(rho2i, gij, lcut=2, other_keys_match=["species_center"], clebsch_gordan=cg)

features = hamiltonian_features(rho2i, rho1ij)

from equistore.io import save
save("models_water/feature.npz", features)
#

def normalize_feats(feat, all_blocks=True): 
    all_norm = 0
    for block_idx, block in feat: 
        block_norm = np.linalg.norm(block.values)
#         print(block_idx, block_norm)
        all_norm += block_norm**2
    normalized_blocks=[]
    for block_idx, block in feat: 
        newblock = TensorBlock(
                        values=block.values/np.sqrt(all_norm ),
                        samples=block.samples,
                        components=block.components,
                        properties= block.properties)
                    
        normalized_blocks.append(newblock) 
        
    norm_feat = TensorMap(feat.keys, normalized_blocks)
    return norm_feat

#norm_feat = normalize_feats(features)

#save("./norm_feat.npz", norm_feat)

from equistore.io import _labels_from_npz
import equistore.operations as operations

class HamiltonianDataset(torch.utils.data.Dataset):
    #Dataset class
    def __init__(self, feature_path, target, frames, feature_nu = 2):
        #
        self.features = np.load(feature_path, mmap_mode = 'r')
        #self.target = np.load(target_path, mmap_mode = 'r') 
        self.target = target #Uncoupled hamiltonian 
        self.keys_features = equistore.io._labels_from_npz(self.features["keys"])
        self.currentkey = self.target.keys[0]
        self.feature_nu = feature_nu
        self.frames = frames
        
        self.allfeatkey = []
        for t_key in self.target.keys:
            feature_key = self.get_feature_keys(t_key)
            self.allfeatkey.append(feature_key)
        #Remove Duplicates
        nodupes = set()
        for x in self.allfeatkey:
            if len(x) > 1:
                for z in x:
                    nodupes.add(tuple(z))
            else:
                nodupes.add(tuple(x[0]))
        
        nodupes = np.array(list(nodupes), np.int32)
        
        self.allfeatkey = Labels(names = self.allfeatkey[0].dtype.names, values = nodupes)
        

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, structure_idx):
        feature_block, feature_key = self.generate_feature_block(self.features, structure_idx)        
        #samples_filter, target_block_samples = self.get_index_from_idx(self.target.block(self.currentkey).samples, structure_idx)

        if self.currentkey is None:
            t_blocks = []
            for _, block in self.target:            
                t_block = operations.slice_block(block, samples = Labels(names = ['structure'], values = (np.array(structure_idx)+1).reshape(-1,1)) )
                t_blocks.append(t_block)
            target_block = TensorMap(self.target.keys, t_blocks)
        else:
            target_block = operations.slice_block(self.target.block(self.currentkey), samples = Labels(names = ['structure'], values = (np.array(structure_idx)+1).reshape(-1,1)) )
        structure = [self.frames[i] for i in structure_idx]
        #Modify feature_block to tensormap
        feature_map = TensorMap(feature_key, feature_block)
        return feature_map, target_block, structure


    def get_feature_keys(self,uncoupled_key):
        return get_feat_keys_from_uncoupled(uncoupled_key, order_nu = self.feature_nu)
    
    def generate_feature_block(self, memmap, structure_idx):
        #Generate the block from npz file
        output = []
        if self.currentkey is None:
            feature_key = self.allfeatkey
                
        else:
            feature_key = self.get_feature_keys(self.currentkey)
            
        for key in feature_key:
            block_index = list(self.keys_features).index(key)
            prefix = f"blocks/{block_index}/values"        
            block_samples = equistore.io._labels_from_npz(memmap[f"{prefix}/samples"])
            block_components = []
            for i in range(1):
                block_components.append(equistore.io._labels_from_npz(memmap[f"{prefix}/components/{i}"]))
            block_properties = equistore.io._labels_from_npz(memmap[f"{prefix}/properties"])
             

            samples_filter, block_samples = self.get_index_from_idx(block_samples, structure_idx)

            block_data = memmap[f"{prefix}/data"][samples_filter]
            block = TensorBlock(block_data, block_samples, block_components, block_properties)
            output.append(block)
        return output, feature_key
    
    def get_n_properties(self, memmap, key):
        block_index = list(self.keys_features).index(key)
        prefix = f"blocks/{block_index}/values"  
        block_properties = equistore.io._labels_from_npz(memmap[f"{prefix}/properties"])
        
        return len(block_properties)
    
    def get_index_from_idx(self, block_samples, structure_idx):
        #Get samples label from IDX
        samples = Labels(names = ['structure'], values = np.array(structure_idx).reshape(-1,1))
        
        all_samples = block_samples[['structure']].tolist()
        set_samples_to_slice = set(samples.tolist())
        samples_filter = np.array(
            [sample in set_samples_to_slice for sample in all_samples]
        )
        new_samples = block_samples[samples_filter]
        
        return samples_filter, new_samples
    
    def collate_output_values(blocks):
        feature_out = []
        target_out = []
        for sample_output in blocks:
            feature_block, target_block, structure = sample_output
            for z in feature_block:
                feature_out.append(torch.tensor(z.values))
            target_out.append(torch.tensor(target_block.values))

        return feature_out, target_out


test_target_path = "models_water/test_fock.npz"
test_feature_path = "models_water/feature.npz"
test = HamiltonianDataset(test_feature_path, blocks, frames)

# ## DataLoader

def collate_blocks(block_tuple):
    feature_tensor_map, target_block, structure_array = block_tuple[0]
    
    return feature_tensor_map, target_block, structure_array
    

from torch.utils.data import DataLoader, BatchSampler, SubsetRandomSampler

#Sampler = torch.utils.data.SubsetRandomSampler(range(1,len(test)+1), generator=None)
Sampler = torch.utils.data.sampler.RandomSampler(test)
BSampler = torch.utils.data.sampler.BatchSampler(Sampler, batch_size = 800, drop_last = False)

dataloader = DataLoader(test, sampler = BSampler, collate_fn = collate_blocks)


# ## Model 

def get_block_samples(t_key, feature_map):
    f_key = get_feat_keys_from_uncoupled(t_key, None , 2)
    ss = feature_map.block(f_key[0]).samples.copy()
    ss["structure"] = ss["structure"]+1
    
    return ss


class HamModel(torch.nn.Module):
    #Handles prediction of entire hamiltonian and derived results
    def __init__(self, Hamiltonian_Dataset, device, regularization=None, seed=None, layer_size=None):
        super().__init__()
#         self.features = features 
#         self.target = target
        self.models = torch.nn.ModuleDict()
        self.loss_history={}
        self.device = device
        self.target_keys = Hamiltonian_Dataset.target.keys
        self.block_samples = {}
        self.block_components = {}
        for key in Hamiltonian_Dataset.target.keys:
#             _block_type, _a_i, _n_i, _l_i, _a_j, _n_j, _l_j = key
#             target_keys = Hamiltonian_Dataset.target.keys[Hamiltonian_Dataset.target.blocks_matching(
#                 block_type = _block_type, a_i = _a_i, n_i = _n_i, l_i = _l_i, a_j = _a_j,
#                 n_j = _n_j, l_j = _l_j)]
            
            #self.block_samples[str(key)] = Hamiltonian_Dataset.target.block(key).samples
            self.block_components[str(key)] = Hamiltonian_Dataset.target.block(key).components
        
    
            n_inputs = []
            model_keys = []

            feature_keys = Hamiltonian_Dataset.get_feature_keys(key)
            for f_key in feature_keys: 
                n_features = Hamiltonian_Dataset.get_n_properties(Hamiltonian_Dataset.features, f_key)
                n_inputs.append(n_features)
                model_keys.append(f_key)
                
                
            n_outputs = np.ones_like(n_inputs)
                
            self.models[str(key)] = BlockModel(cg.decouple,n_inputs, n_outputs, device, model_keys, key, seed = seed, hidden_layers = layer_size)
        self.to(device)
            
    def forward(self, x):
        #Ham model uses target keys
        pred_blocks = []
        for t_key in self.target_keys:
            
            pred = self.models[str(t_key)].forward(x) #feature_tensormap must correspond to the correct features, model returns block
            
            #try:
#             print (pred.shape)
#             print ((2 * t_key['l_i'])+1)
#             print ((2 * t_key['l_j']) + 1)
            pred_block = TensorBlock(
                    values=pred.reshape((-1, (2 * t_key['l_i'])+1, (2 * t_key['l_j']) + 1, 1)), #?
                    samples = get_block_samples(t_key, x),
                    components = self.block_components[str(t_key)] ,
                    properties= Labels(["dummy"], np.asarray([[0]], dtype=np.int32))
                )
#             except:
#                 print (t_key)
#                 print (pred)
#                 print (self.block_samples[str(t_key)])
#                 print (self.block_components[str(t_key)])
                
            pred_blocks.append(pred_block)
        pred_hamiltonian = TensorMap(self.target_keys, pred_blocks)
        return(pred_hamiltonian)
    
    #write/fix forward function for train_indiv
    
    def train_individual(self, train_dataloader, regularization_dict, optimizer_type, n_epochs, loss_function, lr):
        #Iterates through the keys of self.model, then for each key we will fit self.model[key] with data[key]
        total = len(self.models)
        for index, t_key in enumerate(self.target_keys):
            print ("Now training on Block {} of {}".format(index, total))
            train_dataloader.dataset.currentkey = t_key
            
            loss_history_key = self.models[str(t_key)].fit(train_dataloader, loss_function, optimizer_type, lr, regularization_dict[str(t_key)], n_epochs)

            self.loss_history[str(t_key)] = loss_history_key
    
    def train_collective(self, train_dataloader, regularization_dict, optimizer_type, n_epochs, loss_function, lr):
        #for every loop through target keys, we predict the corresponding block and assemble the final hamiltonian
        optimizer_dict = {}
        if optimizer_type == "Adam":
            for key in train_dataloader.dataset.target.keys:
                optimizer_dict[str(key)] = torch.optim.Adam(self.models[str(key)].parameters(), lr = lr, weight_decay = regularization_dict[str(key)])
            threshold = 200
            scheduler_threshold = 200
            tol = 0
            history_step = 1000
        
        elif optimizer_type == "LBFGS":
#             for key in train_dataloader.dataset.target.keys:
#                 optimizer_dict[str(key)] = torch.optim.LBFGS(self.models[str(key)].parameters(), lr = lr)
            optimizer_dict[0] = torch.optim.LBFGS(self.models.parameters(), lr = lr)
            threshold = 30
            scheduler_threshold = 30
            tol = 0
            history_step = 10                
    
        scheduler_dict = {}
        scheduler_dict[0] = torch.optim.lr_scheduler.StepLR(optimizer_dict[0], scheduler_threshold, gamma = 0.5)
#         for key in train_dataloader.dataset.target.keys:
#             scheduler_dict[str(key)] = torch.optim.lr_scheduler.StepLR(optimizer_dict[str(key)], scheduler_threshold, gamma = 0.5)

        reg_weights = torch.tensor(list(regularization_dict.values()))
        best_state = copy.deepcopy(self.state_dict())
        lowest_loss = torch.tensor(9999)
        pred_loss = torch.tensor(0)
        trigger = 0
        loss_history = []
        pbar = tqdm(range(n_epochs))
        
        for epoch in pbar:
            pbar.set_description(f"Epoch: {epoch}")
            pbar.set_postfix(pred_loss = pred_loss.item(), lowest_loss = lowest_loss.item(), trigger = trigger)
            train_dataloader.dataset.currentkey = None
            
            for x_data, y_data, structure in train_dataloader: 
                self.collective_zg(optimizer_dict)
                #x_data, y_data = x_data.to(self.device), y_data.to(self.device)
                if optimizer_type == "LBFGS":
                    def closure():
                        self.collective_zg(optimizer_dict)
                        _pred = self.forward(x_data)
                        _pred_loss = loss_function(_pred, y_data, structure, orbs)       
                        _pred_loss = torch.nan_to_num(_pred_loss, nan=lowest_loss.item(), posinf = lowest_loss.item(), neginf = lowest_loss.item())                          
                        _reg_loss = self.get_regression_values(reg_weights) #Only works for 1 layer #Need to change!!
                        _new_loss = _pred_loss + _reg_loss
                        _new_loss.backward()
                        return _new_loss
                    for value in optimizer_dict.values():
                        value.step(closure)
#                     for param in self.parameters():
#                         print (param.grad)
                elif optimizer_type == "Adam":
                    pred = self.forward(x_data)
                    pred_loss = loss_function(pred, y_data, structure, orbs)  
#                     reg_loss = torch.sum(torch.pow(self.nn.weight,2))#Only works for 1 layer
                    new_loss = pred_loss 
                    new_loss.backward()
                    self.collective_step(optimizer_dict)
            with torch.no_grad():
                current_loss = 0 
                for x_data, y_data, structure in train_dataloader:
                    pred = self.forward(x_data)
                    current_loss  += loss_function(pred, y_data, structure, orbs)   #Loss should be normalized already
                pred_loss = current_loss
                reg_loss = self.get_regression_values(reg_weights)#Only works for 1 layer
                new_loss = pred_loss + reg_loss

                if pred_loss >100000 or (pred_loss.isnan().any()) :
                    print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                    self.load_state_dict(best_state)
                    if optimizer_type == "Adam":
                        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = reg.item())
                    elif optimizer_type == "LBFGS":
                        optimizer = torch.optim.LBFGS(self.parameters(), lr = lr)

                if epoch % history_step == 1:
                    loss_history.append(lowest_loss.item())
                
                if lowest_loss - new_loss > tol: #threshold to stop training             
                    best_state = copy.deepcopy(self.state_dict())
                    lowest_loss = new_loss 
                    trigger = 0 
                    
                    
                else:
                    trigger += 1
                    self.collective_step(scheduler_dict)
                    if trigger > threshold:
                        self.load_state_dict(best_state)
                        print ("Implemented early stopping with lowest_loss: {}".format(lowest_loss))
                        return loss_history
        return loss_history
        
    def collective_step(self, dictionary):
        for value in dictionary.values():
            value.step()
            
    def collective_zg(self, dictionary):
        for value in dictionary.values():
            value.zero_grad()
    
    def get_regression_values(self, reg_weights):
        output = []
        for param in self.parameters():
            output.append(torch.sum(torch.pow(param,2)))
        try:
            output = torch.sum(torch.tensor(output) * reg_weights)
        except:
            output = 0
        return output



class BlockModel(torch.nn.Module): #Currently only 1 model per block
    def __init__(self, reconstruction_function, inputSize, outputSize, device, keys, target_key, seed = None, hidden_layers = None):
        super().__init__()
        self.reconstruction_function = reconstruction_function
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.device = device
        self.keys = keys
        self.target_key = target_key
        self.hidden_layers = hidden_layers
        self.initialize_model(seed)
        
        self.to(device)
    
    def initialize_model(self, seed):
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.models = torch.nn.ModuleDict()
        for index, key in enumerate(self.keys):
            self.models[str(key)] = torch.nn.Linear(self.inputSize[index], self.outputSize[index], bias = False)
        
    def forward(self, feature_tensormap):
        #Block model uses feature keys
        pred_values = {}
        for key in self.keys:
            feature_values = feature_tensormap.block(key).values
            d1, d2, d3 = feature_values.shape
            L = int((d2 -1)/2)
            pred = self.models[str(key)](torch.tensor(feature_values.reshape(d1 * d2, d3)))
            pred = pred.reshape(d1,d2)
            pred_values[L] = pred
        
        pred_block_values = self.reconstruction_function({(self.target_key['l_i'],self.target_key['l_j']) : pred_values})
        

        #pred = torch.hstack(pred_values)
        #pred = self.reconstruction_function(pred_values)
        return pred_block_values 

    
    def fit(self,traindata_loader, loss_function, optimizer_type, lr, reg, n_epochs):
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = reg.item())
            threshold = 200
            scheduler_threshold = 50
            tol = 1e-2
            history_step = 1000
        
        elif optimizer_type == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr = lr)
            threshold = 30
            scheduler_threshold = 10
            tol = 1e-2
            history_step = 10
            
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_threshold, gamma = 0.5)
        best_state = copy.deepcopy(self.state_dict())
        lowest_loss = torch.tensor(9999)
        pred_loss = torch.tensor(0)
        trigger = 0
        loss_history = []
        pbar = tqdm(range(n_epochs))
        
        for epoch in pbar:
            pbar.set_description(f"Epoch: {epoch}")
            pbar.set_postfix(pred_loss = pred_loss.item(), lowest_loss = lowest_loss.item(), trigger = trigger)
            
            for x_data, y_data, structure in traindata_loader: 
                optimizer.zero_grad()
                #x_data, y_data = x_data.to(self.device), y_data.to(self.device)
                if optimizer_type == "LBFGS":
                    def closure():
                        optimizer.zero_grad()
                        _pred = self.forward(x_data)                                        
                        _pred_loss = loss_function(_pred, y_data.values)
                        _pred_loss = torch.nan_to_num(_pred_loss, nan=lowest_loss.item(), posinf = lowest_loss.item(), neginf = lowest_loss.item())                 
                        _reg_loss = self.get_regression_values(reg.item()) #Only works for 1 layer
                        _new_loss = _pred_loss + _reg_loss
                        _new_loss.backward()
                        return _new_loss
                    optimizer.step(closure)

                elif optimizer_type == "Adam":
                    pred = self.forward(x_data)
                    pred_loss = loss_function(pred, y_data.values)
                    #reg_loss = self.get_regression_values(reg.item())#Only works for 1 layer
                    new_loss = pred_loss #+ reg_loss
                    new_loss.backward()

                    optimizer.step()
                
            with torch.no_grad():
                current_loss = 0 
                for x_data, y_data, structure in traindata_loader:
                    pred = self.forward(x_data)
                    current_loss  += loss_function(pred, y_data.values) #Loss should be normalized already
                pred_loss = current_loss
                reg_loss = self.get_regression_values(reg.item()) 
                new_loss = pred_loss + reg_loss
                if pred_loss >100000 or (pred_loss.isnan().any()) :
                    print ("Optimizer shows weird behaviour, reinitializing at previous best_State")
                    self.load_state_dict(best_state)
                    if optimizer_type == "Adam":
                        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = reg.item())
                    elif optimizer_type == "LBFGS":
                        optimizer = torch.optim.LBFGS(self.parameters(), lr = lr)

                if epoch % history_step == 1:
                    loss_history.append(lowest_loss.item())
                
                if lowest_loss - new_loss > tol: #threshold to stop training
                    best_state = copy.deepcopy(self.state_dict())
                    lowest_loss = new_loss 
                    trigger = 0 
                    
                    
                else:
                    trigger += 1
                    scheduler.step()
                    if trigger > threshold:
                        self.load_state_dict(best_state)
                        print ("Implemented early stopping with lowest_loss: {}".format(lowest_loss))
                        return loss_history
        return loss_history
    
    def get_regression_values(self, reg_weights):
        output = []
        for param in self.parameters():
            output.append(torch.sum(torch.pow(param,2)))
        try:
            output = torch.sum(torch.tensor(output) * reg_weights)
        except:
            output = 0
        return output


# ## Training


def mse_block_values(pred, true):
    true = true.reshape(true.shape[:-1])
    MSE = torch.sum(torch.pow(true - pred,2)) / torch.numel(true)
    return MSE*Hartree**2#*1e5
    #return torch.sqrt(MSE)*Hartree#*1e5

def mse_full(pred_blocks, fock,frame, orbs):
    predicted = blocks_to_dense(pred_blocks, frame, orbs)
    fock = torch.tensor(focks)
    mse_loss = torch.empty(len(frame))
    for i in range(len(frame)):
        mse_loss[i] = ((torch.linalg.norm(fock[i]-predicted[i]))**2)/len(fock[i])
        #print("from mse", i, fock[i], mse_loss[i])
    return torch.mean(mse_loss)*(Hartree)**2
    #return torch.sqrt(torch.mean(mse_loss))*Hartree
def mse_eigvals(pred_blocks, fock, frame, orbs):
    fock = torch.tensor(focks)
    predicted = blocks_to_dense(pred_blocks, frame, orbs)
    evanorm = torch.empty(len(frame))
    for i in range(len(frame)):
        evanorm[i] = torch.mean((torch.linalg.eigvalsh(fock[i]) - torch.linalg.eigvalsh(predicted[i]))**2)/len(fock[i])
    #return torch.sqrt(torch.mean(evanorm))*Hartree 
    return torch.mean(evanorm)*(Hartree)**2

#
testham_msefull = HamModel(test, "cpu")
regularization_dict = {}

for key in blocks.keys:
    regularization_dict[str(key)] = torch.tensor(0)
testham_msefull.train_collective(dataloader, regularization_dict, "LBFGS", 2000, mse_full, 1)
torch.save(testham_msefull.state_dict(), "./models_water/mseful.pt")

testham_eigval = HamModel(test, "cpu")
regularization_dict = {}

for key in blocks.keys:
    regularization_dict[str(key)] = torch.tensor(0)
testham_eigval.train_collective(dataloader, regularization_dict, "LBFGS", 2000, mse_eigvals, 1)
torch.save(testham_eigval.state_dict(), "./models_water/eigval.pt")

regularization_dict = {}
for key in blocks.keys:
    regularization_dict[str(key)] = torch.tensor(0)
testham_indiv = HamModel(test, "cpu")
#testham_indiv.train_individual(dataloader, regularization_dict, "LBFGS", 5000, mse_block_values, 1)
torch.save(testham_indiv.state_dict(), "./models_water/indiv_models.pt")
#testham_indiv.load_state_dict(torch.load("./models_water/indiv_models.pt"))

# ## Evaluation

#Load test set
test_frames1 = ase.io.read("data/water-hamiltonian/water_coords_1000.xyz","800:1000")
#test_frames1 = ase.io.read("data/ethanol-hamiltonian/ethanol_4500.xyz", "4400:4500")
test_frames = test_frames1 #+ frames2
for f in test_frames:
    f.cell = [100,100,100]
    f.positions += 50

test_focks1 = np.load("data/water-hamiltonian/water_saph_orthogonal.npy", allow_pickle=True)[800:1000]
#test_focks1 = np.load("data/ethanol-hamiltonian/ethanol_saph_orthogonal.npy", allow_pickle = True)[4400:4500]
test_focks = test_focks1
#test_focks = np.load("data/water-hamiltonian/water_fock.npy", allow_pickle=True)[50:80]
#test_overlap = np.load("data/water-hamiltonian/water_overlap.npy", allow_pickle=True)[50:80]

#test_orthogonal = []
#for i in range(len(test_focks)): 
#    test_focks[i] = fix_pyscf_l1(test_focks[i],test_frames[i], orbs)
#    test_overlap[i] = fix_pyscf_l1(test_overlap[i],test_frames[i], orbs)
#    test_orthogonal.append(lowdin_orthogonalize(test_focks[i], test_overlap[i]))
#test_focks = np.asarray(test_orthogonal, dtype=np.float64)
    
test_blocks = dense_to_blocks(test_focks, test_frames, orbs)
test_fock_bc = couple_blocks(test_blocks, cg)


# In[146]:


test_rhoi = spex.compute(test_frames)
test_gij = pairs.compute(test_frames)
test_rho1i = acdc_standardize_keys(test_rhoi)
test_rho1i.keys_to_properties(['species_neighbor'])
test_gij =  acdc_standardize_keys(test_gij)
test_rho2i = cg_increment(test_rho1i, test_rho1i, lcut=2, other_keys_match=["species_center"], clebsch_gordan=cg)
test_rho1ij = cg_increment(test_rho1i, test_gij, lcut=2, other_keys_match=["species_center"], clebsch_gordan=cg)

test_features = hamiltonian_features(test_rho2i, test_rho1ij)

from equistore.io import save
save("models_water/test_feature.npz", test_features)
#

# In[148]:


#norm_test_feat = normalize_feats(test_features)
#test_target_path = "./test_fock.npz"
test_feature_path = "models_water/test_feature.npz"
testing = HamiltonianDataset(test_feature_path, test_blocks, test_frames)

from torch.utils.data import DataLoader, BatchSampler, SubsetRandomSampler

#Sampler = torch.utils.data.SubsetRandomSampler(range(1,len(test)+1), generator=None)

test_Sampler = torch.utils.data.sampler.RandomSampler(testing)
test_BSampler = torch.utils.data.sampler.BatchSampler(test_Sampler, batch_size = 200, drop_last = False)

test_dataloader = DataLoader(testing, sampler = test_BSampler, collate_fn = collate_blocks)


def mse_block_values(pred, true):
    true = true.reshape(true.shape[:-1])
    MSE = torch.sum(torch.pow(true - pred,2)) / torch.numel(true)
    return MSE*(Hartree)**2 
    #return torch.sqrt(MSE)*(Hartree) 

def mse_full(pred_blocks, fock,frame, orbs):
    predicted = blocks_to_dense(pred_blocks, frame, orbs)
    #fock = torch.tensor(focks)
    mse_loss = torch.empty(len(frame))
    for i in range(len(frame)):
        mse_loss[i] = ((torch.linalg.norm(fock[i]-predicted[i]))**2)/len(fock[i])
        #print("from mse", i, fock[i], mse_loss[i])
    return torch.mean(mse_loss)*Hartree**2
    #return torch.sqrt(torch.mean(mse_loss))*Hartree

def mse_eigvals(pred_blocks, fock, frame, orbs):
    #fock = torch.tensor(focks)
    predicted = blocks_to_dense(pred_blocks, frame, orbs)
    evanorm = torch.empty(len(frame))
    for i in range(len(frame)):
        evanorm[i] = torch.mean((torch.linalg.eigvalsh(fock[i]) - torch.linalg.eigvalsh(predicted[i]))**2)/len(fock[i])
    return torch.mean(evanorm)*Hartree*2
    #return torch.sqrt(torch.mean(evanorm))*Hartree



dataloader.dataset.currentkey = None
for x_data, y_data, structures in dataloader:
    t_pred = testham_msefull(x_data)
    print ("Train error for mse_full is {}".format(mse_full(t_pred, torch.tensor(focks), structures, orbs)))


test_dataloader.dataset.currentkey = None
for x_data, y_data, structures in test_dataloader:
    pred = testham_msefull(x_data)
    print ("Test error for mse_full is {}".format(mse_full(pred, torch.tensor(test_focks), structures, orbs)))


dataloader.dataset.currentkey = None
for x_data, y_data, structures in dataloader:
    t_pred = testham_eigval(x_data)
    print ("Train error for mse_eigval is {}".format(mse_eigvals(t_pred, torch.tensor(focks), structures, orbs)))


test_dataloader.dataset.currentkey = None
for x_data, y_data, structures in test_dataloader:
    pred = testham_eigval(x_data)
    print ("Test error for mse_full is {}".format(mse_eigvals(pred, torch.tensor(test_focks), structures, orbs)))


dataloader.dataset.currentkey = None
for x_data, y_data, structures in dataloader:
    t_pred = testham_indiv(x_data)
    print ("Train error for mse_indiv is {}".format(mse_full(t_pred, torch.tensor(focks), structures, orbs)))


test_dataloader.dataset.currentkey = None
for x_data, y_data, structures in test_dataloader:
    pred = testham_indiv(x_data)
    print ("Test error for mse_indiv is {}".format(mse_full(pred, torch.tensor(test_focks), structures, orbs)))
