import sys
from pathlib import Path
# Добавляем корень проекта в пути поиска
sys.path.append(str(Path(__file__).parent.parent))  # Поднимаемся на 2 уровня вверх

import argparse
import torch
from torch_geometric.data import DataLoader
from AMNet.gin import GIN
import pandas as pd
from AMNet.amnet import FMNet
import rdkit.Chem as Chem
from dataset.molgraphdataset import *
import pickle
import os
import pickle
from utils.plots import *
from utils.utils import get_predicted_atom_mapping, get_acc_on_test

print(20*'-')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

silly_smiles = "O=O"
silly_mol = Chem.MolFromSmiles(silly_smiles)
n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

parser = argparse.ArgumentParser()
parser.add_argument('--num_wl_iterations', type = int, default = 3)
parser.add_argument('--node_features_dim', type = int, default = n_node_features)
parser.add_argument('--edge_feature_dim', type = int, default = None)
parser.add_argument('--santitize', type = bool, default = False)
parser.add_argument('--embedding_dim', type = int, default=512)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--batch_size', type = int, default = 1)

args = parser.parse_args()
test_data = pd.read_csv('SanitizeMol_data/test.csv')
iss = [13,257,526,860,888,1202,1733]#small molecule 
i = 26-1
test_dataset = MolGraphDataset(test_data[i:i+1], args.num_wl_iterations, santitize=args.santitize)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,follow_batch=['x_r', 'x_p'])


gnn = GIN(args.node_features_dim, args.embedding_dim, num_layers=args.num_layers, cat=True )
model = FMNet(gnn)

print(device)
print(model)
print(args)

gnn =  gnn.to(device)
model = model.to(device)

path = 'experiment1/updated_acc'

model.load_state_dict(torch.load(f'experiment1/05_10_edege/model.pth',map_location=torch.device('cpu')))


data = next(iter(test_loader))
data = data.to(device)

print(data.eq_as[0])
print(20*'*')

model.eval()

    
M  = model(data.x_r, data.edge_index_r, data.edge_feat_r,
                        data.x_p, data.edge_index_p, data.edge_feat_p,
                                data.batch) 

pred= get_predicted_atom_mapping(M, data)
acc = get_acc_on_test(pred,data)
print('acc:', acc) 
h1 = model.hits_at_k( 1, M, data.y_r, data.rp_mapper,reduction='mean')
print('h1:',h1)
h3 = model.hits_at_k( 3, M, data.y_r, data.rp_mapper,reduction='mean')
print('h3:',h3)
h5 = model.hits_at_k( 5, M, data.y_r,  data.rp_mapper,reduction='mean')
print('h5:',h5)
h10 = model.hits_at_k( 10, M, data.y_r, data.rp_mapper,reduction='mean')
print('h10:',h10)
plot_M(M,i, 'M')

def entropy_confidence(M0):
    entropies = []
    M0_np = M0.cpu().detach().numpy()
    for row in M0_np:
        entropy = -np.sum(row * np.log(row + 1e-10))  # +1e-10 чтобы избежать log(0)
        entropies.append(entropy)
    max_entropy = np.log(M0_np.shape[1])  # Максимально возможная энтропия
    return 1 - np.mean(entropies)/max_entropy

print(entropy_confidence(M))

print('gt, atom mapping', data.rp_mapper)

print('pred, atom mapping', pred)

