import sys
from pathlib import Path
# Добавляем корень проекта в пути поиска
sys.path.append(str(Path(__file__).parent.parent))  # Поднимаемся на 2 уровня вверх

from localmapper import localmapper
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
parser.add_argument('--edge_feature_dim', type = int, default =n_edge_features)
parser.add_argument('--santitize', type = bool, default = False)
parser.add_argument('--embedding_dim', type = int, default=512)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--batch_size', type = int, default = 1)

args = parser.parse_args()
test_data = pd.read_csv('SanitizeMol_data/test.csv')

test_dataset = MolGraphDataset(test_data, args.num_wl_iterations, santitize=args.santitize)
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

print(20*'*')

model.eval()
all_h1 = []
all_h3 = []
all_h5 = []
all_h10 = []
all_acc = []
all_conf = []

preds = []

def entropy_confidence(M0):
    entropies = []
    M0_np = M0.cpu().detach().numpy()
    for row in M0_np:
        entropy = -np.sum(row * np.log(row + 1e-10)) 
        entropies.append(entropy)
    max_entropy = np.log(M0_np.shape[1])  
    return 1 - np.mean(entropies)/max_entropy

total_loss = total_nodes = total_correct = 0
mapper = localmapper()

for i, data in enumerate(test_loader):
    data = data.to(device)
    # Используем LocalMapper
    result = mapper.get_atom_map(data.rxn, return_dict=True)
    if result[0]['confident']:
        pred = result[0]['mapped_rxn']
        source = "LocalMapper"
        conf = 1.0
    else:
        with torch.no_grad():
            M  = model( data.x_r,data.edge_index_r,data.edge_feat_r,
                                data.x_p, data.edge_index_p,data.edge_feat_p,
                                 data.batch) 
                
            pred= get_predicted_atom_mapping(M, data)
            acc = get_acc_on_test(pred,data)
            preds.append(pred)
            all_acc.append(acc) 
            h1 = model.hits_at_k( 1, M, data.y_r,   data.rp_mapper,reduction='mean')
            all_h1.append(h1)
            h3 = model.hits_at_k( 3, M, data.y_r,   data.rp_mapper,reduction='mean')
            all_h3.append(h3)
            h5 = model.hits_at_k( 5, M, data.y_r,  data.rp_mapper,reduction='mean')
            all_h5.append(h5)
            h10 = model.hits_at_k( 10, M, data.y_r,  data.rp_mapper,reduction='mean')
            all_h10.append(h10)
            confidence = entropy_confidence(M)
            all_conf.append(confidence)

# Преобразуем список в numpy array
accuracies = np.array(all_acc)

# Рассчитываем среднюю точность
mean_accuracy = np.mean(accuracies)
print(f"Средняя точность модели: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")

# Дополнительная статистика
print(f"Медианная точность: {np.median(accuracies):.4f}")
print(f"Стандартное отклонение: {np.std(accuracies):.4f}")
print(f"Минимальная точность: {np.min(accuracies):.4f}")
print(f"Максимальная точность: {np.max(accuracies):.4f}")
    
path = 'experiment1/updated_acc'

if not os.path.exists(path):
    os.makedirs(path)

with open(f'{path}/all_test_acc.txt', 'w', encoding='utf-8') as file:
    for acc in all_acc:
        file.write(f"{acc}\n") 

with open(f'{path}/all_test_h1.txt', 'wb') as file:
        pickle.dump(all_h1 , file)

with open(f'{path}/all_test_h3.txt', 'wb') as file:
        pickle.dump(all_h3 , file)

with open(f'{path}/all_test_h5.txt', 'wb') as file:
        pickle.dump(all_h5 , file)

with open(f'{path}/all_test_h10.txt', 'wb') as file:
        pickle.dump(all_h10 , file)

with open(f'{path}/preds.txt', 'wb') as file:
        pickle.dump(preds , file)

with open(f'{path}/confs.txt', 'w', encoding='utf-8') as file:
    for conf in all_conf:
        file.write(f"{conf}\n")   
