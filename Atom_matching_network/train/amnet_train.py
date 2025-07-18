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
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
silly_smiles = "O=O"
silly_mol = Chem.MolFromSmiles(silly_smiles)
n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

parser = argparse.ArgumentParser()
parser.add_argument('--num_wl_iterations', type = int, default = 3)
parser.add_argument('--node_features_dim', type = int, default = n_node_features)
parser.add_argument('--edge_feature_dim', type = int, default = n_edge_features)
parser.add_argument('--santitize', type = bool, default = False)
parser.add_argument('--embedding_dim', type = int, default=512)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--lr', type=float, default = 0.0001)
parser.add_argument('--n_epochs', type = int, default = 200)
parser.add_argument('--batch_size', type = int, default = 1)

args = parser.parse_args()

train_data = pd.read_csv('SanitizeMol_data/train.csv', names=['reactions', 'edits'])
valid_data =pd.read_csv('SanitizeMol_data/valid.csv', names= ['reactions', 'edits'])

train_dataset = MolGraphDataset(train_data , args.num_wl_iterations, santitize=args.santitize)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False,follow_batch=['x_r', 'x_p'])

validation_dataset = MolGraphDataset(train_data, args.num_wl_iterations, santitize=args.santitize)
validation_loader = DataLoader(validation_dataset, args.batch_size, shuffle=False,follow_batch=['x_r', 'x_p'])

gnn = GIN(args.node_features_dim, args.embedding_dim, num_layers=args.num_layers, cat=True )
model = FMNet(gnn)

print(model)
print(args)


gnn =  gnn.to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.999))

best_valid_loss = float('inf')
patience = 20  # Number of epochs to wait for improvement
counter = 0  # Counter to keep track of epochs without improvement





def train(train_loader):
    model.train()
    total_loss = total_nodes = total_correct = 0
    for i, data in enumerate(train_loader):   
        optimizer.zero_grad()
        data = data.to(device)
        #print(i)
        M_hat = model(data.x_r, data.edge_index_r, data.edge_feat_r,
              data.x_p, data.edge_index_p, data.edge_feat_p,
              data.batch)  
       
        
        M = model.symmetrywise_correspondence_matrix(M_hat,data.eq_as,data.rp_mapper)
        loss = model.loss(M, data.y_r,  data.rp_mapper)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total_correct += model.acc(M , data.y_r, data.rp_mapper, reduction='sum')
        total_nodes += data.y_r.size(0)
     
        
    return total_loss/len(train_loader), total_correct /total_nodes

def validation_loss(loader):
    model.eval()
    total_loss = total_nodes = total_correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            M_hat = model(data.x_r, data.edge_index_r, data.edge_feat_r,
              data.x_p, data.edge_index_p, data.edge_feat_p,
              data.batch)
        
        
            M = model.symmetrywise_correspondence_matrix(M_hat, data.eq_as, data.rp_mapper)
            loss = model.loss(M, data.y_r, data.rp_mapper)
            total_loss += loss.item()

            total_correct += model.acc(M, data.y_r, data.rp_mapper, reduction='sum')
            total_nodes += data.y_r.size(0)
            
            
    return total_loss/len(loader), total_correct/total_nodes

all_train_loss = []
all_train_acc = []

all_valid_loss = []
all_valid_acc = []

for epoch in tqdm(range(1, args.n_epochs+1)):
    print(f'Epoch: {epoch:02d}', 5*'*')
    train_loss , train_acc = train(train_loader)
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}')
    valid_loss , valid_acc  = validation_loss(validation_loader)

    all_train_loss.append(train_loss)
    all_train_acc.append(train_acc)

    all_valid_loss.append(valid_loss)
    all_valid_acc.append(valid_acc)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        counter = 0  
    else:
        counter += 1  
        
    if counter >= patience:
        print(f'Early stopping. No improvement in {epoch} epochs.')
        break
    
path = 'experiment1/05_10_edege'

if not os.path.exists(path):
    os.makedirs(path)

torch.save(model.state_dict(), f'{path}/model.pth')


with open(f'{path}/losses_train.txt', 'wb') as file:
       pickle.dump(all_train_loss, file)
       
with open(f'{path}/acces_train.txt', 'wb') as file:
       pickle.dump(all_train_acc, file)


with open(f'{path}/losses_valid.txt', 'wb') as file:
       pickle.dump(all_valid_loss, file)

with open(f'{path}/acces_valid.txt', 'wb') as file:
       pickle.dump(all_valid_acc, file)
