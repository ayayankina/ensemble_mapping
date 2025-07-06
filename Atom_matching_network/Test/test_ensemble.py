import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Для импорта внутренних модулей

import argparse, torch, pickle, os, math
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
import rdkit.Chem as Chem

from AMNet.gin import GIN
from AMNet.amnet import FMNet
from dataset.molgraphdataset import *
from localmapper import localmapper
from utils.utils import get_predicted_atom_mapping, get_acc_on_test

print(20*'-')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Настройки признаков
silly_mol = Chem.MolFromSmiles("O=O")
n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0, 1)))

# === Аргументы
parser = argparse.ArgumentParser()
parser.add_argument('--num_wl_iterations', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--santitize', type=bool, default=False)
parser.add_argument('--tau', type=float, default=0.6)
args = parser.parse_args()

# === Загрузка теста
test_data = pd.read_csv('SanitizeMol_data/test.csv', header=None)
test_dataset = MolGraphDataset(test_data, args.num_wl_iterations, santitize=args.santitize)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=['x_r', 'x_p'])

# === Загрузка 5 моделей AMNet
amnet_models = []
for fold in range(5):
    gnn = GIN(n_node_features, args.embedding_dim, num_layers=args.num_layers, cat=True).to(device)
    model = FMNet(gnn)
    model.to(device)
    model.load_state_dict(torch.load(f"experiment_folds_{fold}/model.pth", map_location=torch.device('cpu')))
    model.eval()
    amnet_models.append(model)

print(f"✅ Загружено {len(amnet_models)} моделей AMNet")

total_loss = total_nodes = total_correct = 0

# === Функция уверенности

def entropy_confidence(M0):
    entropies = []
    M0_np = M0.cpu().detach().numpy()
    for row in M0_np:
        entropy = -np.sum(row * np.log(row + 1e-10))
        entropies.append(entropy)
    max_entropy = np.log(M0_np.shape[1])
    return 1 - np.mean(entropies)/max_entropy

# === Ансамблевое предсказание

all_h1, all_h3, all_h5, all_h10 = [], [], [], []
all_acc, all_conf, preds, sources = [], [], [], []
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
      M_matrices, confidences = [], []
      
      for model in amnet_models:
        with torch.no_grad():
          M = model(data.x_r, data.edge_index_r, data.edge_feat_r,
                      data.x_p, data.edge_index_p, data.edge_feat_p,
                      data.batch)
          confidence = entropy_confidence(M)  
          M_matrices.append(M)
          confidences.append(confidence)
            
          
      best_idx = int(np.argmax(confidences))
      best_M = M_matrices[best_idx]
      pred = get_predicted_atom_mapping(best_M, data)
      source = "AMNet"
      conf = confidences[best_idx]
      # === Метрики
      acc = get_acc_on_test(pred, data)
      all_acc.append(acc)
      preds.append(pred)
    
      h1 = model.hits_at_k(1, best_M, data.y_r, data.rp_mapper, reduction='mean')
      h3 = model.hits_at_k(3, best_M, data.y_r, data.rp_mapper, reduction='mean')
      h5 = model.hits_at_k(5, best_M, data.y_r, data.rp_mapper, reduction='mean')
      h10 = model.hits_at_k(10, best_M, data.y_r, data.rp_mapper, reduction='mean')
    
      all_h1.append(h1)
      all_h3.append(h3)
      all_h5.append(h5)
      all_h10.append(h10)
      all_conf.append(conf)
      sources.append(source)


# === Вывод метрик
accuracies = np.array(all_acc)
mean_acc = np.mean(accuracies)
print(f"\nСредняя точность: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"Медианная точность: {np.median(accuracies):.4f}")
print(f"Стд. отклонение: {np.std(accuracies):.4f}")
print(f"Мин/Макс: {np.min(accuracies):.4f} / {np.max(accuracies):.4f}")

# === Сохранение
path = 'experiment1/ensemble_acc'
os.makedirs(path, exist_ok=True)

with open(f'{path}/all_test_acc.txt', 'w', encoding='utf-8') as f:
    for acc in all_acc:
        f.write(f"{acc}\n")

with open(f'{path}/all_test_h1.txt', 'wb') as f: pickle.dump(all_h1, f)
with open(f'{path}/all_test_h3.txt', 'wb') as f: pickle.dump(all_h3, f)
with open(f'{path}/all_test_h5.txt', 'wb') as f: pickle.dump(all_h5, f)
with open(f'{path}/all_test_h10.txt', 'wb') as f: pickle.dump(all_h10, f)
with open(f'{path}/preds.txt', 'wb') as f: pickle.dump(preds, f)
with open(f'{path}/confs.txt', 'w', encoding='utf-8') as f:
    for conf in all_conf:
        f.write(f"{conf}\n")
with open(f'{path}/sources.txt', 'w', encoding='utf-8') as f:
    for src in sources:
        f.write(f"{src}\n")
