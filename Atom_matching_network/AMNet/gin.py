import sys
from pathlib import Path
# Добавляем корень проекта в пути поиска
sys.path.append(str(Path(__file__).parent.parent))  # Поднимаемся на 2 уровня вверх

import torch
from torch.nn import Linear as Lin
from torch_geometric.nn import GINConv
from AMNet.mlp import MLP


class GIN(torch.nn.Module):
    'from https://github.com/rusty1s/deep-graph-matching-consensus/blob/master/dgmc/models/gin.py'
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True):
        super(GIN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, out_channels, 2, batch_norm, dropout=0.0)
            self.convs.append(GINConv(mlp, train_eps=True))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        xs = [x]

        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, batch_norm={}, cat={}, '
                'lin={})').format(self.__class__.__name__, self.in_channels,
                                  self.out_channels, self.num_layers,
                                  self.batch_norm, self.cat, self.lin)

    
if __name__ =='__main__':
    from itertools import product
    model = GIN(16, 32, num_layers=2, batch_norm=True, cat=True, lin=True)
    assert model.__repr__() == ('GIN(16, 32, num_layers=2, batch_norm=True, '
                                'cat=True, lin=True)')

    x = torch.randn(100, 16)
    edge_index = torch.randint(100, (2, 400), dtype=torch.long)
    for cat, lin in product([False, True], [False, True]):
        model = GIN(16, 32, 2, True, cat, lin)
        out = model(x, edge_index)
        print(out.size() == (100, 16 + 2 * 32 if not lin and cat else 32))
        print(out.size() == (100, model.out_channels))