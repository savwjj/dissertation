# 构造超图的网络


#-*- coding:utf-8 -*-
import torch
import pandas as pd
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch_geometric.nn import GraphConv,TopKPooling, ResGatedGraphConv, EdgePooling, GatedGraphConv
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.norm import LayerNorm
import torch.nn.functional as F     #在 PyTorch 中，torch.nn.functional 模块包含了许多神经网络的函数，比如激活函数（如 ReLU、Sigmoid）、损失函数（如交叉熵损失）、池化函数、归一化函数等
import numpy as np




class FinalModel(torch.nn.Module):
    def __init__(self, channel, num_features):
        super(FinalModel, self).__init__()
        # 第一层图卷积
        self.conv01 = GCNConv(num_features, channel)
        self.bn01 = torch.nn.BatchNorm1d(channel)
        self.pool1 = TopKPooling(channel, ratio=0.6)

        # 第二层图卷积
        self.conv11 = GCNConv(channel, channel)
        self.bn11 = torch.nn.BatchNorm1d(channel)
        self.pool2 = TopKPooling(channel)

        # 第三层图卷积
        self.conv21 = GCNConv(channel, channel)
        self.bn21 = torch.nn.BatchNorm1d(channel)
        self.pool3 = TopKPooling(channel)

        # 定义线性层
        self.lin1 = torch.nn.Linear(channel * 2, channel)
        self.lin2 = torch.nn.Linear(channel, channel // 2)
        self.lin3 = torch.nn.Linear(channel // 2, 1)
        self.act1 = torch.nn.ReLU()#ELU
        self.act2 = torch.nn.ReLU()#ELU

        self.dropout = torch.nn.Dropout(p=0.5)
    #吴京晶添加，原本在外边
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv01(x, edge_index)
        x1 = self.bn01(x1)
        x1 = F.relu(x1)
        x1, edge_index, _, batch, _, _ = self.pool1(x1, edge_index, None, batch)
   
        x2 = self.conv11(x1, edge_index)
        x2 = self.bn11(x2)
        x2 = F.relu(x2)
        x2, edge_index, _, batch, _, _ = self.pool2(x2, edge_index, None, batch)

        x3 = self.conv21(x2, edge_index)
        x3 = self.bn21(x3)
        x3 = F.relu(x3)
        x3, edge_index, _, batch, _, _ = self.pool3(x3, edge_index, None, batch)

        out = torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch)], dim=1)
        
        out = self.lin1(out)
        out = self.act1(out)
        out = self.lin2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.lin3(out).squeeze(1)  # 输出对数风险比，不需要归一化
        
        return out

    



# def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x1 = self.conv01(x, edge_index)
#         x1 = self.bn01(x1)
#         x1 = F.relu(x1)
#         x1, edge_index, _, batch, _, _ = self.pool1(x1, edge_index, None, batch)
   
#         x2 = self.conv11(x1, edge_index)
#         x2 = self.bn11(x2)
#         x2 = F.relu(x2)
#         x2, edge_index, _, batch, _, _ = self.pool2(x2, edge_index, None, batch)

#         x3 = self.conv21(x2, edge_index)
#         x3 = self.bn21(x3)
#         x3 = F.relu(x3)
#         x3, edge_index, _, batch, _, _ = self.pool3(x3, edge_index, None, batch)

#         out = torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch)], dim=1)
        
#         out = self.lin1(out)
#         out = self.act1(out)
#         out = self.lin2(out)
#         out = self.act2(out)
#         out = self.dropout(out)
#         out = self.lin3(out).squeeze(1)  # 输出对数风险比，不需要归一化
        
#         return out