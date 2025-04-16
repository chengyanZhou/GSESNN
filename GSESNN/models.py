import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.nn import SortAggregation
from torch_geometric.utils import dropout_edge
from layers import *
from util_functions import *


class GSESNN(torch.nn.Module):
    def __init__(self, dataset, di_size, dr_size, gconv=GCNConv, densegcn=DenseGCNConv, 
                 latent_dim=64, k=30, dropout_n=0.4, dropout_e=0.1,dropout_rate=0.2, force_undirected=False):
        super(GSESNN, self).__init__()

        self.dropout_n = dropout_n
        self.dropout_e = dropout_e
        self.dropout_rate = dropout_rate
        self.force_undirected = force_undirected

        self.conv1 = gconv(dataset.num_features, latent_dim[0])
        self.conv2 = gconv(latent_dim[0], latent_dim[1])
        self.conv3 = densegcn(256, 64)
        self.conv4 = densegcn(64, 16)

        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)

        self.lin_di = Linear(di_size, 256)
        self.bn_di = nn.BatchNorm1d(di_size)
        self.dropout_di = nn.Dropout(p=dropout_rate)

        self.lin_dr = Linear(dr_size, 256)
        self.bn_dr = nn.BatchNorm1d(dr_size)
        self.dropout_dr = nn.Dropout(p=dropout_rate)
        self.fcs = Linear(64, 32)
        self.fcs2 = Linear(32, 1)

        self.sort_pool = SortAggregation(self.k)
        conv1d_channels = [128, 256]

        #self.total_latent_dim = latent_dim[0] + dataset.num_features
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)

        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)
        self.lin2 = Linear(128, 32)
        self.batch_norms = nn.BatchNorm1d(32)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()

        self.fcs.reset_parameters()
        self.fcs2.reset_parameters()

        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, di_sim, dr_sim, drug_adj, dis_adj, p):
        x, edge_index, batch, node= data.x, data.edge_index, data.batch, data.node
        nodes = np.array(node)
 
        edge_index, _ = dropout_edge(
            edge_index, p=self.dropout_e,
            force_undirected=self.force_undirected,
            training=self.training
        )

        x1 = torch.relu(self.conv1(x, edge_index))  
        x2 = torch.relu(self.conv2(x1, edge_index))
        

        X = [x1, x2]
        concat_states = torch.cat(X, 1)
        x = self.sort_pool(concat_states,batch) # batch * (k*hidden) 
 
        x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))  
        x = self.maxpool1d(x)               
        x = F.relu(self.conv1d_params2(x))  
        x = x.view(len(x), -1)  # flatten

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_n, training=self.training)
        x = self.lin2(x)
        x = F.relu(self.batch_norms(x))
        

        di_embedding = self.lin_di(di_sim)
        di_embedding = self.dropout_di(di_embedding)
        di_feat = torch.relu(self.conv3(di_embedding, drug_adj)[0])
        di_out = torch.relu(self.conv4(di_feat, drug_adj)[0])
        di = di_out[nodes[:,0]]

        
        dr_embedding = self.lin_dr(dr_sim)
        dr_embedding = self.dropout_dr(dr_embedding)
        dr_feat = torch.relu(self.conv3(dr_embedding, dis_adj)[0])
        dr_out = torch.relu(self.conv4(dr_feat, dis_adj)[0])
        dr = dr_out[nodes[:, 1]]

        feat_out = torch.cat((di, dr), 1)
        sub_info = x * p
        global_info = feat_out * (1 - p)
        
        preds = F.relu(self.fcs(torch.cat((sub_info, global_info), 1)))
        preds = F.dropout(preds, p=0.4, training=self.training)
        preds = self.fcs2(preds)
        return preds[:, 0]

