import pickle
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import Data


# class Node_to_emb(nn.Module):  # transforms input nodes to an embedding (similar to word embedding in NLP)
#     #### why would an embedding layer be useful?

#     def __init__(self, node_feat_dim=14, node_emb_dim=64):
#         super().__init__()
#         self.emb_dim = node_emb_dim
#         self.node_dim = node_feat_dim
#         self.emb = nn.Linear(self.node_dim, self.emb_dim)
        

#     def forward(self, nodes):
#         assert nodes.size(-1) == self.node_dim, 'wrong input dimension of node features! (node_dim: %f, actual dim:%d )'%(self.node_dim, nodes.size(-1))
#         out = self.emb(nodes)
#         return out

    
# class MpLayer(torch.nn.Module):  # a neural message passing layer
#     def __init__(self, hidden_dim, activation=nn.ReLU()):
#         super(MpLayer, self).__init__()
        
#         #  Hint: which neural networks are used in neural message passing?
        
#         self.edge_network = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
#                                           activation,
#                                           nn.Linear(hidden_dim, hidden_dim),
#                                           activation
#                                           )
        
#         self.node_network = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
#                                           activation,
#                                           nn.Linear(hidden_dim, hidden_dim),
#                                           )
        
#     def forward(self, input_to_layer):
#         node_tensor, edge_idx_tensor = input_to_layer
#         edge_messages_input = torch.concat([node_tensor[edge_idx_tensor[0,:]], node_tensor[edge_idx_tensor[1,:]]], dim=-1) # shape (num_edges, 2*node_dim)
#         edge_messages_output = self.edge_network(edge_messages_input) # shape (num_edges, hidden_dim)
        
#         #now aggregate the edge messages for each node the edge points to:
        
#         node_agg_messages = torch.zeros(node_tensor.size(0), node_tensor.size(1)).to(node_tensor.device)
#         node_agg_messages = node_agg_messages.scatter_add_(
#             dim=0, index=edge_idx_tensor[1].unsqueeze(-1).expand(-1, node_tensor.size(1)), src=edge_messages_output
#         )
        
#         #### why does the aggregation function need to be permutationally invariant? What is another aggregation function
#         #### that could be used?
        
#         #put the aggregated messages through the node update network:
#         node_out = self.node_network(torch.cat([node_tensor, node_agg_messages], dim=-1))

#         return node_out, edge_idx_tensor
        
        

# class MpGNN(torch.nn.Module): # a message passing GNN
#     def __init__(self, node_feat_dim, hidden_dim, activation=nn.ReLU(), num_layers=3, num_classes=2):
#         super(MpGNN, self).__init__()
        
#         #  Hint: the MpGNN must embed the categorical node features, apply message passing layers,
#         #        and finally predict the mutagenicity of each graph in the batch.
        
#         self.node_to_emb = Node_to_emb(node_feat_dim, hidden_dim)
#         self.forward_net = nn.Sequential(
#             *[MpLayer(hidden_dim, activation) for i in range(num_layers)]
#         )
#         self.to_pred = nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.Sigmoid())
        
        

#     def forward(self, x, edge_index, batch):
#         x = self.node_to_emb(x)
# #         print(x.size(), edge_index.size())
#         input_model = (x, edge_index)
#         output_model = self.forward_net(input_model)
#         x,_ = output_model
        
#         out = torch.zeros(max(batch)+1, x.size(1)).to(x.device)
#         idx_aggregate_graph = batch.unsqueeze(-1).expand(-1, x.size(1))
    

#         out.scatter_add_(dim=0, index=idx_aggregate_graph, src=x) # aggregate all node embeddings per graph in the batch
        
#         x = self.to_pred(out)
#         return x

class MpGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, activation=nn.ReLU(), num_layers=3, num_classes=2):
        super(MpGNN, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.activation = activation

    def forward(self, data):
        x, edge_index, edge_attr ,batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1st Graph Convolution layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.activation(x)
        
        # 2nd Graph Convolution layer
        x = self.conv2(x, edge_index, edge_attr)        
        # Global pooling
        x = global_mean_pool(x, edge_index, edge_attr)  # Or use global_max_pool(x, batch)
        
        # Fully connected layer
        x = self.fc(x)
        
        x = torch.sigmoid(x)

        return x