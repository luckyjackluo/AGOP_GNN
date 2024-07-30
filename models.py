import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.pool import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        if task in ["node_classify", "graph_classify"]:
            classify = True
        else:
            classify = False
        
        self.classify = classify
        
        if not classify:
            #self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.lin = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.conv3(x, edge_index)
        
        if self.classify:
            return x
        else:
            x = global_mean_pool(x, batch=batch)
            return self.lin(x)
    
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        
        if task in ["node_classify", "graph_classify"]:
            classify = True
        else:
            classify = False
        
        self.classify = classify
        
        if not classify:
            #self.conv3 = GATConv(hidden_dim, hidden_dim)
            self.lin = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.conv3 = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = self.conv3(x, edge_index)
        
        if self.classify:
            return x
        else:
            x = global_mean_pool(x, batch=batch)
            return self.lin(x)
    
    
    

