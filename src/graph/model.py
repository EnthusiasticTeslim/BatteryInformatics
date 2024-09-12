import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNN(nn.Module):
    ''' GNN model for graph regression
    Input: in_dim - int, input feature dimension
           hidden_dim - int, hidden feature dimension
           num_conv_layers - int, number of graph convolution layers
           num_linear_layers - int, number of linear layers
           saliency - bool, whether to compute saliency maps
    Output: hg - torch.Tensor of shape (n_classes,) and h.grad if saliency is True

    '''
    def __init__(self, 
                 in_dim: int=76, hidden_dim: int=12, 
                 add_descriptor: bool= False, extra_in_dim: int = 0,
                 num_conv_layers=3, num_linear_layers=3, 
                 saliency=False):
        super(GNN, self).__init__()
        
        # Graph Convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(dgl.nn.pytorch.GraphConv(in_dim, hidden_dim)) # input Conv layer
        for _ in range(num_conv_layers - 1): # hidden conv layers
            self.convs.append(dgl.nn.pytorch.GraphConv(hidden_dim, hidden_dim))
        
        # Linear classification layers
        self.classifiers = nn.ModuleList()
        for i in range(num_linear_layers - 1): # hidden layers
            self.classifiers.append(nn.Linear(hidden_dim, hidden_dim))
        self.classifiers.append(nn.Linear(hidden_dim, 1)) # output layer
        
        self.saliency = saliency

    def forward(self, g, device=device):
        '''
        Forward pass of the GNN model
        Input: g - DGLGraph
        Output: hg - torch.Tensor of shape (n_classes,)
            '''
        
        h = g.ndata['h'].float().to(device)
        
        if self.saliency:
            h.requires_grad = True
        
        for conv in self.convs:
            h = F.relu(conv(g, h))
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        
        for i, classifier in enumerate(self.classifiers):
            hg = classifier(hg)
            if i < len(self.classifiers) - 1:
                hg = F.relu(hg)
        
        if self.saliency:
            hg.backward()
            return hg, h.grad
        else:
            return hg
