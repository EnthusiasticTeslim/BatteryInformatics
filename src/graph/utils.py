import random, os, dgl, pickle, re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch.backends.cudnn as cudnn
from model import GNN

class AverageMeter(object):
    """Computes and stores the average and current value of metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.rmse = self.avg ** (1/2)

def collate(samples):

    """ 
        Batching a list of samples to create a mini-batch.
    Input: 
        samples - list of individual samples
    Output: 
        batched_graph - a mini-batch of graphs
        labels - labels of the graphs
    """
    smiles, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return smiles, batched_graph, torch.tensor(labels).unsqueeze(-1)


def summarize_graph_data(g):
    """ 
        Summarize graph data 
    Input: 
        g - DGLGraph
    Output: 
        node_data - node data
        edge_data - edge data
        adj_mat - adjacency matrix
    """

    node_data = g.ndata['h'].numpy() # get node data
    edge_data = g.edata # get edge data
    adj_mat = g.adjacency_matrix_scipy(transpose=True,return_edge_ids=False) # get adjacency matrix
    adj_mat = adj_mat.todense().astype(np.float32) # convert to dense matrix

    return node_data, edge_data, adj_mat




def extract_numbers(tensor):
    return [float(re.findall(r"[-+]?\d*\.\d+|\d+", str(t))[0]) for t in tensor]

def save_prediction(smiles, y_pred, y_true, fname, stage, path):
    """Save the prediction result."""
    result = pd.DataFrame(
                        data=np.vstack([
                                        [s[0] for s in smiles], 
                                        extract_numbers(y_pred), 
                                        extract_numbers(y_true)]).T, 
                        columns=['smiles','prediction', 'target']
                        )

    result.to_csv(f'{path}/prediction_{stage}_{fname}.csv', index=False)

def save_saliency(saliency_map, fname, stage, path):
    """Save the saliency map."""
    with open(f'{path}/saliency_{stage}_{fname}.pickle', 'wb') as f:
        pickle.dump(saliency_map, f)

def save_checkpoint(state, fname, path):
    """Save a checkpoint of the training."""
    state.pop('optimizer', None)
    torch.save(state, f'{path}/{fname}.pth.tar')


def train(loader, model, loss_fn, optimizer, args):
    """Train the network on the training set."""
    metrics = AverageMeter()
    model.train()
    
    for data in loader:
        _, graph, label = data
        inputs = graph.ndata['h']
        output = model(graph, inputs)

        if args.gpu >= 0:
            label = label.cuda(args.gpu, non_blocking=True)

        loss = loss_fn(output, label.float())
        metrics.update(loss.item(), label.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return metrics.avg, metrics.rmse

def validate(loader, model, args):
    """Evaluate the network on the entire validation set."""
    metrics = AverageMeter()
    model.eval()
    
    with torch.no_grad():
        for data in loader:
            _, graph, label = data
            inputs = graph.ndata['h']
            output = model(graph, inputs)

            if args.gpu >= 0:
                label = label.cuda(args.gpu, non_blocking=False)
          
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, label.float())
            metrics.update(loss.item(), label.size(0))

    return metrics.avg, metrics.rmse

def predict(loader, model, args, fname, stage, path):
    """Evaluate the network on the entire test set."""
    metrics = AverageMeter()
    model.eval()

    smiles, y_pred, y_true, saliency_map = [], [], [], []

    with torch.set_grad_enabled(True):
        for data in loader:
            
            smile, graph, label = data
            inputs = graph.ndata['h']
            output, grad = model(graph, inputs)

            if args.gpu >= 0:
                label = label.cuda(args.gpu, non_blocking=False)

            label = label.view(-1, 1)     
            smiles.append(smile)       
            y_true.append(label.float())
            y_pred.append(output)
            saliency_map.append(grad)

            loss_fn = nn.MSELoss()
            loss = loss_fn(output, label.float())
            metrics.update(loss.item(), label.size(0))

    save_prediction(smiles, y_pred, y_true, fname, stage, path)
    save_saliency(saliency_map, fname, stage, path) 

def to_loader(dataset, collate, batch_size=1, shuffle=False):
    """Create a DataLoader from a dataset."""
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=shuffle)

def setup_environment(args):
    """Setup the environment"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.docker:
        result_dir = f"{args.parent_directory}/model{datetime.now().strftime('%Y%m%d')}"
    else: 
        result_dir = f"{args.parent_directory}/{args.result_directory}/model{datetime.now().strftime('%Y%m%d')}"
    
    if args.train: # if training, create a new directory else use the existing directory
        if os.path.exists(result_dir):
            if input(f"Directory {result_dir} exists. Overwrite? (y/n)") == 'y':
                os.system(f"rm -r {result_dir}")
            else:
                return
            os.makedirs(result_dir)
        else:
            os.makedirs(result_dir)
    else: # raise error if the directory does not exist
        if not os.path.exists(result_dir):
            raise ValueError(f"Directory {result_dir} does not exist.")
    return result_dir

def load_data(args):
    """Load the data"""
    
    if args.docker:
        train_data = pd.read_csv(f'{args.parent_directory}/{args.train_data}')
        test_data = pd.read_csv(f'{args.parent_directory}/{args.test_data}')
    else:
        train_data = pd.read_csv(f'{args.parent_directory}/{args.data_directory}/{args.train_data}')
        test_data = pd.read_csv(f'{args.parent_directory}/{args.data_directory}/{args.test_data}')
    
    for df in [train_data, test_data]:
        if 'smiles' not in df.columns or 'label' not in df.columns:
            raise ValueError('Columns smiles and label are required in both train and test data')
    
    return (train_data.smiles.values, train_data.label.values), (test_data.smiles.values, test_data.label.values)

def setup_model(args, saliency=False):
    """Setup the model"""
    if args.add_features:
        model = GNN(in_dim=args.dim_input, extra_in_dim=args.num_feat, 
                    add_descriptor=args.add_features, hidden_dim=args.unit_per_layer, saliency=saliency)
        model_arch = 'GCNReg_with_descriptor'
    else:
        model = GNN(in_dim=args.dim_input, hidden_dim=args.unit_per_layer, saliency=saliency)
        model_arch = 'GCNReg'
    
    if args.gpu >= 0:
        model = model.cuda(args.gpu)
        cudnn.benchmark = True
    
    return model, model_arch


def test_model(
        graph_it, model, id, train_data, val_data, test_data, args):
    """Test the model on the train, val, test data."""
    # 0. extract data
    (smiles_test, labels_test) = test_data
    (smiles_val, labels_val) = val_data
    (smiles_train, labels_train) = train_data
    # 1. compute prediction 
    # train dataset
    train_dataset = graph_it(smiles=smiles_train, y=labels_train, add_descriptor=args.add_features, extra_in_dim=args.num_feat)
    predict(loader=to_loader(train_dataset, collate), model=model, args=args, fname=id, stage='train_data', path=args.result_dir)
    # valid dataset
    valid_dataset = graph_it(smiles=smiles_val, y=labels_val, add_descriptor=args.add_features, extra_in_dim=args.num_feat)
    predict(loader=to_loader(valid_dataset, collate), model=model, args=args, fname=id, stage='val_data', path=args.result_dir)
    # test dataset
    test_dataset = graph_it(smiles=smiles_test, y=labels_test, add_descriptor=args.add_features, extra_in_dim=args.num_feat)
    predict(loader=to_loader(test_dataset, collate), model=model, args=args, fname=id, stage='test_data', path=args.result_dir)
