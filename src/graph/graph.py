from typing import List, Tuple, Callable, Optional
import torch
import dgl
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import mol_to_bigraph, smiles_to_complete_graph
from rdkit import Chem
import numpy as np

class GraphDataset:
    """
        Graph dataset for regression tasks
    Input: 
            smiles - list of SMILES strings
           y - list of target values
           node_featurizer - node featurizer
           edge_featurizer - edge featurizer
           graph_constructor - graph constructor
           canonical_atom_order - bool, whether to use canonical atom order
    Output: 
            graphs - list of DGLGraphs
            labels - list of target values
    """
    def __init__(self, 
                 smiles: List[str], 
                 y: List[float],
                extra_in_dim: int = 0,
                add_descriptor: bool = False,
                 node_featurizer: Callable = CanonicalAtomFeaturizer(),
                 edge_featurizer: Optional[Callable] = None,
                 graph_constructor: Callable = mol_to_bigraph,
                 canonical_atom_order: bool = False):
        
        self.smiles = smiles
        self.y = y
        self.extra_in_dim = extra_in_dim
        self.add_descriptor = add_descriptor
        self.graph_constructor = graph_constructor
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.canonical_atom_order = canonical_atom_order
        
        self.graphs: List[dgl.DGLGraph] = []
        self.labels: List[torch.Tensor] = []
        self._generate()

    def __len__(self) -> int:
        """Return the number of graphs in the dataset"""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """Get the graph and label at index idx"""
        return self.smiles[idx], self.graphs[idx], self.labels[idx]

    def node_to_atom(self, idx: int) -> List[str]:
        """Get the atom types of the nodes in the graph at index idx"""
        g = self.graphs[idx]
        # allowable_set = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'B', 'Si', 'P',
        #                  'Li', 'Na', 'K', 'Mg', 'Ca', 'Al']
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        
        node_feat = g.ndata['h'].numpy()[:, :len(allowable_set)]
        return [allowable_set[np.where(feat == 1)[0][0]] for feat in node_feat]

    def _generate(self):
        """Generate graphs and labels"""
        for smiles, label in zip(self.smiles, self.y):
            if self.graph_constructor == mol_to_bigraph:
                mol = Chem.MolFromSmiles(smiles)
                g = self.graph_constructor(mol, node_featurizer=self.node_featurizer,
                                           edge_featurizer=self.edge_featurizer,
                                           canonical_atom_order=self.canonical_atom_order)
            elif self.graph_constructor == smiles_to_complete_graph:
                g = self.graph_constructor(smiles, node_featurizer=self.node_featurizer,
                                           edge_featurizer=self.edge_featurizer,
                                           canonical_atom_order=self.canonical_atom_order)
            else:
                raise ValueError("Unsupported graph constructor")
            
            self.graphs.append(g)
            self.labels.append(torch.tensor(label))
