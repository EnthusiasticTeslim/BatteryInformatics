from tqdm import tqdm  # add a progress bar
from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG

IPythonConsole.ipython_useSVG = True



def make_molecules_from_smiles(smiles):
    '''
    Generate RDKit molecules from SMILES strings
    '''
    molecules = []
    for smilei in tqdm(smiles):
        mol = Chem.MolFromSmiles(smilei)
        rdCoordGen.AddCoords(mol)
        molecules.append(mol)
    return molecules


def canonical_smiles(smiles):
    '''
    Generate canonical SMILES strings
    '''
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))