import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdCoordGen, Descriptors

def molecules_from_smiles(smiles):
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



def compute_rdkit_descriptors(molecules, drop_na=False):
    """
    Compute RDKit descriptors for a list of molecules, dropping NaNs by default and returning a DataFrame.
    """
    # Get the list of descriptor names and functions
    desc_list = Descriptors.descList
    
    # Initialize dictionary to store results
    results = {name: [] for name, _ in desc_list}
    
    # Iterate over molecules with a progress bar
    for mol in tqdm(molecules, desc="Computing descriptors"):
        for name, func in desc_list:
            try:
                value = func(mol)
            except:
                value = np.nan
            results[name].append(value)
    
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Drop NaNs if requested
    if drop_na:
        df = df.dropna(axis=1)
    
    return df