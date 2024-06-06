from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') #Disable "WARNING: not removing hydrogen atom without neighbors"

def calculate_max_nodes(smiles_data):
    max_nodes = 0
    for key in smiles_data.keys():
        smiles = smiles_data[key]['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()
        if num_atoms > max_nodes:
            max_nodes = num_atoms
    return max_nodes