import os
from PIL import Image
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import Dataset

class GraphImageDataset(Dataset):
    def __init__(self, plot_paths, safety_paths, smiles_data, transform=None, max_nodes=100):
        self.plot_paths = plot_paths
        self.safety_paths = safety_paths
        self.smiles_data = smiles_data
        self.keys = [key for key in plot_paths.keys() if key in safety_paths and key in smiles_data and key in plot_paths]
        self.transform = transform
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        ir_filename = os.path.basename(self.plot_paths[key]['nist_plot'])
        c13_nmr_filename = os.path.basename(self.safety_paths[key]['Screenshots']['13C NMR'])
        h1_nmr_filename = os.path.basename(self.safety_paths[key]['Screenshots']['1H NMR'])

        ir_path = f'../Data/IR and MS Plots/IR Plots/No Axes/{ir_filename}'
        c13_nmr_path = f'../Data/NMR Data/Screenshots/13C_NMR/{c13_nmr_filename}'
        h1_nmr_path = f'../Data/NMR Data/Screenshots/1H_NMR/{h1_nmr_filename}'

        ir_image = Image.open(ir_path).convert('L')
        c13_nmr_image = Image.open(c13_nmr_path).convert('L')
        h1_nmr_image = Image.open(h1_nmr_path).convert('L')

        if self.transform:
            ir_image = self.transform(ir_image)
            c13_nmr_image = self.transform(c13_nmr_image)
            h1_nmr_image = self.transform(h1_nmr_image)

        smiles = self.smiles_data[key]['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        graph = self.get_graph(mol)

        return ir_image, c13_nmr_image, h1_nmr_image, graph

    def get_graph(self, mol):
        AllChem.Compute2DCoords(mol)
        coords, atom_types, hybridizations, aromaticities = [], [], [], []
        index_map = {}
        atoms = [atom for atom in mol.GetAtoms()]
        for i, atom in enumerate(atoms):
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            x, y = pos.x, pos.y
            coords.append([x, y])
            atom_types.append(atom.GetSymbol())
            hybridizations.append(atom.GetHybridization())
            aromaticities.append(atom.GetIsAromatic())
            index_map[atom.GetIdx()] = i

        n = len(atom_types)
        edges = np.zeros((n, n), dtype=int)
        ring_info = mol.GetRingInfo()
        for bond in mol.GetBonds():
            s = index_map[bond.GetBeginAtomIdx()]
            t = index_map[bond.GetEndAtomIdx()]
            bond_type = bond.GetBondTypeAsDouble()
            edges[s, t] = bond_type
            edges[t, s] = bond_type

        # Determine ring membership
        rings = ring_info.AtomRings()
        ring_bonds = set()
        for ring in rings:
            for i in range(len(ring)):
                s = index_map[ring[i]]
                t = index_map[ring[(i + 1) % len(ring)]]
                ring_bonds.add((min(s, t), max(s, t)))

        # Pad or truncate node and edge features to max_nodes
        coords = np.pad(coords, ((0, self.max_nodes - n), (0, 0)), 'constant')[:self.max_nodes]
        edges = np.pad(edges, ((0, self.max_nodes - n), (0, self.max_nodes - n)), 'constant')[:self.max_nodes, :self.max_nodes]
        atom_types = atom_types + [''] * (self.max_nodes - n)  # Pad atom types
        hybridizations = hybridizations + [Chem.rdchem.HybridizationType.UNSPECIFIED] * (self.max_nodes - n)  # Pad hybridizations
        aromaticities = aromaticities + [False] * (self.max_nodes - n)  # Pad aromaticities

        graph = {
            'coords': np.array(coords),
            'atom_types': atom_types,
            'hybridizations': hybridizations,
            'aromaticities': aromaticities,
            'edges': edges,
            'ring_bonds': ring_bonds,
            'num_atoms': len(atom_types)
        }
        return graph

