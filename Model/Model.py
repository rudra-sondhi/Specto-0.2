import yaml
import argparse
import logging
import os
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import math
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable "WARNING: not removing hydrogen atom without neighbors"
logging.getLogger("PIL").setLevel(logging.WARNING)


from Graph_Neural_Network import calculate_max_nodes
from GraphImageDataset import *
from Util import*
from Encoder_Decoder import*
from BondPredictionModel import*

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up logging to file and console
log_file = 'application.log'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logger = logging.getLogger()

# Constants
ELEMENTS = ['O', 'C', 'H', 'N', 'Br', 'Cl', 'S', 'Si']
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP]
NUM_HYBRIDIZATIONS = len(HYBRIDIZATIONS)
NUM_ELEMENTS = len(ELEMENTS)

#Model Specification
BATCH_SIZE = 32
EPOCHS = 32
LR=1e-3

def evaluate_bond_predictor(model, dataloader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for ir_img, c13_img, h1_img, graph in dataloader:
            ir_img = ir_img.to(device)
            c13_img = c13_img.to(device)
            h1_img = h1_img.to(device)

            atom_type_target = graph['atom_types'].to(device)
            hybridization_target = graph['hybridizations'].to(device)
            aromaticity_target = graph['aromaticities'].to(device)
            bond_target = graph['edges'].to(device)
            ring_target = graph['edges'].to(device)

            atom_type_pred, hybridization_pred, aromaticity_pred, bond_pred, ring_pred = model(ir_img, c13_img, h1_img)

            # Calculate losses for each prediction
            atom_type_loss = F.cross_entropy(atom_type_pred.view(-1, NUM_ELEMENTS), atom_type_target.view(-1, NUM_ELEMENTS).argmax(dim=1))
            hybridization_loss = F.cross_entropy(hybridization_pred.view(-1, NUM_HYBRIDIZATIONS), hybridization_target.view(-1, NUM_HYBRIDIZATIONS).argmax(dim=1))
            aromaticity_loss = F.binary_cross_entropy_with_logits(aromaticity_pred, aromaticity_target)
            bond_loss = F.mse_loss(bond_pred, bond_target)
            ring_loss = F.binary_cross_entropy_with_logits(ring_pred, ring_target)

            loss = atom_type_loss + hybridization_loss + aromaticity_loss + bond_loss + ring_loss
            test_loss += loss.item()

    print(f'Test Loss: {test_loss/len(dataloader.dataset)}')


# Training and evaluation functions
def train_bond_predictor(model, dataloader, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        train_loss = 0

        for ir_img, c13_img, h1_img, graph in dataloader:
            ir_img = ir_img.to(device)
            c13_img = c13_img.to(device)
            h1_img = h1_img.to(device)

            atom_type_target = graph['atom_types'].to(device)
            hybridization_target = graph['hybridizations'].to(device)
            aromaticity_target = graph['aromaticities'].to(device)
            bond_target = graph['edges'].to(device)
            ring_target = graph['edges'].to(device)

            optimizer.zero_grad()
            atom_type_pred, hybridization_pred, aromaticity_pred, bond_pred, ring_pred = model(ir_img, c13_img, h1_img)

            # Calculate losses for each prediction
            atom_type_loss = F.cross_entropy(atom_type_pred.view(-1, NUM_ELEMENTS), atom_type_target.view(-1, NUM_ELEMENTS).argmax(dim=1))
            hybridization_loss = F.cross_entropy(hybridization_pred.view(-1, NUM_HYBRIDIZATIONS), hybridization_target.view(-1, NUM_HYBRIDIZATIONS).argmax(dim=1))
            aromaticity_loss = F.binary_cross_entropy_with_logits(aromaticity_pred, aromaticity_target)
            bond_loss = F.mse_loss(bond_pred, bond_target)
            ring_loss = F.binary_cross_entropy_with_logits(ring_pred, ring_target)

            loss = atom_type_loss + hybridization_loss + aromaticity_loss + bond_loss + ring_loss
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        logger.info(f'Epoch {epoch+1}, Loss: {train_loss/len(dataloader.dataset)}')

def collate_fn_factory(max_nodes):
    def collate_fn(batch):
        ir_images, c13_nmr_images, h1_nmr_images, graphs = zip(*batch)

        # Convert lists to numpy arrays before converting to tensors
        coords = [graph['coords'] for graph in graphs]
        edges = [graph['edges'] for graph in graphs]
        atom_types = [graph['atom_types'] for graph in graphs]
        hybridizations = [graph['hybridizations'] for graph in graphs]
        aromaticities = [graph['aromaticities'] for graph in graphs]
        ring_bonds = [graph['ring_bonds'] for graph in graphs]

        coords = np.stack(coords)
        edges = np.stack(edges)

        # Convert to tensors
        atom_types_tensor = torch.zeros((len(graphs), max_nodes, len(ELEMENTS)))
        for i, atoms in enumerate(atom_types):
            for j, atom in enumerate(atoms):
                if atom in ELEMENTS:
                    atom_types_tensor[i, j, ELEMENTS.index(atom)] = 1.0

        hybridizations_tensor = torch.zeros((len(graphs), max_nodes, len(HYBRIDIZATIONS)))
        for i, hybrids in enumerate(hybridizations):
            for j, hybrid in enumerate(hybrids):
                if hybrid in HYBRIDIZATIONS:
                    hybridizations_tensor[i, j, HYBRIDIZATIONS.index(hybrid)] = 1.0

        aromaticities_tensor = torch.tensor(aromaticities, dtype=torch.float32)

        ring_bonds_tensor = torch.zeros((len(graphs), max_nodes, max_nodes), dtype=torch.float32)
        for i, bonds in enumerate(ring_bonds):
            for s, t in bonds:
                ring_bonds_tensor[i, s, t] = 1.0
                ring_bonds_tensor[i, t, s] = 1.0

        return torch.stack(ir_images), torch.stack(c13_nmr_images), torch.stack(h1_nmr_images), {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'edges': torch.tensor(edges, dtype=torch.float32),
            'atom_types': atom_types_tensor,
            'hybridizations': hybridizations_tensor,
            'aromaticities': aromaticities_tensor,
            'ring_bonds': ring_bonds_tensor
        }
    return collate_fn


def image_transformations():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()  # Ensures the tensor values are in the range [0, 1]
    ])
    return transform


def argument_loader():
    parser = argparse.ArgumentParser()
    parser.add_argument('--IR_path', default='../Data/IR and MS Plots/plot_paths.yaml', help="IR plot paths")
    parser.add_argument('--NMR_path', default='../Data/NMR Data/Safety.yaml', help="NMR plot paths")
    parser.add_argument('--SMILES_path', default='../Data/Preprocess Data/All SMILES, SELFIES, Taut.yaml', help="SMILES data")
    args = parser.parse_args()
    return args


def main():
    logger.info("Starting main function")
    args = argument_loader()
    logger.debug(f"Arguments parsed: {args}")

    IR_data, NMR_data, SMILES_data = get_all_YAML_data(args.IR_path, args.NMR_path, args.SMILES_path)

    try:
        max_nodes = calculate_max_nodes(SMILES_data)
        logger.debug(f"Max nodes calculated: {max_nodes}")
    except Exception as e:
        logger.error(f"Error in calculate_max_nodes: {e}")
        return

    transform = image_transformations()

    # Imported from "GraphImageDataset.py"
    dataset = GraphImageDataset(IR_data, NMR_data, SMILES_data, transform=transform, max_nodes=max_nodes)
    logger.info("Dataset created successfully")

    train_keys, test_keys = train_test_split(dataset.keys, test_size=0.2, random_state=42)
    logger.debug("Dataset split into training and testing sets")

    logger.info(f"Number of training examples: {len(train_keys)}")
    logger.info(f"Number of testing examples: {len(test_keys)}")

    # Create datasets for training and testing
    train_dataset = GraphImageDataset({k: dataset.plot_paths[k] for k in train_keys},
                                      {k: dataset.safety_paths[k] for k in train_keys},
                                      {k: dataset.smiles_data[k] for k in train_keys},
                                      transform=transform,
                                      max_nodes=max_nodes)

    test_dataset = GraphImageDataset({k: dataset.plot_paths[k] for k in test_keys},
                                     {k: dataset.safety_paths[k] for k in test_keys},
                                     {k: dataset.smiles_data[k] for k in test_keys},
                                     transform=transform,
                                     max_nodes=max_nodes)

    collate_fn = collate_fn_factory(max_nodes)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    train_shapes, test_shapes = logging_shapes(train_dataloader, test_dataloader) #Found in "util.py"
    logger.info(f"Training shape: {train_shapes}")
    logger.info(f"Testing shape: {test_shapes}")


    logger.info("Starting training...")
    model = BondPredictionModel(max_nodes=max_nodes).to(device)
    train_bond_predictor(model, train_dataloader)
    evaluate_bond_predictor(model, test_dataloader)

if __name__ == "__main__":
    main()
