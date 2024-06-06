import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

# Constants
ELEMENTS = ['O', 'C', 'H', 'N', 'Br', 'Cl', 'S', 'Si']
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP]
NUM_HYBRIDIZATIONS = len(HYBRIDIZATIONS)
NUM_ELEMENTS = len(ELEMENTS)
BATCH_SIZE = 32

from Encoder_Decoder import*


class BondPredictionModel(nn.Module):
    def __init__(self, img_channels=1, d_model=512, num_heads=8, num_layers=6, dim_feedforward=2048, max_nodes=96, dropout=0.1):
        super(BondPredictionModel, self).__init__()

        self.d_model = d_model
        self.max_nodes = max_nodes

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, d_model, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

        # Decoder
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

        # Node Predictions
        self.fc_out_atom_types = nn.Linear(d_model, max_nodes * NUM_ELEMENTS)  # Predict atom types
        self.fc_out_hybridizations = nn.Linear(d_model, max_nodes * NUM_HYBRIDIZATIONS)  # Predict hybridizations
        self.fc_out_aromaticity = nn.Linear(d_model, max_nodes)  # Predict aromaticity (binary)

        # Edge Predictions
        self.fc_out_bonds = nn.Linear(d_model, max_nodes * max_nodes)  # Predict bond matrix
        self.fc_out_rings = nn.Linear(d_model, max_nodes * max_nodes)  # Predict ring membership (binary)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1, self.d_model)  # Reshape to (batch_size, num_patches, d_model)
        x = x.transpose(0, 1)  # Transpose to (num_patches, batch_size, d_model)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, z):
        z = z.transpose(0, 1)  # Transpose to (batch_size, num_patches, d_model)
        for layer in self.decoder_layers:
            z = layer(z, z)
        z = z.transpose(0, 1)  # Transpose back to (num_patches, batch_size, d_model)
        z = z.mean(dim=0)  # Average over patches
        return z

    def forward(self, ir_img, c13_img, h1_img):
        enc_ir = self.encode(ir_img)
        enc_c13 = self.encode(c13_img)
        enc_h1 = self.encode(h1_img)

        latent_combined = (enc_ir + enc_c13 + enc_h1) / 3  # Combine the latent representations

        decoded = self.decode(latent_combined)

        # Node Predictions
        atom_type_prediction = self.fc_out_atom_types(decoded)
        atom_type_prediction = atom_type_prediction.view(-1, self.max_nodes, NUM_ELEMENTS)  # Reshape to (batch_size, max_nodes, num_elements)

        hybridization_prediction = self.fc_out_hybridizations(decoded)
        hybridization_prediction = hybridization_prediction.view(-1, self.max_nodes, NUM_HYBRIDIZATIONS)  # Reshape to (batch_size, max_nodes, num_hybridizations)

        aromaticity_prediction = self.fc_out_aromaticity(decoded)
        aromaticity_prediction = aromaticity_prediction.view(-1, self.max_nodes)  # Reshape to (batch_size, max_nodes)

        # Edge Predictions
        bond_prediction = self.fc_out_bonds(decoded)
        bond_prediction = bond_prediction.view(-1, self.max_nodes, self.max_nodes)  # Reshape to (batch_size, max_nodes, max_nodes)

        ring_prediction = self.fc_out_rings(decoded)
        ring_prediction = ring_prediction.view(-1, self.max_nodes, self.max_nodes)  # Reshape to (batch_size, max_nodes, max_nodes)

        return atom_type_prediction, hybridization_prediction, aromaticity_prediction, bond_prediction, ring_prediction