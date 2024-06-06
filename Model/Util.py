import yaml

def load_yaml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_all_YAML_data(IR_path, NMR_path, SMILES_path):
    IR_data = load_yaml(IR_path)
    NMR_data = load_yaml(NMR_path)
    SMILES_data = load_yaml(SMILES_path)

    return IR_data, NMR_data, SMILES_data

def logging_shapes(train_dataloader, test_dataloader):
    # Print out shapes of one batch from train and test dataloaders
    train_batch = next(iter(train_dataloader))
    test_batch = next(iter(test_dataloader))

    train_shapes = {
        "IR Images Tensor Shape": train_batch[0].shape,
        "C13 NMR Images Tensor Shape": train_batch[1].shape,
        "H1 NMR Images Tensor Shape": train_batch[2].shape,
        "Graph Coords Shape": train_batch[3]['coords'].shape,
        "Graph Edges Shape": train_batch[3]['edges'].shape,
        "Graph Atom Types Shape": train_batch[3]['atom_types'].shape,
        "Graph Hybridizations Shape": train_batch[3]['hybridizations'].shape,
        "Graph Aromaticities Shape": train_batch[3]['aromaticities'].shape,
        "Graph Ring Bonds Shape": train_batch[3]['ring_bonds'].shape
    }

    test_shapes = {
        "IR Images Tensor Shape": test_batch[0].shape,
        "C13 NMR Images Tensor Shape": test_batch[1].shape,
        "H1 NMR Images Tensor Shape": test_batch[2].shape,
        "Graph Coords Shape": test_batch[3]['coords'].shape,
        "Graph Edges Shape": test_batch[3]['edges'].shape,
        "Graph Atom Types Shape": test_batch[3]['atom_types'].shape,
        "Graph Hybridizations Shape": test_batch[3]['hybridizations'].shape,
        "Graph Aromaticities Shape": test_batch[3]['aromaticities'].shape,
        "Graph Ring Bonds Shape": test_batch[3]['ring_bonds'].shape
    }

    return train_shapes, test_shapes