import os
import requests
import logging
import pandas as pd
from rdkit import Chem
import argparse
import selfies as sf
import time
import yaml
import socket
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from rdkit.Chem import rdMolHash, rdMolDescriptors
import colorlog

SMARTS_PATTERNS = {
    'alkene': '[CX3]=[CX3]', 'alkyne': '[CX2]#C',
    'alcohols': '[#6][OX2H]', 'amines': '[NX3;H2,H1;!$(NC=O)]',
    'nitriles': '[NX1]#[CX2]', 'aromatics': '[$([cX3](:*):*),$([cX2+](:*):*)]',
    'alkyl halides': '[#6][F,Cl,Br,I]', 'esters': '[#6][CX3](=O)[OX2H0][#6]',
    'ketones': '[#6][CX3](=O)[#6]', 'aldehydes': '[CX3H1](=O)[#6]',
    'carboxylic acids': '[CX3](=O)[OX2H1]', 'ether': '[OD2]([#6])[#6]',
    'acyl halides': '[CX3](=[OX1])[F,Cl,Br,I]', 'amides': '[NX3][CX3](=[OX1])[#6]',
    'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]', 'imine': '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]',
    'enol': '[OX2H][#6X3]=[#6]', 'hydrazone': '[NX3][NX2]=[*]', 'enamine': '[NX3][CX3]=[CX3]',
    'phenol': '[OX2H][cX3]:[c]'
}

CHECKPOINT_FILE = "checkpoint.yaml"


def setup_logging(data_dir, log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(data_dir, log_file))

    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'PROGRESS': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }
    )

    file_formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def cas_to_smiles(cas_number, max_retries=1, retry_delay=2, error_500_delay=300):
    url = f"http://cactus.nci.nih.gov/chemical/structure/{cas_number}/smiles"

    for attempt in range(1, max_retries + 1):
        try:
            response = urlopen(url)
            return response.read().decode('utf-8').strip()
        except HTTPError as e:
            if e.code == 500:
                logging.error(f"HTTP Error 500 when converting CAS number {cas_number}: {e.reason}.")
                return None
            else:
                logging.error(f"HTTP Error {e.code} when converting CAS number {cas_number}: {e.reason}")
                return None
        except (URLError, socket.error) as e:
            if "Connection reset by peer" in str(e):
                logging.info(f"Retrying... Attempt {attempt} of {max_retries} for CAS number {cas_number}")
                time.sleep(retry_delay)
            else:
                logging.error(f"Non-retriable error when converting CAS number {cas_number}: {e}")
                return None
        except Exception as e:
            logging.error(f"Unexpected error when converting CAS number {cas_number}: {e}")
            return None

    logging.error(f"Failed to convert CAS number {cas_number} after {max_retries} attempts")
    return None


def smiles_to_selfies(smiles):
    try:
        encoded_selfies = sf.encoder(smiles)
        return encoded_selfies
    except Exception as e:
        logging.error(f"Error converting SMILES to SELFIES: {e}")
        return None


def match_smarts(smiles, func_grp_smarts):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        present_groups = []
        for func_name, func_smarts in func_grp_smarts.items():
            func_mol = Chem.MolFromSmarts(func_smarts)
            if func_mol and mol.HasSubstructMatch(func_mol):
                present_groups.append(func_name)
        return present_groups
    except Exception as e:
        logging.error(f"Error matching SMARTS: {e}")
        return None


def get_het_atom_tautomer_v2(molecule):
    try:
        if 'HetAtomTautomerv2' in rdMolHash.HashFunction.names:
            result = rdMolHash.MolHash(molecule, rdMolHash.HashFunction.names['HetAtomTautomerv2'])
            return result
        else:
            return None
    except Exception as e:
        logging.error(f"Error getting tautomer hash: {e}")
        return None


def save_all_to_yaml(data_dir, output_data, cas_id):
    try:
        file_path = os.path.join(data_dir, "all_data_SMARTS_FINAL_2024.yaml")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                existing_data = yaml.safe_load(file) or {}
        else:
            existing_data = {}

        existing_data.update(output_data)

        with open(file_path, 'w') as file:
            yaml.dump(existing_data, file)
    except Exception as e:
        logging.error(f"Error saving data to YAML: {e}")


def load_yaml_data(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                data = yaml.safe_load(file)
                if data is None:
                    return {}
                return data
        return {}
    except Exception as e:
        logging.error(f"Error loading YAML data: {e}")
        return {}


def setup_arguments():
    """
    Set up command-line arguments for the script.

    Returns:
    argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='/Users/rudrasondhi/Desktop/Specto/Specto/Data_FINAL', help="Directory path to store scrapped data")
    parser.add_argument('--cas_csv', default='/Users/rudrasondhi/Desktop/Specto/Specto/Data/filtered_molecules.csv', help="CSV file containing CAS number and formula of molecules")
    return parser.parse_args()


def save_skipped_molecules(data_dir, cas_id):
    try:
        skipped_molecules_file = os.path.join(data_dir, 'skipped_molecules.txt')
        with open(skipped_molecules_file, 'a') as file:
            file.write(f"{cas_id}\n")
    except Exception as e:
        logging.error(f"Error saving skipped molecule {cas_id}: {e}")


def save_checkpoint(data_dir, cas_id, combined_data):
    try:
        checkpoint_data = {'last_processed_cas': cas_id, 'data': combined_data}
        with open(os.path.join(data_dir, CHECKPOINT_FILE), 'w') as file:
            yaml.dump(checkpoint_data, file)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")


def load_checkpoint(data_dir):
    try:
        checkpoint_file = os.path.join(data_dir, CHECKPOINT_FILE)
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as file:
                checkpoint_data = yaml.safe_load(file)
                return checkpoint_data['last_processed_cas'], checkpoint_data['data']
        return None, {}
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return None, {}


def main():
    combined_data = {}
    cas_id = None
    args = setup_arguments()
    assert os.path.isfile(args.cas_csv), f"No file named {args.cas_csv} exists"

    data_dir = os.path.join(args.save_dir, 'IR_Spectra')
    os.makedirs(data_dir, exist_ok=True)
    setup_logging(data_dir, 'scrape.log')

    species_df = pd.read_csv(args.cas_csv)
    valid_cas_df = species_df[species_df['CAS Number'].str.contains('-', na=False)]
    cas_numbers = valid_cas_df['CAS Number'].tolist()
    output_yaml = os.path.join(data_dir, 'all_data_SMARTS_FINAL_2024.yaml')
    output_data = load_yaml_data(output_yaml)

    # Load checkpoint if it exists
    last_processed_cas, checkpoint_data = load_checkpoint(data_dir)
    combined_data.update(checkpoint_data)
    if last_processed_cas:
        start_index = cas_numbers.index(last_processed_cas) + 1
    else:
        start_index = 0

    counter = start_index
    logging.info(f"Starting from checkpoint: %s" % counter)

    for cas_id in cas_numbers[start_index:]:
        smiles = cas_to_smiles(cas_id)
        if smiles is None:
            save_skipped_molecules(data_dir, cas_id)
            counter += 1
            #logging.info(f"{cas_id} skipped. (SMILES ERROR)")
            continue

        selfies = smiles_to_selfies(smiles)
        if selfies is None:
            save_skipped_molecules(data_dir, cas_id)
            #logging.info(f"{cas_id} skipped. (SELFIES ERROR)")
            counter += 1
            continue

        functional_group = match_smarts(smiles, SMARTS_PATTERNS)
        if functional_group is None or not functional_group:
            save_skipped_molecules(data_dir, cas_id)
            #logging.info(f"{cas_id} skipped. (FG ERROR)")
            counter += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        empirical_formula = rdMolDescriptors.CalcMolFormula(mol)
        tautomer = get_het_atom_tautomer_v2(mol)

        output_data[cas_id] = {
            'SMILES': smiles,
            'SELFIES': selfies,
            'Tautomer': tautomer,
            'Functional Groups': functional_group,
            'Empirical Formula': empirical_formula
        }
        combined_data.update(output_data)

        counter += 1

        if counter % 250 == 0:
            logging.info(f'Saving data for batch ending with CAS ID {cas_id}')
            save_all_to_yaml(data_dir, combined_data, "batch_" + str(counter))
            save_checkpoint(data_dir, cas_id, combined_data)
            combined_data = {}
            logging.info(f'Finished saving data for batch ending with CAS ID {cas_id}')

    if combined_data:
        logging.info('Saving remaining data...')
        save_all_to_yaml(data_dir, combined_data, "final_batch")
        save_checkpoint(data_dir, cas_id, combined_data)
        logging.info('Done Saving.')


if __name__ == "__main__":
    main()
