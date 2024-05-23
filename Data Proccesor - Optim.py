import os
import requests
import logging
import pandas as pd
from rdkit import Chem
import argparse
import selfies as sf
import matplotlib.pyplot as plt
import time
import yaml
import json
from rdkit.Chem import rdMolHash
import string
import colorlog
import signal
import sys


from time import sleep

from rdkit.Chem import rdMolDescriptors
import socket
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


# Reference: https://github.com/Ohio-State-Allen-Lab/FTIRMachineLearning/blob/main/cas_inchi.py
SMARTS_PATTERNS = {'alkene':'[CX3]=[CX3]','alkyne':'[CX2]#C',
                   'alcohols':'[#6][OX2H]','amines':'[NX3;H2,H1;!$(NC=O)]', 'nitriles':'[NX1]#[CX2]', 
                   'aromatics':'[$([cX3](:*):*),$([cX2+](:*):*)]','alkyl halides':'[#6][F,Cl,Br,I]', 
                   'esters':'[#6][CX3](=O)[OX2H0][#6]', 'ketones':'[#6][CX3](=O)[#6]','aldehydes':'[CX3H1](=O)[#6]', 
                   'carboxylic acids':'[CX3](=O)[OX2H1]', 'ether': '[OD2]([#6])[#6]','acyl halides':'[CX3](=[OX1])[F,Cl,Br,I]',
                   'amides':'[NX3][CX3](=[OX1])[#6]','nitro':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]', 'imine': '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]', 
                   'enol':'[OX2H][#6X3]=[#6]', 'hydrazone': '[NX3][NX2]=[*]', 'enamine': '[NX3][CX3]=[CX3]', 'phenol': '[OX2H][cX3]:[c]'}

def setup_logging(data_dir, log_file):
    """
    Sets up logging to file and console with colorized console output for specific log levels.

    Args:
    data_dir (str): Directory path for the log file.
    log_file (str): Name of the log file.

    Returns:
    None
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler for logging
    file_handler = logging.FileHandler(os.path.join(data_dir, log_file))

    # Console handler with color
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

    # Formatter for file logging
    file_formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def cas_to_smiles(cas_number, max_retries=1, retry_delay=2, error_500_delay=300):
    """
    Converts a CAS number to a SMILES string with retries for network errors and a delay for HTTP 500 errors.

    Args:
    cas_number (str): CAS number to be converted.
    max_retries (int): Maximum number of retries for network errors.
    retry_delay (int): Delay in seconds before retrying.
    error_500_delay (int): Delay in seconds before retrying after an HTTP 500 error.

    Returns:
    str: SMILES string or None if conversion fails.
    """
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

def fetch_iupac_name(smiles):
    """
    Fetches the IUPAC name for a given SMILES string.

    Args:
    smiles (str): SMILES string of the molecule.

    Returns:
    str: IUPAC name of the molecule or None if an error occurs.
    """
    url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"
    try:
        response = requests.get(url)
        response.raise_for_status()

        iupac_name = response.text.strip()
        
        # Check for HTML content
        if "<html>" in iupac_name.lower() or "<!DOCTYPE html>" in iupac_name.lower():
            logging.warning(f"HTML content received for SMILES '{smiles}'.")
            return None
                # Check for LaTeX markup
        if "$" in iupac_name:
            logging.warning(f"LaTeX markup detected in IUPAC name for SMILES '{smiles}'.")
            return None
        return iupac_name

    except requests.RequestException as e:
        logging.error(f"Error fetching IUPAC name for SMILES '{smiles}': {e}")
        return None
    

def match_smarts(smiles, func_grp_smarts):
    """
    Identifies functional groups in a molecule represented by a SMILES string.

    Args:
    smiles (str): SMILES string representing the molecule.
    func_grp_smarts (dict): Dictionary mapping functional group names to their SMARTS strings.

    Returns:
    list: Names of functional groups present in the molecule, or None if an error occurs.
    """

    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Identify functional groups and return their names
        present_groups = []
        for func_name, func_smarts in func_grp_smarts.items():
            func_mol = Chem.MolFromSmarts(func_smarts)
            if func_mol and mol.HasSubstructMatch(func_mol):
                present_groups.append(func_name)
        return present_groups
    except:
        return None

def get_het_atom_tautomer_v2(molecule):
    """
    Get the tautomer hash of a molecule using the 'HetAtomTautomerv2' hash function.

    Args:
        molecule (Chem.rdchem.Mol): RDKit molecule object.

    Returns:
        str: Tautomer hash of the molecule, or None if the hash function is unavailable.
    """
    # Ensure the hash function 'HetAtomTautomerv2' is available
    if 'HetAtomTautomerv2' in rdMolHash.HashFunction.names:
        # Use the specific hash function on the provided molecule
        result = rdMolHash.MolHash(molecule, rdMolHash.HashFunction.names['HetAtomTautomerv2'])
        return result
    else:
        return None

def save_all_to_yaml(data_dir, output_data, cas_id):
    """
    Save data to a YAML file.

    Args:
    data_dir (str): Directory path where the YAML file will be saved.
    output_data (dict): Data to be saved to the YAML file.
    cas_id (str): CAS ID to be added.

    Returns:
    None
    """
    file_path = os.path.join(data_dir, "all_data_SMARTS_test.yaml")
    if os.path.exists(file_path):
        # Read the existing data
        with open(file_path, 'r') as file:
            existing_data = yaml.safe_load(file) or {}
    else:
        existing_data = {}

    # Update the existing data with the new data
    existing_data.update(output_data)

    # Save the updated data back to the file
    with open(file_path, 'w') as file:
        yaml.dump(existing_data, file)

def load_yaml_data(filepath):
    """
    Load data from a YAML file.

    Args:
    filepath (str): Path to the YAML file.

    Returns:
    dict: Data loaded from the YAML file, or an empty dictionary if the file is empty or does not exist.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            if data is None:
                return {}  # Return empty dict if the file is empty
            return data
    return {}  # Return empty dict if the file does not exist


def setup_arguments():
    """
    Set up command-line arguments for the script.

    Returns:
    argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='/Users/rudrasondhi/Desktop/Specto/Specto/Data', help="Directory path to store scrapped data")
    parser.add_argument('--cas_csv', default='/Users/rudrasondhi/Desktop/Specto/Specto/Data/filtered_molecules.csv', help="CSV file containing CAS number and formula of molecules")
    return parser.parse_args()



def find_start_index(cas_numbers, output_data):
    """
    Determine the starting index in cas_numbers based on the last processed CAS number from the output data.

    Args:
        cas_numbers (list): List of CAS numbers.
        output_data (dict): Dictionary containing processed data.

    Returns:
        int: Starting index in cas_numbers or 0 if no data is processed.
    """

    last_processed_cas = list(output_data.keys())[-1] if output_data else None
    if last_processed_cas:
        try:
            return cas_numbers.index(last_processed_cas) + 1
        except ValueError:
            pass
    return 0

def calculate_time_estimate(num_molecules, seconds_per_molecule):
    """
    Calculate and log the estimated time to process the given number of molecules.

    Args:
        num_molecules (int): Number of molecules to process.
        seconds_per_molecule (float): Seconds required to process each molecule.

    Returns:
        None
    """
    total_seconds = num_molecules * seconds_per_molecule
    hours = total_seconds / 3600
    days = hours / 24
    logging.info(f"{num_molecules} molecules to process. Estimated time: {hours:.2f} hours or {days:.2f} days.")

def log_progress(num_processed, num_remaining, total, seconds_per_molecule):
    """
    Log the progress of processing.

    Args:
        num_processed (int): Number of molecules processed.
        num_remaining (int): Number of molecules remaining to be processed.
        total (int): Total number of molecules.
        seconds_per_molecule (float): Seconds required to process each molecule.

    Returns:
        None
    """

    time_remaining_hours = (num_remaining * seconds_per_molecule) / 3600
    percent_progress = (num_processed / total) * 100
    logging.info(f"{num_remaining} molecules left. Estimated time remaining: {time_remaining_hours:.2f} hours.")
    logging.info(f"Progress: {percent_progress:.2f}%")

def save_statistics(data_dir, num_processed, num_skipped):
    """
    Save the statistics of processed and skipped molecules to a text file.

    Args:
        data_dir (str): Directory path where the statistics file will be saved.
        num_processed (int): Number of molecules processed.
        num_skipped (int): Number of molecules skipped.

    Returns:
        None
    """

    stats_path = os.path.join(data_dir, 'processing_stats.txt')
    with open(stats_path, 'w') as file:
        file.write(f"Processed: {num_processed}\n")
        file.write(f"Skipped: {num_skipped}\n")

def load_statistics(data_dir):
    """
    Load the statistics of processed and skipped molecules from a text file.

    Args:
        data_dir (str): Directory path where the statistics file is located.

    Returns:
        Tuple[int, int]: Number of molecules processed and number of molecules skipped.
    """
    stats_path = os.path.join(data_dir, 'processing_stats.txt')
    try:
        with open(stats_path, 'r') as file:
            num_processed = int(file.readline().split(":")[1].strip())
            num_skipped = int(file.readline().split(":")[1].strip())
            return num_processed, num_skipped
    except Exception:
        return 0, 0

def save_skipped_molecules(data_dir, cas_id):
    """
    Append the CAS ID of skipped molecules to a text file.

    Args:
        data_dir (str): Directory path where the text file will be saved.
        cas_id (str): CAS ID of the skipped molecule.

    Returns:
        None
    """

    skipped_molecules_file = os.path.join(data_dir, 'skipped_molecules.txt')
    with open(skipped_molecules_file, 'a') as file:
        file.write(f"{cas_id}\n")
    
def smiles_to_selfies(smiles):
    """
    Converts a SMILES string to SELFIES representation.

    Args:
    smiles (str): SMILES string of the molecule.

    Returns:
        str: SELFIES representation of the molecule, or None if an error occurs.
    """
    try:
        encoded_selfies = sf.encoder(smiles)
        return encoded_selfies
    except:
        return None

def save_error_statistics(data_dir, error_type):
    """
    Count and store the types of errors (SMILES, SELFIES, FG) encountered.

    Args:
        data_dir (str): Directory path where the error statistics file will be saved.
        error_type (str): Type of error encountered (SMILES, SELFIES, FG).

    Returns:
        None
    """
    error_stats_file = os.path.join(data_dir, 'error_statistics.txt')
    try:
        with open(error_stats_file, 'r') as file:
            error_stats = eval(file.read())
    except Exception:
        error_stats = {'SMILES': 0, 'SELFIES': 0, 'FG': 0}

    error_stats[error_type] += 1

    with open(error_stats_file, 'w') as file:
        file.write(str(error_stats))


def main():
    combined_data = {}
    cas_id = None  # Initialize cas_id to a default value (e.g., None)
    args = setup_arguments()
    assert os.path.isfile(args.cas_csv), f"No file named {args.cas_csv} exists"

    data_dir = os.path.join(args.save_dir, 'IR_Spectra')
    os.makedirs(data_dir, exist_ok=True)
    setup_logging(data_dir, 'scrape.log')

    species_df = pd.read_csv(args.cas_csv)
    valid_cas_df = species_df[species_df['CAS Number'].str.contains('-', na=False)]
    cas_numbers = valid_cas_df['CAS Number'].tolist()
    output_yaml = os.path.join(data_dir, 'all_data_SMARTS_test.yaml')
    output_data = load_yaml_data(output_yaml)

    counter = 0
    for cas_id in cas_numbers:

        smiles = cas_to_smiles(cas_id)
        if smiles is None:
            save_skipped_molecules(data_dir, cas_id)
            counter += 1
            logging.info(f"{cas_id} skipped. (SMILES ERROR)")

            continue
        selfies = smiles_to_selfies(smiles)
        if selfies is None:

            save_skipped_molecules(data_dir, cas_id)
            logging.info(f"{cas_id} skipped. (SELFIES ERROR)")
            counter += 1
            continue

        functional_group = match_smarts(smiles, SMARTS_PATTERNS)
        if functional_group is None or not functional_group:

            save_skipped_molecules(data_dir, cas_id)
            logging.info(f"{cas_id} skipped. (FG ERROR)")
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

        if counter % 500 == 0:
            logging.info(f'Saving data for batch ending with CAS ID {cas_id}')
            save_all_to_yaml(data_dir, combined_data, "batch_" + str(counter))
            combined_data = {}  # Reset the dictionary after saving
    # Save remaining data if any
    if combined_data:
        logging.info('Saving remaining data...')
        save_all_to_yaml(data_dir, combined_data, "final_batch")
        logging.info('Done Saving.')
    


"""
def main():
    cas_id = None  # Initialize cas_id to a default value (e.g., None)

    try:
        args = setup_arguments()
        assert os.path.isfile(args.cas_csv), f"No file named {args.cas_csv} exists"

        data_dir = os.path.join(args.save_dir, 'IR_Spectra')
        os.makedirs(data_dir, exist_ok=True)
        setup_logging(data_dir, 'scrape.log')

        species_df = pd.read_csv(args.cas_csv)
        valid_cas_df = species_df[species_df['CAS Number'].str.contains('-', na=False)]
        cas_numbers = valid_cas_df['CAS Number'].tolist()
        calculate_time_estimate(len(cas_numbers), 4)

        output_yaml = os.path.join(data_dir, 'all_data_SMARTS.yaml')
        output_data = load_yaml_data(output_yaml)
        start_index = find_start_index(cas_numbers, output_data)

        num_processed, num_skipped = load_statistics(data_dir)
        if num_processed != 0 or num_skipped != 0:

            logging.info(f"Processed: {num_processed}, Skipped: {num_skipped}")
            logging.info(f"Start Index: {start_index}")

        start_time = time.time()  # Start time for measuring processing rate
        molecules_processed_since_last_check = 0


        for cas_id in cas_numbers[start_index:]:
            if cas_id in output_data:
                logging.info(f"{cas_id} skipped. (In database)")
                continue

            smiles = cas_to_smiles(cas_id)
            if smiles is None:
                num_skipped += 1
                save_skipped_molecules(data_dir, cas_id)
                save_error_statistics(data_dir, 'SMILES')

                logging.info(f"{cas_id} skipped. (SMILES ERROR)")

                continue

            selfies = smiles_to_selfies(smiles)
            if selfies is None:
                num_skipped += 1
                save_skipped_molecules(data_dir, cas_id)
                logging.info(f"{cas_id} skipped. (SELFIES ERROR)")
                save_error_statistics(data_dir, 'SELFIES')

                continue

            functional_group = match_smarts(smiles, SMARTS_PATTERNS)
            if functional_group is None or not functional_group:
                num_skipped += 1
                save_skipped_molecules(data_dir, cas_id)
                logging.info(f"{cas_id} skipped. (FG ERROR)")
                save_error_statistics(data_dir, 'FG')
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

            save_all_to_yaml(data_dir, output_data, cas_id)
            num_processed += 1
            save_statistics(data_dir, num_processed, num_skipped)

            molecules_processed_since_last_check += 1
            current_time = time.time()
            elapsed_time = current_time - start_time


            if elapsed_time >= 1500:  # 1500 seconds = 25 minutes
                rate = molecules_processed_since_last_check / (elapsed_time / 60)  # Molecules per minute
                logging.info(f"Processing rate: {rate:.2f} molecules/minute.")
                start_time = current_time  # Reset start time for next interval
                molecules_processed_since_last_check = 0

            if num_processed % 700 == 0:
                log_progress(num_processed, len(cas_numbers) - start_index - num_processed, len(cas_numbers), 2.5)
                logging.info(f"Processed: {num_processed}, Skipped: {num_skipped}")
                logging.info("Taking a power name for 3 minutes...")
                time.sleep(120)

    except KeyboardInterrupt:
        # Handle manual interruption (Ctrl+C)
        logging.info("Program interrupted by user. Saving progress before exit.")
    except Exception as e:
        # Handle other exceptions
        logging.error(f"Unexpected error: {e}. Saving progress before exit.")
    finally:
        # This block will always be executed, whether an exception occurred or not
        # Perform final saving and cleanup here
        logging.info("Performing final save and cleanup...")
        save_all_to_yaml(data_dir, output_data, cas_id)
        save_statistics(data_dir, num_processed, num_skipped)
        sys.exit(0)

"""
if __name__ == "__main__":
    main()