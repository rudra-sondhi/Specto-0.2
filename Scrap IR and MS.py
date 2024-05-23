

import os
import requests
import argparse
import logging
import pandas as pd
import yaml

def set_logger(model_dir, log_name):
    '''Set logger to write info to terminal and save in a file.

    Args:
        model_dir: (string) path to store the log file

    Returns:
        None
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Don't create redundant handlers every time set_logger is called
    if not logger.handlers:
        # File handler with debug level stored in model_dir/generation.log
        fh = logging.FileHandler(os.path.join(model_dir, log_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
        logger.addHandler(fh)

        # Stream handler with info level written to terminal
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)
    
    return logger

def scrap_data(cas_id, params, data_dir, save_iter, nist_url="https://webbook.nist.gov/cgi/cbook.cgi"):
    '''Collect data from NIST database and store them in jdx format.

    Args:
        cas_id: (string) CAS number of the molecule
        params: (dict) Parameters to include in the request
        data_dir: (string) Path to store the data
        save_iter: (bool) Indicates if saving should be done after every molecule
        nist_url: (string) URL of the NIST database

    Returns:
        dict: Data about the scraped file or False if not found
    '''

    params['JCAMP'] = 'C' + cas_id
    params['Units'] = 'SI'
    #params['Type'] = 'IR-SPEC'  # Assuming this is fixed for simplicity, adjust as needed
    params['Index'] = '0'  # Ensure index is always 0
    #nist_url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID={cas_id}&Units={params['Units']}&Type={params['Type']}&Index={params['Index']}"

    spectra_type_path = os.path.join(data_dir, params['Type'].lower())
    if not os.path.exists(spectra_type_path):
        os.makedirs(spectra_type_path)

    response = requests.get(nist_url, params=params)

    if response.text == '##TITLE=Spectrum not found.\n##END=\n':
        logging.info(nist_url, params)
        return False
    

    file_path = os.path.join(spectra_type_path, cas_id + '.jdx')
    with open(file_path, 'wb') as data:
        data.write(response.content)

    output_data = {cas_id: {'Type': params['Type'].lower(), 'Path': file_path}}
    if save_iter:
        save_all_to_yaml(data_dir, output_data, cas_id)
        logging.info('Saving')

    else: 
        logging.info(f'Created {params["Type"].lower()} spectrum for CAS ID: {cas_id}')
        return output_data

    logging.info(f'Created {params["Type"].lower()} spectrum for CAS ID: {cas_id}')
    return True


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
    try:
        file_path = os.path.join(data_dir, "all_data_MS&IR_PATHS_2.yaml")
        if os.path.exists(file_path):
            # Read the existing data
            with open(file_path, 'r') as file:
                existing_data = yaml.safe_load(file) or {}
        else:
            existing_data = {}

        # Merge the new data with the existing data
        existing_data.update(output_data)

        # Update the YAML file with the merged data
        with open(file_path, 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False)
        logging.info(f'Successfully saved data for CAS ID {cas_id}')

    except Exception as e:
        logging.error(f'Failed to save {cas_id}: {e}')




def scrap_inchi(cas_ls, params, data_dir, nist_url="https://webbook.nist.gov/cgi/cbook.cgi"):
	'''Collect Inchi keys from NIST database and store them in txt format.

    Args:
        cas_ls: (list) CAS ids to download data for
		params: (dict) queries to be added to url
		data_dir: (string) path to store the data

    Returns:
        None
    '''	

	#Create file path for storing inchi keys
	inchi_path = os.path.join(data_dir, 'inchi.txt')
	num_created = 0
	with open(inchi_path,'a') as file:
		content = '{}\t{}\n'.format('cas_id', 'inchi')
		file.write(content)

		for cas_id in cas_ls:
			params['GetInChI'] = 'C' + cas_id
			response = requests.get(nist_url, params=params)

			num_created+=1
			logging.info('Creating InChi key for id: {}. Total keys created {}'.format(cas_id, num_created))
			content = '{}\t{}\n'.format(cas_id,response.content.decode("utf-8"))
			file.write(content)

def read_yaml_cas_ids(filename):
    '''Read CAS IDs from a YAML file

    Args:
        filename: (string) path to the YAML file

    Returns:
        dict: Dictionary of CAS IDs and associated data
    '''
    with open(filename, 'r') as file:
        data_yaml = yaml.load(file, Loader=yaml.FullLoader)
        data = list(data_yaml)
    return data

def main():
    counter = 0
    # Constants
    nist_url = "https://webbook.nist.gov/cgi/cbook.cgi"

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Super Cool Data_2', help="Directory path to store scrapped data")
    parser.add_argument('--cas_list', default='/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/all_data_SMARTS.yaml', help="File containing CAS number and formula of molecules")
    parser.add_argument('--scrap_IR', default=True, help="Whether to download IR or not")
    parser.add_argument('--scrap_MS', default=True, help="Whether to download MS or not")
    parser.add_argument('--scrap_InChi', default=True, help="Whether to download InChi or not")

    #Sports mode
    parser.add_argument('--save_every_molecule', default=False, help="Whether the yaml saves at the end or after every mol.")

    args = parser.parse_args()

    ir_yaml = {}
    ms_yaml = {}
    combined_data = {}

    # Verify CAS list file
    assert os.path.isfile(args.cas_list), f"No file named {args.cas_list} exists"

    # Setup logging
    data_dir = args.save_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    logger = set_logger(data_dir, 'scrap_for_MSIR.log')
    logging.info('Booting')


    # Read CAS IDs from YAML
    cas_dict = read_yaml_cas_ids(args.cas_list)

    logging.info('Booting 2')

    for cas_id in cas_dict:
        # Process data
        if args.scrap_MS:

            params = {'Type': 'Mass'}
            ms = scrap_data(cas_id, params, data_dir, args.save_every_molecule)
            if ms == False:

                continue
            elif isinstance(ms, dict):
                counter += 1
                combined_data.update(ms)


        if args.scrap_IR:
            params = {'Type': 'IR'}

            ir = scrap_data(cas_id, params, data_dir, args.save_every_molecule)
            if ir == False:

                continue
            elif isinstance(ir, dict):

                combined_data.update(ir)

        
        # Check if the counter is divisible by 500
        if counter % 20 == 0:
            logging.info(f'Saving data for batch ending with CAS ID {cas_id}')
            save_all_to_yaml(data_dir, combined_data, "batch_" + str(counter))
            combined_data = {}  # Reset the dictionary after saving

    # Save remaining data if any
    if combined_data:
        logging.info('Saving remaining data...')
        save_all_to_yaml(data_dir, combined_data, "final_batch")
        logging.info('Done Saving. Starting InCHi Scrap')
    

    if args.scrap_InChi:
        params={}
        scrap_inchi(cas_dict, params, data_dir)
    



if __name__ == "__main__":
    main()
