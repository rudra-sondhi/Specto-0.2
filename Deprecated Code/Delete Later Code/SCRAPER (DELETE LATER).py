import os
import requests
import argparse
import logging
import yaml

def set_logger(model_dir, log_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(model_dir, log_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)
    
    return logger

def scrap_data(cas_id, params, data_dir, save_iter, combined_data, nist_url="https://webbook.nist.gov/cgi/cbook.cgi"):
    params['JCAMP'] = 'C' + cas_id

    spectra_type_path = os.path.join(data_dir, params['Type'].lower())
    if not os.path.exists(spectra_type_path):
        os.makedirs(spectra_type_path)

    response = requests.get(nist_url, params=params)
    response_content = response.content

    if response.text == '##TITLE=Spectrum not found.\n##END=\n':
        logging.info(f'Spectrum not found for CAS ID {cas_id} with params {params}')
        return combined_data

    file_path = os.path.join(spectra_type_path, cas_id + '.jdx')
    with open(file_path, 'wb') as data:
        data.write(response_content)

    if cas_id not in combined_data:
        combined_data[cas_id] = {}

    combined_data[cas_id][params['Type'].lower()] = {'Type': params['Type'].lower(), 'Path': file_path}
    
    if save_iter:
        save_all_to_yaml(data_dir, combined_data, cas_id)

    return combined_data

def save_all_to_yaml(data_dir, combined_data, cas_id):
    try:
        file_path = os.path.join(data_dir, "all_data_MS&IR_PATHS.yaml")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                existing_data = yaml.safe_load(file) or {}
        else:
            existing_data = {}

        for key, value in combined_data.items():
            if key in existing_data:
                existing_data[key].update(value)
            else:
                existing_data[key] = value

        with open(file_path, 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False)

        logging.info(f'Successfully saved data for CAS ID {cas_id} in {file_path}')


    except Exception as e:
        logging.error(f'Failed to save {cas_id}: {e}')

def scrap_inchi(cas_ls, params, data_dir, nist_url="https://webbook.nist.gov/cgi/cbook.cgi"):
    inchi_path = os.path.join(data_dir, 'inchi.txt')
    num_created = 0
    with open(inchi_path, 'a') as file:
        content = '{}\t{}\n'.format('cas_id', 'inchi')
        file.write(content)

        for cas_id in cas_ls:
            params['GetInChI'] = 'C' + cas_id
            response = requests.get(nist_url, params=params)

            num_created += 1
            logging.info(f'Creating InChi key for id: {cas_id}. Total keys created {num_created}')
            content = '{}\t{}\n'.format(cas_id, response.content.decode("utf-8"))
            file.write(content)

def read_yaml_cas_ids(filename):
    with open(filename, 'r') as file:
        data_yaml = yaml.load(file, Loader=yaml.FullLoader)
        data = list(data_yaml)
    return data

def get_processed_cas_ids(data_dir):
    file_path = os.path.join(data_dir, "all_data_MS&IR_PATHS.yaml")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file) or {}
        return set(data.keys())
    return set()

def main():
    counter = 0
    nist_url = "https://webbook.nist.gov/cgi/cbook.cgi"

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='/Users/rudrasondhi/Desktop/Specto 0.2/Specto-0.2/Data/Mass and IR', help="Directory path to store scrapped data")
    parser.add_argument('--cas_list', default='/Users/rudrasondhi/Desktop/Specto/Specto/Data_FINAL/IR_Spectra/all_data_SMARTS_FINAL_2024.yaml', help="File containing CAS number and formula of molecules")
    parser.add_argument('--scrap_IR', default=True, help="Whether to download IR or not")
    parser.add_argument('--scrap_MS', default=True, help="Whether to download MS or not")
    parser.add_argument('--scrap_InChi', default=False, help="Whether to download InChi or not")
    parser.add_argument('--save_every_molecule', default=False, help="Whether the yaml saves at the end or after every mol.")
    args = parser.parse_args()

    combined_data = {}

    assert os.path.isfile(args.cas_list), f"No file named {args.cas_list} exists"

    data_dir = args.save_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    logger = set_logger(data_dir, 'scrap_for_MSIR.log')
    logging.info('Booting')

    cas_dict = read_yaml_cas_ids(args.cas_list)
    logging.info('Booting 2')

    processed_cas_ids = get_processed_cas_ids(data_dir)
    logging.info(f'Already processed CAS IDs: {processed_cas_ids}')

    for cas_id in cas_dict:
        if cas_id in processed_cas_ids:
            logging.info(f'Skipping already processed CAS ID: {cas_id}')
            continue

        if args.scrap_MS:
            params = {'Type': 'Mass'}
            combined_data = scrap_data(cas_id, params, data_dir, args.save_every_molecule, combined_data)
            if cas_id in combined_data:
                counter += 1
            else:
                logging.info(f'No Mass spectrum found for CAS ID: {cas_id}')

        if args.scrap_IR:
            params = {'Type': 'IR'}
            combined_data = scrap_data(cas_id, params, data_dir, args.save_every_molecule, combined_data)
            if cas_id in combined_data:
                logging.info(f'IR added for CAS_ID: {cas_id}')
            else:
                logging.info(f'No IR spectrum found for CAS ID: {cas_id}')

        if counter % 10 == 0 and combined_data:
            logging.info(f'Saving data for batch ending with CAS ID {cas_id}')
            save_all_to_yaml(data_dir, combined_data, "batch_" + str(counter))
            combined_data = {}

    if combined_data:
        logging.info('Saving remaining data...')
        save_all_to_yaml(data_dir, combined_data, "final_batch")
        logging.info('Done Saving. Starting InCHi Scrap')

    if args.scrap_InChi:
        params = {}
        scrap_inchi(cas_dict, params, data_dir)

if __name__ == "__main__":
    main()
