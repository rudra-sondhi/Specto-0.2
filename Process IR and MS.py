import yaml
from jcamp import jcamp_read
import numpy as np

# Constants for default values and units
DEFAULT_VALUE = 'N/A'
GAS_STATE = 'gas'
MICROMETERS = 'micrometers'
ABSORBANCE = 'absorbance'
INFRARED_SPECTRUM = 'infrared spectrum'
MASS_SPECTRUM = 'mass spectrum'
UNITS_CM = '1/cm'
MASS = 'mass'

def read_jdx(filename):
    """Reads a JCAMP-DX file and returns its data with the filename included."""
    try:
        with open(filename, 'r', encoding='latin-1') as filehandle:
            data = jcamp_read(filehandle)
        data['filename'] = filename
        return data
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def check_spectra_prop(mol_dict):
    """Checks if the spectra meets the required conditions."""
    cond1 = mol_dict.get('state', DEFAULT_VALUE).lower() == GAS_STATE
    cond2 = mol_dict.get('yunits', DEFAULT_VALUE).lower() == ABSORBANCE
    return all([cond1, cond2])

def check_ir_or_mass(mol_dict):
    """Determines if the data type is either infrared or mass spectrum."""
    data_type = mol_dict.get('data type', DEFAULT_VALUE).lower()
    return data_type if data_type in [INFRARED_SPECTRUM, MASS_SPECTRUM, MASS] else False

def array_and_units(mol_dict, ir_or_mass):
    """Processes arrays based on the spectrum type and units."""
    if ir_or_mass != INFRARED_SPECTRUM:
        return False, None


    x_array = np.array(mol_dict.get('x', []))
    y_array = np.array(mol_dict.get('y', []))
    x_units = mol_dict.get('xunits', DEFAULT_VALUE).lower()
    y_units = mol_dict.get('yunits', DEFAULT_VALUE).lower()


    if x_units == UNITS_CM:
        output_x = np.divide(10e5, x_array)
    elif x_units == MICROMETERS:
        output_x = x_array
    else:
        print(x_units,y_units)
        return False, None

    output_y = y_array if y_units == ABSORBANCE else None
    return output_x, output_y

def process_files(config_path):
    """Reads YAML config and processes files according to their types."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    for cas_id, value in config.items():
        file_path = value['Path']
        file_type = value['Type']

        mol_dict = read_jdx(file_path)



        if mol_dict:
            ir_or_ms = check_ir_or_mass(mol_dict)

            if ir_or_ms == INFRARED_SPECTRUM:
                spec_cond = check_spectra_prop(mol_dict)
                if spec_cond is False:
                    continue
                array_x, array_y = array_and_units(mol_dict, ir_or_ms)
                print(array_x, array_y)
                if array_x is not False:
                    print(f"Processed arrays successfully for {file_type}.")
                else:
                    print(f"Failed to process arrays due to incompatible units. CAS_ID: {cas_id}")
            elif ir_or_ms == MASS_SPECTRUM or ir_or_ms == MASS:
                print(f"Data type for {cas_id} is mass spectrum.")
            else:
                print(f"Data type for {file_type} is neither infrared nor mass spectrum.")
        else:
            print(f"Failed to read data from file at {file_path}.")

if __name__ == "__main__":
    config_file_path = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Super Cool Data/all_data_MS&IR_PATHS_2.yaml'  # Update this with the actual path
    process_files(config_file_path)
