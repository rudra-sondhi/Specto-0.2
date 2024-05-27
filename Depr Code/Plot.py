import yaml
from jcamp import jcamp_read
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Constants for default values and units
DEFAULT_VALUE = 'N/A'
GAS_STATE = 'gas'
MICROMETERS = 'micrometers'
ABSORBANCE = 'absorbance'
INFRARED_SPECTRUM = 'infrared spectrum'
MASS_SPECTRUM = 'mass spectrum'
UNITS_CM = '1/cm'
MASS = 'mass'

min_ir = 399
max_ir = 4001
step_ir = 3.25

min_mass = 1 
max_mass = 650
step_mass = 1

eps = 1e-4

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
        print(x_units, y_units)
        return False, None

    output_y = y_array if y_units == ABSORBANCE else None

    if output_y is not None:
        percent_transmittance = 10**(2 - output_y)
        output_y = np.divide(percent_transmittance, 100)

    return output_x, output_y

def plot_and_save_spectrum(array_x, array_y, mol_dict, cas_id, save_dir):
    """Plots the spectrum and saves it as a PNG file."""
    title = mol_dict.get('title', 'IR Spectrum')

    # Create the plot with a specific aspect ratio
    plt.figure(figsize=(18, 5))  # Aspect ratio similar to the uploaded graph

    # Plotting the transformed data
    plt.plot(array_x, array_y, marker='', linestyle='-', linewidth=1.5)

    # Adjusting x-axis and y-axis
    plt.gca().invert_xaxis()

    # Adding a title and labels with the font size
    plt.title(title, fontsize=20)
    plt.xlabel('Wavenumbers (cm-1)', fontsize=16)
    plt.ylabel('Transmittance', fontsize=16)

    # Adding grid for better readability
    plt.grid(True)

    # Set Y-axis to display labels without scientific notation or offset
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    plt.gca().yaxis.set_major_formatter(y_formatter)

    plt.xlim([max(array_x), min(array_x)])  # Adjust as per the data range

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, f'{cas_id}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot for {cas_id} at {save_path}")

def plot_and_save_spectrum_NIST_STYLE(array_x, array_y, mol_dict, cas_id, save_dir):
    """Plots the spectrum and saves it as a PNG file."""

    # Create the plot with a specific aspect ratio
    plt.figure(figsize=(18, 5))  # Aspect ratio similar to the uploaded graph

    # Plotting the transformed data
    plt.plot(array_x, array_y, marker='', linestyle='-', linewidth=1.5, color = '#d55e00')

    # Adjusting x-axis and y-axis
    plt.gca().invert_xaxis()

    # Adding grid for better readability
    plt.grid(True)

    # Set Y-axis to display labels without scientific notation or offset
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    plt.gca().yaxis.set_major_formatter(y_formatter)

    plt.xlim([max(array_x), min(array_x)])  # Adjust as per the data range

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, f'{cas_id}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot for {cas_id} at {save_path}")

def process_files(config_path, save_dir, save_dir_nist):
    """Reads YAML config and processes files according to their types."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    for cas_id, value in config.items():
        try:
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

                    if array_x is not False:
                        plot_and_save_spectrum(array_x, array_y, mol_dict, cas_id, save_dir)
                        plot_and_save_spectrum_NIST_STYLE(array_x, array_y, mol_dict, cas_id, save_dir_nist)
                    else:
                        print(f"Failed to process arrays due to incompatible units. CAS_ID: {cas_id}")
                elif ir_or_ms == MASS_SPECTRUM or ir_or_ms == MASS:
                    print(f"Data type for {cas_id} is mass spectrum.")
                else:
                    print(f"Data type for {file_type} is neither infrared nor mass spectrum.")
            else:
                print(f"Failed to read data from file at {file_path}.")
        except Exception as e:
            print(f"An error occurred while processing {cas_id}: {e}")
            continue

if __name__ == "__main__":
    config_file_path = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Super Cool Data/all_data_MS&IR_PATHS_2.yaml'  # Update this with the actual path
    save_directory = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/TITLE/'  # Update this with the desired save directory
    save_directory_nist = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/NIST/'
    process_files(config_file_path, save_directory, save_directory_nist)
