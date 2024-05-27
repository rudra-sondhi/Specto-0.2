import yaml
from jcamp import jcamp_read
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from Helper_Functions import read_jdx
from JDX_Mass_Spec import extract_npoints_and_title, extract_xy_pairs, plot_xy_pairs, plot_xy_pairs_no_axes

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
        output_x = x_array
    elif x_units == MICROMETERS:
        return False, None
    else:
        print(f"Unsupported units: x_units={x_units}, y_units={y_units}")
        return False, None

    output_y = y_array if y_units == ABSORBANCE else None

    if output_y is not None:
        percent_transmittance = 10**(2 - output_y)
        output_y = np.divide(percent_transmittance, 100) 

    return output_x, output_y

def update_progress_file(progress_file, cas_id):
    """Updates the progress file with the latest processed CAS_ID."""
    with open(progress_file, 'w') as file:
        file.write(cas_id)
    print(f"Progress updated for CAS_ID: {cas_id}")

def read_progress_file(progress_file):
    """Reads the last processed CAS_ID from the progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            return file.readline().strip()
    return None

def update_yaml_file(yaml_file, cas_id, plot_path, nist_plot_path, csv_path):
    """Updates the YAML file with the plot paths for the given CAS_ID."""
    data = {}
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file) or {}

    data[cas_id] = {
        'standard_plot': plot_path,
        'nist_plot': nist_plot_path,
        'csv_path': csv_path
    }

    with open(yaml_file, 'w') as file:
        yaml.safe_dump(data, file)
    print(f"YAML file updated for CAS_ID: {cas_id}")

def update_yaml_file_mass(yaml_file, cas_id, axes_plot_path, no_axes_plot_path, csv_path):
    """Updates the YAML file with the plot paths for the given CAS_ID."""
    data = {}
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file) or {}

    data[cas_id] = {
        'axes_plot': axes_plot_path,
        'no_axes_plot': no_axes_plot_path,
        'csv_path': csv_path
    }

    with open(yaml_file, 'w') as file:
        yaml.safe_dump(data, file)
    print(f"YAML file updated for CAS_ID: {cas_id}")

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

    return save_path

def plot_and_save_spectrum_NIST_STYLE(array_x, array_y, mol_dict, cas_id, save_dir):
    """Plots the spectrum and saves it as a PNG file."""

    # Create the plot with a specific aspect ratio
    plt.figure(figsize=(18, 5))  # Aspect ratio similar to the uploaded graph

    # Plotting the transformed data
    plt.plot(array_x, array_y, marker='', linestyle='-', linewidth=1.5, color='#d55e00')

    # Adjusting x-axis and y-axis
    plt.gca().invert_xaxis()

    # Adding grid for better readability
    plt.grid(True)

    # Set Y-axis to display labels without scientific notation or offset
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    plt.gca().yaxis.set_major_formatter(y_formatter)

    plt.xlim([max(array_x), min(array_x)])  # Adjust as per the data range

    # Turn off the axis
    plt.axis('off')

    # Adjust layout to remove white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, f'{cas_id}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved NIST-style plot for {cas_id} at {save_path}")

    return save_path

def create_ir_bins(array_x, array_y, save_dir, cas_id):
    """Creates bins for IR data and saves them as a CSV file."""
    bins = np.arange(min_ir, max_ir, step_ir)
    bin_indices = np.digitize(array_x, bins)
    
    binned_data = {
        'bin_start': bins[:-1],
        'bin_end': bins[1:],
        'mean_transmittance': [array_y[bin_indices == i].mean() if len(array_y[bin_indices == i]) > 0 else np.nan for i in range(1, len(bins))]
    }
    
    binned_df = pd.DataFrame(binned_data)
    binned_df['bin_range'] = binned_df.apply(lambda row: f"{row['bin_start']}-{row['bin_end']}", axis=1)
    binned_df = binned_df.set_index('bin_range')
    binned_df = binned_df.drop(columns=['bin_start', 'bin_end'])
    binned_df = binned_df.dropna(subset=['mean_transmittance'])
    
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, f'{cas_id}_binned.csv')
    binned_df.to_csv(csv_path)
    
    print(f"Saved binned data for {cas_id} at {csv_path}")
    
    return csv_path

def save_xy_pairs_to_csv_mass(xy_pairs, save_dir, cas_id):
    # Convert the xy_pairs to a DataFrame
    df = pd.DataFrame(xy_pairs, columns=['x', 'y'])
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the CSV file path
    csv_path = os.path.join(save_dir, f'{cas_id}.csv')
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)
    
    print(f"Saved CSV file for {cas_id} at {csv_path}")
    return csv_path

def process_files(config_path, save_dir, save_dir_nist, progress_file, yaml_file, csv_save_dir, mass_yaml_file_path, mass_csv_save_directory, save_directory_mass_axes, save_directory_mass_no_axes):
    """Reads YAML config and processes files according to their types."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    last_processed = read_progress_file(progress_file)
    process_next = not bool(last_processed)

    for cas_id, value in config.items():
        if not process_next:
            if cas_id == last_processed:
                process_next = True
            continue

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
                        plot_path = plot_and_save_spectrum(array_x, array_y, mol_dict, cas_id, save_dir)
                        nist_plot_path = plot_and_save_spectrum_NIST_STYLE(array_x, array_y, mol_dict, cas_id, save_dir_nist)
                        csv_path = create_ir_bins(array_x, array_y, csv_save_dir, cas_id)
                        update_yaml_file(yaml_file, cas_id, plot_path, nist_plot_path, csv_path)
                        update_progress_file(progress_file, cas_id)
                    else:
                        print(f"Failed to process arrays due to incompatible units. CAS_ID: {cas_id}")
                elif ir_or_ms == MASS_SPECTRUM or ir_or_ms == MASS:
                    print(f"Data type for {cas_id} is mass spectrum.")
                    npoints, title = extract_npoints_and_title(mol_dict)
                    xy_pairs = extract_xy_pairs(mol_dict)
                    if xy_pairs is None:
                        continue
                    else:
                        if len(xy_pairs) != npoints:
                            continue
                        else: 
                            axes_mass_path = plot_xy_pairs(xy_pairs, title, cas_id, save_directory_mass_axes)
                            no_axes_mass_path = plot_xy_pairs_no_axes(xy_pairs, cas_id, save_directory_mass_no_axes)
                            
                            mass_csv_path = save_xy_pairs_to_csv_mass(xy_pairs, mass_csv_save_directory, cas_id)
                            update_yaml_file_mass(mass_yaml_file_path, cas_id, axes_mass_path, no_axes_mass_path, mass_csv_path)
                else:
                    print(f"Data type for {file_type} is neither infrared nor mass spectrum.")
            else:
                print(f"Failed to read data from file at {file_path}.")
        except Exception as e:
            print(f"An error occurred while processing {cas_id}: {e}")
            continue

if __name__ == "__main__":
    config_file_path = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Super Cool Data/all_data_MS&IR_PATHS_2.yaml'  # Update this with the actual path
    save_directory = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/IR Plots/Axes'  # Update this with the desired save directory
    save_directory_nist = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/IR Plots/No Axes/'
    progress_file_path = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/progress.txt'  # Path for progress tracking
    yaml_file_path = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/plot_paths.yaml'  # Path for storing plot paths
    mass_yaml_file_path = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/mass_plot_paths.yaml'  # Path for storing plot paths

    csv_save_directory = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/Bin CSV/'  # Directory for saving CSV files
    mass_csv_save_directory = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/Mass CSV/'  # Directory for saving CSV files


    save_directory_mass_axes = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/Mass/Axes'
    save_directory_mass_no_axes = '/Users/rudrasondhi/Desktop/Specto/Specto/Plots/Mass/No Axes'

    process_files(config_file_path, save_directory, save_directory_nist, progress_file_path, yaml_file_path, csv_save_directory, mass_yaml_file_path, mass_csv_save_directory, save_directory_mass_axes, save_directory_mass_no_axes)
