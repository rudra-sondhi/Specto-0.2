import pandas as pd
import re



elements = {'O', 'C', 'H', 'N', 'Br', 'Cl', 'S', 'Si'}
def valid_formula(formula):
    """Check if the formula contains only the desired elements."""
    # Regex pattern for valid formulas (combinations of the elements and their counts)
    pattern = r'^[' + ''.join(elements) + r'0-9]+$'
    return bool(re.match(pattern, formula))

def filter_molecules(file_path):
    """Read, filter, and save the molecules data."""
    # Read data
    data = pd.read_csv(file_path, sep="\t", header=None, names=["Name", "Formula", "CAS Number"])

    # Filter out rows where 'Formula' or 'CAS Number' is NaN or 'CAS Number' is 'N/A'
    data = data.dropna(subset=['Formula', 'CAS Number'])
    data = data[data['CAS Number'] != "N/A"]

    # Filter molecules based on elements
    filtered_data = data[data['Formula'].apply(valid_formula)]

    # Save to new CSV
    filtered_data.to_csv('/Users/rudrasondhi/Desktop/Specto/Data/IR_Spectra/filtered_molecules.csv', index=False)

    return filtered_data

filter_molecules('/Users/rudrasondhi/Downloads/species.txt')