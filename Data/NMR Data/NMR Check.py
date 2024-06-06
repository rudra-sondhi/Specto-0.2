import os
import yaml

def check_screenshots(yaml_file, log_file):
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    lenth = len(data) 
    print(f"Lenght of data: {lenth}")
    
    missing_files = []
    
    # Iterate over the data to find screenshot paths
    for key, value in data.items():
        if 'Screenshots' in value:
            for nmr_type, path in value['Screenshots'].items():
                if not os.path.exists(path):
                    missing_files.append(path)
    
    # Write the missing files to the log file
    if missing_files:
        with open(log_file, 'a') as file:
            for path in missing_files:
                file.write(f"{path}\n")
    
    if not missing_files:
        print("All screenshot files exist.")
    else:
        print(f"Missing files logged in {log_file}.")

# Example usage
check_screenshots('/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/NMR Data/nmr_results.yaml', '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Screenshots/missing_screenshots.txt')
