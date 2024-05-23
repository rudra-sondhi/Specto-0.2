import pickle

def load_pickle_file(file_path):
    """
    Load a pickle file from the specified path.
    
    Args:
    file_path (str): The path to the .pickle file.
    
    Returns:
    object: The Python object loaded from the .pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def print_contents(data, indent=0):
    """
    Recursively print the contents of the data from a pickle file.

    Args:
    data (object): The data to be printed.
    indent (int): The indentation level for nested structures.
    """
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}:")
            print_contents(value, indent + 1)
    elif isinstance(data, (list, tuple, set)):
        for index, item in enumerate(data):
            print(f"{prefix}Index {index}:")
            print_contents(item, indent + 1)
    else:
        print(f"{prefix}{data}")

def main():
    file_path = '/Users/rudrasondhi/Downloads/qm9_train_test_val_ir_nmr.pickle'  # Change this to your .pickle file path
    try:
        data = load_pickle_file(file_path)
        print_contents(data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
