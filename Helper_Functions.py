from jcamp import jcamp_read

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
    
