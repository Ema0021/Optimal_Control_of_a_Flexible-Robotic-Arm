import pickle
from termcolor import colored

def save_simulation_data(filename, data):
    """
    Save the simulation data to a file using pickle.
    
    Parameters:
        - filename (str): The name of the file where the data will be saved.
        - data (dict): The simulation data to be saved.

    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(colored(f"\n Data saved successfully in {filename}", "green", attrs=["bold"]))
    except Exception as e:
        print(colored(f"\n Error saving data: {e}", "red", attrs=["bold"]))

def load_simulation_data(filename):
    """
    Load the simulation data from a file using pickle.
    
    Parameters:
        - filename (str): The name of the file to load the data from.
    
    Returns:
        - dict: The loaded simulation data.
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(colored(f"File {filename} not found. Starting fresh simulation.", "yellow", attrs=["bold"]))
        raise
    except Exception as e:
        print(colored(f"Error loading data: {e}", "red", attrs=["bold"]))
        raise