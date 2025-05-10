import os
import numpy as np

def save_structure_constants(SC, save_file='structure_constants.npy'):
    """
    Saves the structure constants (SC) to a specified file location.
    """
    # Ensure the directory exists
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the structure constants
    np.save(save_file, SC)
    print(f"Structure constants saved to {save_file}")