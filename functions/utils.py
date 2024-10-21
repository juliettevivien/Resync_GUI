"""
utilisation function
"""

import os
import json
from tkinter.filedialog import askdirectory
import scipy
import operator
import pandas as pd
import numpy as np


parameters = {}
def _update_and_save_multiple_params(
        dictionary: dict, 
        session_ID: str, 
        saving_path: str
        ):
    """
    This function is used to update the parameters dictionary and save it in a json file.

    Inputs:
        - dictionary: dict, contains multiple keys and their values
        - session_ID: str, the session identifier
        - saving_path: str, the path where to save/find the json file
    """
    for key, value in dictionary.items():
        parameters[key] = value
    
    parameter_filename = "parameters_" + str(session_ID) + ".json"
    json_file_path = os.path.join(saving_path, parameter_filename)
    with open(json_file_path, "w") as json_file:
        json.dump(parameters, json_file, indent=4)



def _update_and_save_params(key, value, session_ID: str, saving_path: str):
    """
    This function is used to update the parameters dictionary and save it in a json file.

    Inputs:
        - key: the key of the parameter to update
        - value: the new value of the parameter
        - session_ID: str, the session identifier
        - saving_path: str, the path where to save/find the json file
    """

    parameters[key] = value
    parameter_filename = "parameters_" + str(session_ID) + ".json"
    json_file_path = os.path.join(saving_path, parameter_filename)
    with open(json_file_path, "w") as json_file:
        json.dump(parameters, json_file, indent=4)



def _check_for_empties(
        session_ID: str, 
        fname_lfp: str, 
        fname_external: str, 
        ch_idx_lfp: int, 
        BIP_ch_name: str, 
        index: int
        ):
    """
    This function checks if the input parameters are empty and prints a message
    if they are. It returns a boolean value to indicate if the analysis should be skipped.

    Inputs:
        - session_ID: str, the subject ID
        - fname_lfp: str, the filename of the LFP file
        - fname_external: str, the filename of the external file
        - ch_idx_lfp: int, the index of the LFP channel
        - BIP_ch_name: str, the name of the external channel
        - index: int, the index of the row in the dataframe

    Returns:
        - SKIP: bool, indicates if the analysis should be skipped
    """

    SKIP = False
    if pd.isna(session_ID):
        print(
            f"Skipping analysis for row {index + 2}" f" because session_ID is empty."
        )
        SKIP = True
    if pd.isna(fname_lfp):
        print(
            f"Skipping analysis for row {index + 2}" f" because fname_lfp is empty."
        )
        SKIP = True
    if pd.isna(fname_external):
        print(
            f"Skipping analysis for row {index + 2}"
            f" because fname_external is empty."
        )
        SKIP = True
    if pd.isna(ch_idx_lfp):
        print(
            f"Skipping analysis for row {index + 2}" f" because ch_idx_lfp is empty."
        )
        SKIP = True
    if pd.isna(BIP_ch_name):
        print(
            f"Skipping analysis for row {index + 2}"
            f" because BIP_ch_name is empty."
        )
        SKIP = True
    return SKIP


def _is_channel_in_list(
        channel_array, 
        desired_channel_name
        ):
    """
    This function checks if the desired channel name is in the list of channels.

    Inputs:
        - channel_array: the list of channels
        - desired_channel_name: str, the desired channel name

    Returns:
        - bool, indicates if the channel is in the list
    """

    if desired_channel_name.lower() in (channel.lower() for channel in channel_array):
        return True
    else:
        return False


def _extract_elements(data_list, indices_to_extract):
    # Create an itemgetter object with the indices specified in indices_to_extract
    getter = operator.itemgetter(*indices_to_extract)

    # Use the itemgetter to extract the elements from the data_list
    extracted_elements = getter(data_list)

    return extracted_elements


def _get_input_y_n(message: str) -> str:
    """Get `y` or `n` user input."""

    while True:

        user_input = input(f"{message} (y/n)? ")

        if user_input.lower() in ["y", "n"]:

            break

        print(
            f"Input must be `y` or `n`. Got: {user_input}."
            " Please provide a valid input."
        )

    return user_input


def _get_user_input(message: str):
    """Get user input that can be either an integer or a float."""

    while True:
        user_input = input(f"{message}? ")
        try:
            # Try to convert to float
            user_input_float = float(user_input)
            # Check if the float is actually an integer
            if user_input_float.is_integer():
                return int(user_input_float)
            else:
                return user_input_float
        except ValueError:
            print("Input must be a number. Please provide a valid input.")

def _get_user_input(message: str):
    """Get user input that can be either an integer or a float."""

    while True:
        user_input = input(f"{message}? ")
        try:
            # Try to convert to float
            user_input_float = float(user_input)
            # Check if the float is actually an integer
            if user_input_float.is_integer():
                return int(user_input_float)
            else:
                return user_input_float
        except ValueError:
            print("Input must be a number. Please provide a valid input.")


def _detrend_data(data: np.ndarray):
    """
    This function is used to detrend the data using a high-pass filter.

    Inputs:
        - data: np.ndarray, the data to detrend

    Returns:
        - detrended_data: np.ndarray, the detrended data
    """

    b, a = scipy.signal.butter(1, 0.05, "highpass")
    detrended_data = scipy.signal.filtfilt(b, a, data)

    return detrended_data


def _define_folders():
    """
    This function is used only in the notebook, if the user hasn't already define
    the saving path in the config.json file (back up function).
    """

    # import settings
    json_path = os.path.join(os.getcwd(), "config")
    json_filename = "config.json"
    with open(os.path.join(json_path, json_filename), "r") as f:
        loaded_dict = json.load(f)

    saving_folder = askdirectory(title="Select Saving Folder")
    saving_path = os.path.join(saving_folder, loaded_dict["subject_ID"])
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    return saving_path


def _find_closest_index(array, value, tolerance=1e-6):
    diff = np.abs(array - value)
    if np.any(diff <= tolerance):
        return np.argmin(diff)
    else:
        raise ValueError(f"No element found within {tolerance} of {value}")



def _get_onedrive_path(
    folder: str = 'onedrive', sub: str = None
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'DATA']
    """
    """
    folder_options = [
        'onedrive'
        ]

    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be {folder_options}')
    """

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path) # path is now leading to Users/username

    # get the onedrive folder containing "onedrive" and "charit" and add it to the path
    onedrive_f = [
        f for f in os.listdir(path) if np.logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower())
            ]

    path = os.path.join(path, onedrive_f[0]) # path is now leading to Onedrive folder


    # add the folder DATA to the path and from there open the folders depending on input folder
    path = os.path.join(path, 'DATA')

    return path

    """
    if folder == 'onedrive':

        assert os.path.exists(path), f'wanted path ({path}) not found'
        
        return path

    elif folder == 'sourcedata':

        path = os.path.join(path, 'sourcedata')
        if sub: path = os.path.join(path, f'sub-{sub}')

        assert os.path.exists(path), f'wanted path ({path}) not found'
            
        return path
    """