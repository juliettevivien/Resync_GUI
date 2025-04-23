# import modules
import mne
from mne.io import read_raw_fieldtrip
from mnelab.io.readers import read_raw
from numpy.core.records import fromarrays
from scipy.io import savemat
from pyxdf import resolve_streams
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from os.path import basename, dirname, join

from functions.tmsi_poly5reader import Poly5Reader
from pyxdftools.xdfdata import XdfData

##############################  INPUT FUNCTIONS  ###############################

#### LFP DATASET ####
def load_mat_file(self):
    """Load .mat file."""
    file_name, _ = QFileDialog.getOpenFileName(self, "Select MAT File", "", "MAT Files (*.mat);;All Files (*)")
    
    if file_name:
        try:
            # Load the .mat file using mne's read_raw_fieldtrip
            raw_data = read_raw_fieldtrip(file_name, info={}, data_name="data")
            self.dataset_intra.raw_data = raw_data  # Assign to dataset
            self.dataset_intra.sf = raw_data.info["sfreq"]  # Assign sampling frequency
            self.dataset_intra.ch_names = raw_data.ch_names  # Assign channel names#
            self.dataset_intra.times = np.linspace(0, raw_data.get_data().shape[1]/self.dataset_intra.sf, raw_data.get_data().shape[1])
            self.file_label_intra.setText(f"Selected File: {basename(file_name)}")
            self.dataset_intra.file_name = basename(file_name)
            self.dataset_intra.file_path = dirname(file_name)

            # Show channel selection and plot buttons for intracranial
            self.btn_select_channel_intra.setEnabled(True)
            self.channel_label_intra.setEnabled(True)
            self.btn_choose_channel.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load .mat file: {e}")




#### EXTERNAL DATASET ####

def load_ext_file(self):
    """Load external file. Supported file formats are .xdf, .poly5"""
    file_name, _ = QFileDialog.getOpenFileName(self, "Select External File", "", "All Files (*);;XDF Files (*.xdf);;Poly5 Files (*.Poly5)")
    self.file_label_xdf.setText(f"Selected File: {basename(file_name)}")
    self.dataset_extra.file_name = basename(file_name)
    self.dataset_extra.file_path = dirname(file_name)
    
    if file_name.endswith(".xdf"):
        load_xdf_file(self, file_name)

    elif file_name.endswith(".Poly5"):
        load_poly5_file(self, file_name)


def load_poly5_file(self, file_name):
    """Load .poly5 file."""
    try:
        TMSi_data = Poly5Reader(file_name)
        toMNE = True
        TMSi_rec = TMSi_data.read_data_MNE()
        self.dataset_extra.raw_data = TMSi_rec
        self.dataset_extra.sf = TMSi_rec.info["sfreq"]  # Get the sampling frequency
        self.dataset_extra.ch_names = TMSi_rec.ch_names  # Get the channel names
        self.dataset_extra.times = TMSi_rec.times # Get the timescale

        # Show channel selection and plot buttons for .xdf
        self.channel_label_xdf.setEnabled(True)
        self.btn_select_channel_xdf.setEnabled(True)
        self.btn_select_ecg_channel.setEnabled(True)
        self.ecg_channel_label.setEnabled(True)
        
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load .poly5 file: {e}")


def load_xdf_file(self, file_name):
    """Load .xdf file."""
    try:
        # Load the .xdf file using the read_raw function
        stream_id = find_EEG_stream(file_name, stream_name='SAGA')
        raw_data = read_raw(file_name, stream_ids=[stream_id], preload=True)
        self.dataset_extra.raw_data = raw_data
        self.dataset_extra.sf = raw_data.info["sfreq"]  # Get the sampling frequency
        self.dataset_extra.ch_names = raw_data.ch_names  # Get the channel names
        self.dataset_extra.times = raw_data.times # Get the timescale

        # Show channel selection and plot buttons for .xdf
        self.channel_label_xdf.setEnabled(True)
        self.btn_select_channel_xdf.setEnabled(True)
        self.btn_select_ecg_channel.setEnabled(True)
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load .xdf file: {e}")


def find_EEG_stream(fpath_external, stream_name):
    """Find the EEG stream in the .xdf file."""
    xdf_datas = resolve_streams(fpath_external)
    streams_dict = {stream['name']: stream['stream_id'] for stream in xdf_datas}
    stream_id = streams_dict.get(stream_name)

    if stream_id is None:
        raise ValueError(f"Stream '{stream_name}' not found in the XDF file.")
    
    return stream_id




##############################  OUTPUT FUNCTIONS  ##############################
def select_saving_folder(self):
    # Open a QFileDialog to select a folder
    folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
    self.folder_path = folder_path
    self.label_saving_folder.setText(f"Results will be saved in: {folder_path}")
    
    if folder_path:  # Check if the user selected a folder
        print(f"Selected folder: {folder_path}")


def save_datasets_as_set(self):
    print("events from annotations extraction")
    events, _ = mne.events_from_annotations(self.dataset_extra.raw_data)
    inv_dic = {v: str(k) for k, v in _.items()}

    ## offset intracranial recording (crop everything that is more than 1s before the artifact)
    #tmax_lfp = self.dataset_intra.times[-2]
    #tmax_lfp = max(self.dataset_intra.raw_data.times)
    new_start_intracranial = self.dataset_intra.art_start - 1
    #lfp_rec_offset = self.dataset_intra.raw_data.copy().crop(tmin=new_start_intracranial, tmax=tmax_lfp)
    lfp_rec_offset = self.dataset_intra.synced_data
    
    #print(f"tmax_lfp for cropping is: {tmax_lfp}")

    ## offset external recording (crop everything that is more than 1s before the artifact)
    #tmax_external = max(self.dataset_extra.times)
    new_start_external = self.dataset_extra.art_start - 1
    #TMSi_rec_offset = self.dataset_extra.raw_data.copy().crop(tmin=new_start_external, tmax=tmax_external)
    TMSi_rec_offset = self.dataset_extra.synced_data

    ## transfer of the events from the external to the intracranial recording
    # create a duplicate of the events to manipulate it without changing the external one
    events_lfp = deepcopy(events)

    # get the events from the external in time instead of samples to account for the different sampling frequencies
    events_in_time = events[:,0]/self.dataset_extra.sf

    # then offset the events in time to the new start of the external recording
    events_in_time_offset = events_in_time - new_start_external

    if self.dataset_intra.eff_sf is not None:
        lfp_sf = self.dataset_intra.eff_sf
    else:
        lfp_sf = self.dataset_intra.sf

    # convert the events in time offset to samples corresponding to the sampling frequency of the intracranial recording
    # because the annotations object works with samples, not timings
    events_in_time_offset_lfp = events_in_time_offset * lfp_sf
    events_lfp[:,0] = events_in_time_offset_lfp

    # Recast event descriptions to standard Python strings
    #events_lfp[:, 2] = events_lfp[:, 2].astype(str)

    ## create an annotation object for the intracranial recording
    annotations_lfp = mne.annotations_from_events(events_lfp, sfreq=lfp_sf, event_desc=inv_dic)

    lfp_rec_offset.set_annotations(None) # make sure that no annotations are present
    lfp_rec_offset.set_annotations(annotations_lfp) # set the new annotations

    external_title = ("SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-4]) + ".set")

    if len(self.dataset_intra.synced_data.ch_names) > 6:
        lfp_title = ("SYNCHRONIZED_INTRACRANIAL_CLEANED_" + str(self.dataset_intra.file_name[:-4]) + ".set")
    else:
        lfp_title = ("SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".set")


    if self.folder_path is not None:
        fname_external_out=join(self.folder_path, external_title)
        fname_lfp_out =join(self.folder_path, lfp_title)
    else:
        fname_external_out = external_title
        fname_lfp_out = lfp_title

    TMSi_rec_offset_annotations_onset= (TMSi_rec_offset.annotations.onset) - new_start_external
    lfp_rec_offset_annotations_onset= (lfp_rec_offset.annotations.onset) - new_start_intracranial
    
    lfp_timescale = np.linspace(0, self.dataset_intra.synced_data.get_data().shape[1]/lfp_sf, self.dataset_intra.synced_data.get_data().shape[1])
    
    write_set(
        fname_external_out, 
        TMSi_rec_offset, 
        TMSi_rec_offset_annotations_onset,
        TMSi_rec_offset.info['sfreq'],
        TMSi_rec_offset.times
        )
    write_set(
        fname_lfp_out, 
        lfp_rec_offset, 
        lfp_rec_offset_annotations_onset,
        lfp_sf,
        lfp_timescale
        )

    QMessageBox.information(self, "Synchronization", "Synchronization done. Both files saved as .SET")



def synchronize_datasets_as_pickles(self): 
    ## Intracranial ##
    # Crop beginning of LFP intracranial recording 1 second before first artifact:
    time_start_LFP_0 = self.dataset_intra.art_start - 1  # 1s before first artifact
    index_start_LFP = time_start_LFP_0 * (self.dataset_intra.sf)

    LFP_array = self.dataset_intra.raw_data.get_data()

    LFP_cropped = LFP_array[:, int(index_start_LFP) :].T
    LFP_df_offset = pd.DataFrame(LFP_cropped)
    LFP_df_offset.columns = self.dataset_intra.ch_names
    LFP_timescale_offset_s = self.dataset_intra.times[int(index_start_LFP):] - time_start_LFP_0

    # Save as pickle file:
    LFP_df_offset["sf_LFP"] = self.dataset_intra.sf
    LFP_df_offset["time_stamp"] = LFP_timescale_offset_s
    lfp_title = ("SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".pkl")
    if self.folder_path is not None:
        LFP_filename = join(self.folder_path, lfp_title)
    else: LFP_filename = lfp_title
    # Save the dataset to a pickle file
    with open(LFP_filename, "wb") as file:
        pickle.dump(LFP_df_offset, file)


    ## External##
    stream_names = []
    stream_ids = []
    streams_dict = {}

    filepath = join(self.dataset_extra.file_path, self.dataset_extra.file_name)
    xdf_datas = XdfData(filepath).resolve_streams()

    for streams in range(1, len(xdf_datas['name'])+1, 1):
        stream_names.append(xdf_datas['name'][streams])

    for name in stream_names:
        stream_ids.append(xdf_datas[xdf_datas['name'] == name].index[0])

    for stream_nb in zip(stream_names, stream_ids):
        # create a dictionnary associating each streaming name from stream_names list to its corresponding stream_id:
        streams_dict[stream_nb[0]] = stream_nb[1]

    # LOAD ALL STREAMS IN A DICTIONNARY

    streams = {}

    for idx, (name, stream_id) in enumerate(streams_dict.items(), start=1):
        stream_name = f"{name}_stream"
        streams[stream_name] = XdfData(filepath).load(stream_id=[stream_id])

    # CREATE GLOBAL VARIABLES FOR EACH STREAM
    for stream_name, stream_data in streams.items():
        globals()[stream_name] = stream_data.data()
        print(stream_name)

    # Convert self.dataset_extra.art_start into the xdf timescale from the BIP data
    art_start_0 = self.dataset_extra.art_start - 1
    # Get the original timestamps from both sources
    timestamps_global = globals()['SAGA_stream']['time_stamp']
    times_real = self.dataset_extra.times

    # Find the index in self.dataset_extra.times corresponding to art_start_0
    art_start_index = (times_real >= art_start_0).argmax()

    # Filter the timestamps from the global clock based on this index
    filtered_timestamps_global = timestamps_global[art_start_index:]
    art_start_in_globals = np.array(filtered_timestamps_global.iloc[0])
    print(f"Art start in global timestamps: {art_start_in_globals}")


    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Find the index corresponding to the value of art_start_in_globals
        index = np.argmax(globals()[stream_name]['time_stamp'] >= art_start_in_globals)
        print(f"Index corresponding to art_start_in_globals in {stream_name}: {index}")
        
        # Crop the stream data from the index onwards
        stream_offset = globals()[stream_name].iloc[index:]
        
        # Create a copy of the cropped stream data
        stream_offset_copy = stream_offset.copy()
        
        # Offset the 'time_stamp' column
        stream_offset_copy['time_stamp'] = stream_offset_copy['time_stamp'] - art_start_in_globals
        
        # Reset the index
        stream_offset_copy.reset_index(drop=True, inplace=True)
        
        # Update the global variable with the modified data
        globals()[stream_name] = stream_offset_copy

    # Create a dictionary to hold the extracted DataFrames
    extracted_streams = {}

    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Extract the current stream DataFrame
        extracted_streams[f"df_{stream_name}"] = globals()[stream_name].copy()

    ## saving as pickle:
    # Iterate over the extracted DataFrames
    for df_name, df_data in extracted_streams.items():
        # Generate the filename
        external_title = (f"{df_name}SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-4]) + ".pkl")
        # Create the full path to the file
        if self.folder_path is not None:
            filepath = join(self.folder_path, external_title)
        else:
            filepath = external_title
        # Save the DataFrame to a pickle file
        df_data.to_pickle(filepath)
    QMessageBox.information(self, "Synchronization", "Synchronization done. All files saved separately as .pickle")



def synchronize_datasets_as_one_pickle(self):
    ## Intracranial ##
    # Crop beginning of LFP intracranial recording 1 second before first artifact:
    time_start_LFP_0 = self.dataset_intra.art_start - 1  # 1s before first artifact
    index_start_LFP = time_start_LFP_0 * (self.dataset_intra.sf)

    LFP_array = self.dataset_intra.raw_data.get_data()

    LFP_cropped = LFP_array[:, int(index_start_LFP) :].T
    LFP_df_offset = pd.DataFrame(LFP_cropped)
    LFP_df_offset.columns = self.dataset_intra.ch_names
    LFP_timescale_offset_s = self.dataset_intra.times[int(index_start_LFP):] - time_start_LFP_0

    # Prepare LFP dataframe
    LFP_df_offset["sf_LFP"] = self.dataset_intra.sf
    LFP_df_offset["time_stamp"] = LFP_timescale_offset_s


    ## External ##
    stream_names = []
    stream_ids = []
    streams_dict = {}

    filepath = join(self.dataset_extra.file_path, self.dataset_extra.file_name)
    xdf_datas = XdfData(filepath).resolve_streams()

    for streams in range(1, len(xdf_datas['name'])+1, 1):
        stream_names.append(xdf_datas['name'][streams])

    for name in stream_names:
        stream_ids.append(xdf_datas[xdf_datas['name'] == name].index[0])

    for stream_nb in zip(stream_names, stream_ids):
        # create a dictionnary associating each streaming name from stream_names list to its corresponding stream_id:
        streams_dict[stream_nb[0]] = stream_nb[1]

    # LOAD ALL STREAMS IN A DICTIONNARY
    streams = {}

    for idx, (name, stream_id) in enumerate(streams_dict.items(), start=1):
        stream_name = f"{name}_stream"
        streams[stream_name] = XdfData(filepath).load(stream_id=[stream_id])

    # CREATE GLOBAL VARIABLES FOR EACH STREAM
    for stream_name, stream_data in streams.items():
        globals()[stream_name] = stream_data.data()
        print(stream_name)

    # Convert self.dataset_extra.art_start into the xdf timescale from the BIP data
    art_start_0 = self.dataset_extra.art_start - 1
    # Get the original timestamps from both sources
    timestamps_global = globals()['SAGA_stream']['time_stamp']
    times_real = self.dataset_extra.times

    # Find the index in self.dataset_extra.times corresponding to art_start_0
    art_start_index = (times_real >= art_start_0).argmax()

    # Filter the timestamps from the global clock based on this index
    filtered_timestamps_global = timestamps_global[art_start_index:]
    art_start_in_globals = np.array(filtered_timestamps_global.iloc[0])
    print(f"Art start in global timestamps: {art_start_in_globals}")


    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Find the index corresponding to the value of art_start_in_globals
        index = np.argmax(globals()[stream_name]['time_stamp'] >= art_start_in_globals)
        print(f"Index corresponding to art_start_in_globals in {stream_name}: {index}")

        # Crop the stream data from the index onwards
        stream_offset = globals()[stream_name].iloc[index:]

        # Create a copy of the cropped stream data
        stream_offset_copy = stream_offset.copy()

        # Offset the 'time_stamp' column
        stream_offset_copy['time_stamp'] = stream_offset_copy['time_stamp'] - art_start_in_globals
        
        # Reset the index
        stream_offset_copy.reset_index(drop=True, inplace=True)
        
        # Update the global variable with the modified data
        globals()[stream_name] = stream_offset_copy

    # Create a dictionary to hold the extracted DataFrames
    extracted_streams = {}

    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Extract the current stream DataFrame
        extracted_streams[f"df_{stream_name}"] = globals()[stream_name].copy()


    # Iterate over the extracted DataFrames
    for df_name, df_data in extracted_streams.items():
        # Create separate DataFrame variables with specific names
        globals()[df_name] = pd.DataFrame(df_data)


    # Create an empty list to hold the DataFrames
    all_dfs = []

    # Iterate over the extracted DataFrames
    for df_name, df_data in extracted_streams.items():
        # Create a DataFrame with a MultiIndex containing the df_name as the top level
        df = pd.DataFrame(df_data)
        df.columns = pd.MultiIndex.from_product([[df_name], df.columns])  # Add df_name as the top level of column index
        all_dfs.append(df)

    # Concatenate all DataFrames in the list along axis 1 (columns)
    LSL_df = pd.concat(all_dfs, axis=1)

    # Assuming LFP_df_offset is your new DataFrame
    # First, adjust its column names to include a MultiIndex with the header 'LFP'
    LFP_df_offset.columns = pd.MultiIndex.from_product([['df_LFP'], LFP_df_offset.columns])

    # Concatenate LFP_df_offset on top of big_df along axis 1 (columns)
    final_df = pd.concat([LFP_df_offset, LSL_df], axis=1)
    
    ## saving as pickle:
    # Generate the filename
    filename = f"{self.dataset_extra.file_name[:-4]}_synchronized_data.pkl"
    # Create the full path to the file
    if self.folder_path is not None:
        filepath = join(self.folder_path, filename)
    else:
        filepath = filename
    # Save the DataFrame to a pickle file
    final_df.to_pickle(filepath)
    print(f"DataFrame {filename} saved as pickle to {filepath}")
    QMessageBox.information(self, "Synchronization", "Synchronization done. Everything saved as one .pickle file")



def synchronize_datasets_as_mat(self):
    index_start_LFP = (self.dataset_intra.art_start - 1) * self.dataset_intra.sf
    LFP_array = self.dataset_intra.raw_data.get_data()
    LFP_cropped = LFP_array[:, int(index_start_LFP) :].T

    ## External ##
    # Crop beginning of external recordings 1s before first artifact:
    time_start_external = (self.dataset_extra.art_start) - 1
    index_start_external = time_start_external * self.dataset_extra.sf
    external_file = self.dataset_extra.raw_data.get_data()
    external_cropped = external_file[:, int(index_start_external) :].T

    # Check which recording is the longest,
    # crop it to give it the same duration as the other one:
    LFP_rec_duration = len(LFP_cropped) / self.dataset_intra.sf
    external_rec_duration = len(external_cropped) / self.dataset_extra.sf

    if LFP_rec_duration > external_rec_duration:
        index_stop_LFP = external_rec_duration * self.dataset_intra.sf
        LFP_synchronized = LFP_cropped[: int(index_stop_LFP), :]
        external_synchronized = external_cropped
    elif external_rec_duration > LFP_rec_duration:
        index_stop_external = LFP_rec_duration * self.dataset_extra.sf
        external_synchronized = external_cropped[: int(index_stop_external), :]
        LFP_synchronized = LFP_cropped
    else:
        LFP_synchronized = LFP_cropped
        external_synchronized = external_cropped  

    # save the synchronized data in mat format          
    LFP_df_offset = pd.DataFrame(LFP_synchronized)
    LFP_df_offset.columns = self.dataset_intra.ch_names
    external_df_offset = pd.DataFrame(external_synchronized)
    external_df_offset.columns = self.dataset_extra.ch_names

    lfp_title = ("SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".mat")
    external_title = (f"SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-6]) + ".mat")
    
    if self.folder_path is not None:
        LFP_filename = join(self.folder_path, lfp_title)
        external_filename = join(self.folder_path, external_title)
    else:
        LFP_filename = lfp_title
        external_filename = external_title

    savemat(
        LFP_filename,
        {
            "data": LFP_df_offset.T,
            "fsample": self.dataset_intra.sf,
            "label": np.array(
                LFP_df_offset.columns.tolist(), dtype=object
            ).reshape(-1, 1),
        },
    )
    savemat(
        external_filename,
        {
            "data": external_df_offset.T,
            "fsample": self.dataset_extra.sf,
            "label": np.array(
                external_df_offset.columns.tolist(), dtype=object
            ).reshape(-1, 1),
        },
    )
    QMessageBox.information(self, "Synchronization", "Synchronization done. Both files saved as .mat files")       



def write_set(fname, raw, annotations_onset, fs, times):
    """Export raw to EEGLAB .set file."""
    data = raw.get_data() * 1e6  # convert to microvolts
    #data = raw.get_data()
    ch_names = raw.info["ch_names"]
    chanlocs = fromarrays([ch_names], names=["labels"])
    events = fromarrays([raw.annotations.description,
                         annotations_onset * fs + 1,
                         raw.annotations.duration * fs],
                        names=["type", "latency", "duration"])
    savemat(fname, dict(EEG=dict(data=data,
                                 setname=fname,
                                 nbchan=data.shape[0],
                                 pnts=data.shape[1],
                                 trials=1,
                                 srate=fs,
                                 xmin=times[0],
                                 xmax=times[-1],
                                 chanlocs=chanlocs,
                                 event=events,
                                 icawinv=[],
                                 icasphere=[],
                                 icaweights=[])),
            appendmat=False)

