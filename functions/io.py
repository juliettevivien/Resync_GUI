# import modules
from mne.io import read_raw_fieldtrip
from os.path import join
from numpy.core.records import fromarrays
from scipy.io import savemat
from pyxdf import resolve_streams

# import custom-made functions
from functions.utils import _update_and_save_multiple_params



##############################  INPUT FUNCTIONS  ###############################

#### LFP DATASET ####
def load_mat_file(
        session_ID: str, 
        filename: str, 
        saving_path: str, 
        source_path: str
        ):
    """
    Reads .mat-file in FieldTrip structure using mne-function

    Input:
        - session_ID: str
        - filename: str, .mat-filename
        - saving_path: str, path to save the parameters
        - source_path: str, path to the source file

    Returns:
        - data: mne-object of .mat file
    """

    # Error if filename doesn´t end with .mat
    assert filename[-4:] == ".mat", f"filename no .mat INCORRECT extension: {filename}"

    dictionary = {"SESSION_ID": session_ID, "FNAME_LFP":filename}
    _update_and_save_multiple_params(dictionary,session_ID,saving_path)


    data = read_raw_fieldtrip(
        filename,
        info={},
        data_name="data",
    )

    return data



def load_intracranial_artifact_channel(lfp_rec, ch_idx_lfp):
    if type(ch_idx_lfp) == float:
        ch_idx_lfp = int(ch_idx_lfp)
    lfp_sig = lfp_rec.get_data()[ch_idx_lfp]

    return lfp_sig



#### EXTERNAL DATA RECORDER DATASET ####
def find_EEG_stream(fpath_external, stream_name):
    # Determine which stream contains the EEG data:
    xdf_datas = resolve_streams(fpath_external)
    streams_dict = {}

    for streams in range(0, len(xdf_datas), 1):
        streams_dict[xdf_datas[streams]['name']] = xdf_datas[streams]['stream_id']
    
    # in streams_dict, find the stream_id corresponding to the EEG stream:
    stream_id = streams_dict[stream_name]

    return stream_id



def load_xdf_artifact_channel(
    TMSi_rec,
    BIP_ch_name: str,
    session_ID,
    saving_path
    ):

    assert BIP_ch_name in TMSi_rec.ch_names, "{} is not in externally recorded channels. Please choose from the available channels: {}".format(BIP_ch_name, TMSi_rec.ch_names)
    
    ch_index = TMSi_rec.ch_names.index(BIP_ch_name)
    BIP_channel = TMSi_rec.get_data()[ch_index]
    dictionary = {"EXTERNAL_CH_NAME": BIP_ch_name, "EXTERNAL_CH_IDX":ch_index}
    _update_and_save_multiple_params(dictionary, session_ID, saving_path)


    return BIP_channel, ch_index




##############################  OUTPUT FUNCTIONS  ##############################

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


"""
def write_set(fname, raw, annotations_onset):  # didn't work when we needed to update the effective sampling frequency of the intracranial data
    '''Export raw to EEGLAB .set file.'''
    data = raw.get_data() * 1e6  # convert to microvolts
    fs = raw.info["sfreq"]
    times = raw.times
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
"""