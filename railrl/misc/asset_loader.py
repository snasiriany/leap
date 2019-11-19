import joblib
import numpy as np
import pickle

from railrl.config.launcher_config import LOCAL_LOG_DIR
import os

PICKLE = 'pickle'
NUMPY = 'numpy'
JOBLIB = 'joblib'


def local_path_from_s3_or_local_path(filename):
    relative_filename = os.path.join(LOCAL_LOG_DIR, filename)
    if os.path.isfile(filename):
        return filename
    elif os.path.isfile(relative_filename):
        return relative_filename
    else:
        return None

def split_s3_full_path(s3_path):
    """
    Split "s3://foo/bar/baz" into "foo" and "bar/baz"
    """
    bucket_name_and_directories = s3_path.split('//')[1]
    bucket_name, *directories = bucket_name_and_directories.split('/')
    directory_path = '/'.join(directories)
    return bucket_name, directory_path


def load_local_or_remote_file(filepath, file_type=None):
    local_path = local_path_from_s3_or_local_path(filepath)
    if local_path is None:
        return None
    if file_type is None:
        extension = local_path.split('.')[-1]
        if extension == 'npy':
            file_type = NUMPY
        else:
            file_type = PICKLE
    else:
        file_type = PICKLE
    if file_type == NUMPY:
        object = np.load(open(local_path, "rb"))
    elif file_type == JOBLIB:
        object = joblib.load(local_path)
    else:
        object = pickle.load(open(local_path, "rb"))
    print("loaded", local_path)
    return object
