import os

# variable definitions
from config import *

def generate_data_output_dir():

    if not os.path.isdir(MODEL_DATA_DIR):
        try:
            os.mkdir(MODEL_DATA_DIR)
        except:
            sys.exit("New model data folder could not be created")


def generate_unique_params_name(date):

    dir_list = os.listdir(MODEL_DATA_DIR)
    return MODEL1_PARAMS_FILE + "_" + date + ".pt"