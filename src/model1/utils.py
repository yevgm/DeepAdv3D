import os

# variable definitions
from config import *

def generate_data_output_dir():

    if not os.path.isdir(MODEL_DATA_DIR):
        try:
            os.mkdir(MODEL_DATA_DIR)
        except:
            sys.exit("New model data folder could not be created")
