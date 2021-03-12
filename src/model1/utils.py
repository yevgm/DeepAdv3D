import os

# variable definitions
from config import *

from vista.geom_vis import plot_mesh, plot_mesh_montage


def generate_data_output_dir():

    if not os.path.isdir(MODEL_DATA_DIR):
        try:
            os.mkdir(MODEL_DATA_DIR)
        except:
            sys.exit("New model data folder could not be created")


# def generate_unique_params_name(date):
#
#     dir_list = os.listdir(MODEL_DATA_DIR)
#     return MODEL1_PARAMS_DIR + "_" + date + ".pt"

def get_param_file(dir_name):
    dir_list = os.listdir(dir_name)
    for file in dir_list:
        if file.endswith(".pt"):
            return os.path.join(dir_name, file)


def dump_adversarial_example_image(orig_vertices, adex, faces, step_num, file_path):

    p, _ = plot_mesh_montage([orig_vertices[0].T, adex[0].T], [faces[0], faces[0]], screenshot=True)
    path = os.path.join(file_path, "step_" + str(step_num) + ".png")
    p.show(screenshot=path, full_screen=True)