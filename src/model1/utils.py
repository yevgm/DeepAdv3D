import os

# variable definitions
from config import *
from vista.adv_plotter import labels

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

def dump_adversarial_example_image_batch(orig_vertices, adex, faces, orig_class, targets, logits, perturbed_logits, file_path):
    # orig_v_list = []
    # adex_v_list = []
    # color_list = []
    # faces_list = []
    # target_list = []
    # success_list = []
    # for i in range(orig_vertices.shape[0]):
    #     color = (orig_vertices[i] - adex[i]).norm(p=2, dim=-1)
    #     orig_v_list.append(orig_vertices[i])
    #     adex_v_list.append(adex[i])
    #     faces_list.append(faces[i])

    # p, _ = plot_mesh_montage([orig_vertices[0].T, adex[0].T], [faces[0], faces[0]], screenshot=True)
    # path = os.path.join(file_path, "test_examples.png")
    # p.show(screenshot=path, full_screen=True)

    perturbed_l = []
    faces_l = []
    color_l = []
    target_l = []
    success_l = []
    class_success_l = []
    classified_as_ = logits.data.max(1)[1]
    perturbed_class_ = perturbed_logits.data.max(1)[1]
    # fill the lists with needed information from main list
    for i in range(orig_vertices.shape[0]):
        perturbed = adex[i]
        pos = orig_vertices[i]
        color = (pos - perturbed).norm(p=2, dim=0)
        original_class = orig_class[i].item()
        classified_as = classified_as_[i].item()
        perturbed_class = perturbed_class_[i].item()
        target = targets[i].item()

        class_success_l.append((classified_as == original_class))
        success_l.append((perturbed_class == target) | (original_class == target))

        target_l.append([labels[original_class], labels[target]])
        perturbed_l.append(perturbed)
        faces_l.append(faces[i])
        color_l.append(color)

    # Plot all:
    p, _ = plot_mesh_montage(perturbed_l, fb=faces_l, clrb=color_l, labelb=target_l,
                             success=success_l, classifier_success=class_success_l,
                             cmap='OrRd', screenshot=True)
    path = os.path.join(file_path, "test_examples.png")
    p.show(screenshot=path, full_screen=True)