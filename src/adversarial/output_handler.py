import os
from datetime import datetime

from utils.ios import write_off
import vista.adv_plotter
from vista.adv_plotter import show_perturbation, show_all_perturbations
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),".."))

def save_results(example_list:list, batch_time=None):
    '''
    If saving one figure at a time, you mut pass a unique folder ID -
        batch_time
    '''
    out_folder = os.path.join(REPO_ROOT, 'outputFolder')
    rand_example_path = os.path.join(out_folder, 'random_examples')
    group_example_path = os.path.join(out_folder, 'group_examples')
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    if len(example_list) == 1:

        if not os.path.isdir(rand_example_path):
            os.mkdir(rand_example_path)

        adex = example_list[0]
        original_class = adex.y.item()
        classified_as = adex.logits.argmax().item()
        perturbed_class = adex.perturbed_logits.argmax().item()
        target = adex.target.item()

        # Debug:
        # perturbed_class = 1
        # target = 1
        if (original_class == target) | (classified_as != original_class):
            print('Original class equals target or shape is misclassified, saving .obj file aborted')
            return
        elif perturbed_class != target:
            print('Attack is not successful, saving .obj file aborted')
            return
        else:
            file_str = str(original_class) + '_to_' + str(target) + batch_time
            file_path = os.path.join(rand_example_path, file_str)

            v = adex.perturbed_pos.cpu().detach().numpy()
            f = adex.faces.cpu().detach().numpy()

            write_off(file_path, v, f)
            p = show_perturbation(example_list, screenshot=True)
            p.link_views()
            p.show(screenshot=file_path+'.png', full_screen=True)

    elif (len(example_list) > 1) & (not os.path.isdir(group_example_path)):
        os.mkdir(group_example_path)
    elif len(example_list) > 1:
        now = datetime.now()
        d = now.strftime("_%b-%d-%Y_%H-%M-%S")
        file_str = str(len(example_list)) + '_shapes' + d
        file_path = os.path.join(group_example_path, file_str)
        p = show_all_perturbations(example_list, screenshot=True)
        p.show(screenshot=file_path + '.png', full_screen=True)


def output_reader():
    pass