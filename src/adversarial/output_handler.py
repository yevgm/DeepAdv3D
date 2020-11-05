import os
from datetime import datetime

from utils.ios import write_off
import vista.adv_plotter
from vista.adv_plotter import show_perturbation, show_all_perturbations
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),".."))

def save_results(example_list:list, testdata, CWparams=None, hyperParams=None,
                 folder_name=None, file_name=None):
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
        adv_coeff = adex.logger.adv_example.adversarial_coeff
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
            rand_example_path = os.path.join(rand_example_path, folder_name)
            if not os.path.isdir(rand_example_path):
                os.mkdir(rand_example_path)

            file_str = file_name + '-' + str(original_class) + '_PerturbedTo_' + str(target)
            file_path = os.path.join(rand_example_path, file_str)

            v = adex.perturbed_pos.cpu().detach().numpy()
            f = adex.faces.cpu().detach().numpy()

            # save .obj
            write_off(file_path, v, f)
            # save .png
            p = show_perturbation(example_list, testdata, screenshot=True)
            p.link_views() # not sure if it is needed
            p.show(screenshot=file_path+'.png', full_screen=True)
            # concatenate hyper params to .csv
            add_hp_to_csv(rand_example_path, file_str, CWparams, hyperParams,
                          adv_coeff)

    elif (len(example_list) > 1):
        if (not os.path.isdir(group_example_path)):
            os.mkdir(group_example_path)

        now = datetime.now()
        d = now.strftime("_%b-%d-%Y_%H-%M-%S")
        file_str = str(len(example_list)) + '_shapes' + d
        file_path = os.path.join(group_example_path, file_str)
        p = show_all_perturbations(example_list, screenshot=True)
        p.show(screenshot=file_path + '.png', full_screen=True)


def add_hp_to_csv(mapper_location, filename,  CWparams, hyperParams, adv_coeff):
    '''
    This function adds the adversarial example hyper-params to a given list
    '''
    mapper_file = os.path.join(mapper_location, 'Mapper.csv')
    lr = str(CWparams['learning_rate'])
    c = str(adv_coeff)
    reg_coeff = str(CWparams['regularization_coeff'])
    k = str(CWparams['k_nearest_neighbors'])
    cutoff = str(CWparams['knn_cutoff_parameter'])
    lowband = str(hyperParams['lowband_perturbation'])
    loss = str(hyperParams['similarity_loss'])

    if not os.path.isfile(mapper_file):
        with open(mapper_file, 'a') as f:
            f.write('Filename,'+'lr,'+'c,'+'reg_coeff,'+'k,'+'cutoff,'+'lowband,'+'loss'+'\n')

    # MIN_IT = "minimization_iterations"

    line = filename+','+lr+','+c+','+reg_coeff+','+k+','+cutoff+','+lowband+','+loss+'\n'
    with open(mapper_file, "a") as f:
        f.write(line)


def output_reader():
    pass