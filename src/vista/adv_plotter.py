import numpy as np
import torch
import matplotlib.pyplot as plt

from vista.geom_vis import plot_mesh, plot_mesh_montage

labels = [
    'Standing',
    'Leaning on table',
    'Against Wall',
    'Parallel hands',
    'Right leg up',
    'Standing head tilt',
    'Clapping hands',
    'Dancer',
    'Handshake',
    'Hands Up'
]

def show_perturbation(example_list, screenshot=False):
    adex = example_list[0] # unpack from list

    perturbed = adex.perturbed_pos.cpu()
    pos = adex.pos.cpu()
    p1 = adex.logits.cpu().detach().numpy()
    p2 = adex.perturbed_logits.cpu().detach().numpy()
    m = min([p1.min(), p2.min()])
    num_classes = p1.shape[1]

    x_ticks = np.array(range(num_classes), dtype=float)
    ax = plt.subplot(111)
    ax.bar(x_ticks - 0.2, (p1 - m)[0], width=0.4, color='b', align='center')
    ax.bar(x_ticks + 0.2, (p2 - m)[0], width=0.4, color='y', align='center')
    ax.legend(["standard", "perturbed towards " + str(adex.target.item())])
    ax.set_title("Class Probabilities with/without Perturbation")
    plt.show()
    color = (pos - perturbed).norm(p=2, dim=-1)

    # Plot one mesh:
    # plot_mesh(perturbed, f=adex.faces, clr=color, label='Perturbed',
    #           slabel='L2 perturbation difference', lighting=1, clr_map='rainbow')

    # Plot three meshes using lists:
    original_class = adex.y.item()
    classified_as = adex.logits.argmax().item()
    perturbed_class = adex.perturbed_logits.argmax().item()
    target = adex.target.item()
    class_success_l = (classified_as == original_class)
    success_l = (perturbed_class == target) | (original_class == target)

    gt_const_mid = color.max().item() / 2
    gt_const_min = color.min().item()
    vlist = [pos, perturbed, [pos, perturbed]]
    flist = [adex.faces, adex.faces, [adex.faces, adex.faces]]
    clist = [torch.zeros_like(color)+gt_const_min, color, [torch.zeros_like(color)+gt_const_mid, color]]
    Opacitylist = [1, 1, [0.35, 1]]
    p, _ = plot_mesh_montage(vlist, fb=flist, clrb=clist,
                            labelb=[labels[original_class], 'Perturbed to\n'+labels[target], 'Both'],
                            slabelb=[ 'GT', 'L2 difference', 'L2 difference'],
                            success=[class_success_l, success_l, None],
                            opacity=Opacitylist, cmap='OrRd', screenshot=screenshot)
    return p

def show_all_perturbations(example_list, screenshot=False):

    perturbed_l = []
    faces_l = []
    color_l = []
    target_l = []
    success_l = []
    class_success_l = []
    # fill the lists with needed information from main list
    for adex in example_list:
        perturbed = adex.perturbed_pos.cpu()
        pos = adex.pos.cpu()
        color = (pos - perturbed).norm(p=2, dim=-1)
        original_class = adex.y.item()
        classified_as = adex.logits.argmax().item()
        perturbed_class = adex.perturbed_logits.argmax().item()
        target = adex.target.item()

        class_success_l.append( (classified_as == original_class) )
        success_l.append((perturbed_class == target) | (original_class == target) )

        target_l.append([labels[original_class], labels[target]])
        perturbed_l.append(perturbed)
        faces_l.append(adex.faces)
        color_l.append(color)

    # Plot all:
    p, _ = plot_mesh_montage(perturbed_l, fb=faces_l, clrb=color_l, labelb=target_l,
                     success=success_l, classifier_success=class_success_l,
                      cmap='YlOrRd', screenshot=screenshot)
    return p
