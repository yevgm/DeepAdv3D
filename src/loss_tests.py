import torch

from deep_adv_3d.loss import *
from utils.torch.nn import *
from model_trainer_main import load_datasets
from models.pointnet import PointNet
from models.deep_adv_3d_model1 import Regressor

# class AdversarialLoss_single_batch(LossFunction):
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, perturbed_logits, target) -> torch.Tensor:
#         Z = perturbed_logits
#         values, index = torch.sort(Z, dim=1)
#         index = index[-1]
#         argmax = index[-1] if index[-1] != target else index[-2]  # max{Z(i): i != target}
#         Z = Z[-1]
#         Ztarget, Zmax = Z[target], Z[argmax]
#         return torch.max(Zmax - Ztarget, 0)


if __name__ == '__main__':
    # load data
    # set seed for all platforms
    set_determinsitic_run()

    # Data Loading and pre-processing
    trainLoader, testLoader = load_datasets(train_batch=TRAIN_BATCH_SIZE, test_batch=TEST_BATCH_SIZE)

    # classifier and model definition
    classifier = PointNet(k=10, feature_transform=False, global_transform=False)
    classifier.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE),
                               strict=CLS_STRICT_PARAM_LOADING)  # strict = False for dropping running mean and var of train batchnorm
    model = Regressor(numVertices=K)

    if OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                      weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                     weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=0.5)


    for i, data in enumerate(trainLoader, 0):
        orig_vertices, label, _, eigvecs, vertex_area, targets, faces, edges = data
        # plot_mesh(orig_vertices[0], faces[0], grid_on=True)  # plot some mesh for debugging
        cur_batch_len = orig_vertices.shape[0]
        orig_vertices = orig_vertices.transpose(2, 1)

        optimizer.zero_grad()
        # get Eigenspace vector field
        eigen_space_v = self.model(orig_vertices).transpose(2, 1)

        # adversarial example (smoothly perturbed)
        adex = orig_vertices + torch.bmm(eigvecs, eigen_space_v).transpose(2, 1)

        # DEBUG - visualize the adex
        # if epoch == 30:  # plot first adex in a batch
        # plot_mesh(adex[0].transpose(0, 1), faces[0], grid_on=True)
        if PLOT_TRAIN_IMAGES & (step_cntr > 0) & (step_cntr % SHOW_TRAIN_SAMPLE_EVERY == 0):
            dump_adversarial_example_image(orig_vertices, adex, faces, step_cntr, tensor_log_dir)

        # no grad is already implemented in the constructor
        perturbed_logits, _, _ = self.classifier(adex)

        # debug
        logits, _, _ = self.classifier(orig_vertices)
        label = label[:, 0]
        pred_orig = logits.data.max(1)[1]
        pred_choice = perturbed_logits.data.max(1)[1]
        num_classified = pred_orig.eq(label).cpu().sum()
        num_misclassified = pred_choice.eq(targets).cpu().sum()

        MisclassifyLoss = AdversarialLoss(perturbed_logits, targets)
        if LOSS == 'l2':
            Similarity_loss = L2Similarity(orig_vertices, adex, vertex_area)
        else:
            Similarity_loss = LocalEuclideanSimilarity(orig_vertices.transpose(2, 1), adex.transpose(2, 1), edges)

        missloss = MisclassifyLoss()
        similarity_loss = Similarity_loss()
        loss = missloss + RECON_LOSS_CONST * similarity_loss

        # Back-propagation step
        loss.backward()
        # if epoch > 1:  # trying to draw the gradients flow, not working since our grads are None for some reason
        #     plot_grad_flow_v2(self.model.named_parameters())
        optimizer.step()

        # Metrics
        self.loss_values.append(loss.item())
        pred_choice = perturbed_logits.data.max(1)[1]
        num_misclassified = pred_choice.eq(targets).sum().cpu()

        # report to tensorboard
        report_to_tensorboard(writer, i, step_cntr, cur_batch_len, epoch, self.num_batch, loss.item(),
                              similarity_loss.item(), missloss.item(), num_misclassified)

        step_cntr += 1