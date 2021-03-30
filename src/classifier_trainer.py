# from __future__ import print_function
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import numpy as np

# variable definitions
from config import *

# repository modules
from models.Origin_pointnet import PointNetCls, Regressor
from model1.loss import *
from model1.utils import *
from model1.tensor_board import *

def weights_init_normal(m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
            m.bias.data.fill_(0)

def train(train_data,
			test_data,
			classifier:torch.nn.Module,
			batchSize:int,
			parameters_file:str,
			epoch_number:int = 1,
			learning_rate:float=1e-3,
		  	train:bool=True):

	# load parameters:
	# classifier.load_state_dict(torch.load(PARAMS_FILE, map_location=DEVICE))

	# initialize weights with normal distribution
	classifier.apply(weights_init_normal)

	# pre-train preparations
	generate_data_output_dir()
	now = datetime.now()
	d = now.strftime("%b-%d-%Y_%H-%M-%S")
	tensor_log_dir = generate_new_tensorboard_results_dir(d, model='classifier')
	writer = SummaryWriter(tensor_log_dir, flush_secs=FLUSH_RESULTS)
	save_weights_dir = os.path.join(tensor_log_dir, PARAM_FILE_NAME)

	if OPTIMIZER == 'AdamW':
		optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, betas=(0.9, 0.999),
									  weight_decay=WEIGHT_DECAY)
	else:
		optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, betas=(0.9, 0.999),
									 weight_decay=WEIGHT_DECAY)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=0.5)


	step_cntr = 0
	if train:
		for epoch in range(epoch_number):
			if epoch != 0:
				scheduler.step()
			for i, data in enumerate(train_data, 0):
				points, target, _, _, _, _, _, _ = data
				target = target[:, 0]
				cur_batch_len = len(points)
				points = points.transpose(2, 1)

				optimizer.zero_grad()
				classifier = classifier.train()  # changes some of the classifier's layers, like batchnorm
				pred, trans, trans_feat = classifier(points)
				pred = F.log_softmax(pred, dim=1)
				loss = F.nll_loss(pred, target)

				loss.backward()
				optimizer.step()

				# Metrics
				pred_choice = pred.data.max(1)[1]
				correct = pred_choice.eq(target.data).cpu().sum()

				# report to tensorboard
				classifier_report_to_tensorboard(writer, i, step_cntr, cur_batch_len, epoch, len(train_data), loss.item(), correct)

				step_cntr += 1

			if (step_cntr > 0) & (step_cntr % SAVE_PARAMS_EVERY == 0):
				torch.save(classifier.state_dict(), save_weights_dir)


		torch.save(classifier.state_dict(), save_weights_dir)

	# evaluate the model right after training
	evaluate(test_data, classifier, test_param_dir=tensor_log_dir)


def evaluate(test_data, classifier, test_param_dir=TEST_PARAMS_DIR):
	# pre-test preparations
	s_writer = SummaryWriter(test_param_dir, flush_secs=FLUSH_RESULTS)
	test_param_file = get_param_file(test_param_dir)
	classifier.load_state_dict(torch.load(test_param_file, map_location=DEVICE))
	classifier = classifier.eval()  # set to test mode


	num_classified = 0
	running_total_loss = 0.0
	test_len = len(test_data.dataset)
	with torch.no_grad():
		for epoch in range(TEST_EPOCHS):
			# the evaluation is based purely on the classifications amount on the test set
			for i, data in enumerate(test_data):
				orig_vertices, label, _, _, _, _, _, _ = data
				label = label[:, 0]
				orig_vertices = orig_vertices.transpose(2, 1)

				# pass through classifier
				logits, _, _ = classifier(orig_vertices)
				pred = F.log_softmax(logits, dim=1)
				loss = F.nll_loss(pred, label)

				pred_choice = logits.data.max(1)[1]
				num_classified += pred_choice.eq(label).cpu().sum()

				running_total_loss += loss

		classifier_report_test_to_tensorboard(s_writer, running_total_loss / TEST_EPOCHS,
											  num_classified, TEST_EPOCHS * test_len)

	# total_correct = 0
	# total_testset = 0
	# total_loss = 0
	# test_loss_values = []
	# for i,data in tqdm(enumerate(test_data, 0)):
	# 	points, target = data
	# 	target = target[:, 0]
	# 	points = points.transpose(2, 1)
	# 	if torch.cuda.is_available():
	# 		points, target = points.cuda(), target.cuda()
	# 	classifier = classifier.eval()
	# 	pred, _, _ = classifier(points)
	# 	pred = F.log_softmax(pred, dim=1)
	# 	loss = F.nll_loss(pred, target)
	#
	# 	pred_choice = pred.data.max(1)[1]
	# 	correct = pred_choice.eq(target.data).cpu().sum()
	# 	test_loss_values.append(loss.item())
	# 	total_correct += correct.item()
	# 	total_testset += points.size()[0]
	#
	# test_accuracy = total_correct/len(test_data.dataset)
	# test_mean_loss = sum(test_loss_values)/len(test_loss_values)
	# return loss_values, test_mean_loss, test_accuracy