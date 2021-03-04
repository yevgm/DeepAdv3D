# from __future__ import print_function
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm


def train(train_data,
			test_data,
			classifier:torch.nn.Module,
			batchSize:int,
			parameters_file:str,
			epoch_number:int = 1,
			learning_rate:float=1e-3,
		  	train:bool=True):

	blue = lambda x: '\033[94m' + x + '\033[0m'

	# print(len(train_data.dataset), len(test_data.dataset))
	# num_classes = int(len(train_data.dataset) / batchSize)
	# print('classes', num_classes)


	# load parameters:

	# if opt.model != '':
	#     classifier.load_state_dict(torch.load(opt.model))

	loss_values = []

	optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999))
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	if torch.cuda.is_available():
	    classifier.cuda()

	num_batch = int(len(train_data.dataset) / batchSize)

	if train==True:

		for epoch in range(epoch_number):

			for i, data in enumerate(train_data, 0):
				points, target = data
				target = target[:, 0]
				cur_batch_len = len(points)
				points = points.transpose(2, 1)
				if torch.cuda.is_available():
					points, target = points.cuda(), target.cuda()
				optimizer.zero_grad()
				classifier = classifier.train()
				pred, trans, trans_feat = classifier(points)
				pred = F.log_softmax(pred, dim=1)
				loss = F.nll_loss(pred, target)

				loss_values.append(loss.item())
				# if opt.feature_transform:
				#     loss += feature_transform_regularizer(trans_feat) * 0.001
				loss.backward()
				optimizer.step()
				pred_choice = pred.data.max(1)[1]
				correct = pred_choice.eq(target.data).cpu().sum()
				print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(cur_batch_len)))

				# if i % 3 == 0:
				# 	j, data = next(enumerate(test_data, 0))
				# 	points, target = data
				# 	target = target[:, 0]
				# 	points = points.transpose(2, 1)
				# 	if torch.cuda.is_available():
				# 		points, target = points.cuda(), target.cuda()
				# 	classifier = classifier.eval()
				# 	pred, _, _ = classifier(points)
				# 	pred = F.log_softmax(pred, dim=1)
				# 	loss = F.nll_loss(pred, target)
				# 	pred_choice = pred.data.max(1)[1]
				# 	correct = pred_choice.eq(target.data).cpu().sum()
				# 	print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(test_data.batch_size)))
			scheduler.step()
		torch.save(classifier.state_dict(), parameters_file)

	total_correct = 0
	total_testset = 0
	total_loss = 0
	test_loss_values = []
	for i,data in tqdm(enumerate(test_data, 0)):
		points, target = data
		target = target[:, 0]
		points = points.transpose(2, 1)
		if torch.cuda.is_available():
			points, target = points.cuda(), target.cuda()
		classifier = classifier.eval()
		pred, _, _ = classifier(points)
		pred = F.log_softmax(pred, dim=1)
		loss = F.nll_loss(pred, target)

		pred_choice = pred.data.max(1)[1]
		correct = pred_choice.eq(target.data).cpu().sum()
		test_loss_values.append(loss.item())
		total_correct += correct.item()
		total_testset += points.size()[0]

	test_accuracy = total_correct/len(test_data.dataset)
	test_mean_loss = sum(test_loss_values)/len(test_loss_values)
	return loss_values, test_mean_loss, test_accuracy