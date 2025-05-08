import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics

from dataloader import get_loader
from vgg import raw_vgg19, dy_vgg19

def main():
	args = parse.parse_args()
	name = args.name
	epoches = args.epoches
	model_name = args.model_name
	
	output_path = os.path.join('./output', name)
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	torch.backends.cudnn.benchmark=True

	train_data, val_data, test_data, train_loader, val_loader, test_loader = get_loader(args)
	train_dataset_size = len(train_data)
	val_dataset_size = len(val_data)

	model = eval(args.model)

	if args.continue_train:
		print('HI')
		model_path = args.model_path
		state_dict = torch.load(model_path)
		model.load_state_dict(state_dict)

	model = model.to(f'cuda:{device_ids[0]}')
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	model = nn.DataParallel(model)
	best_model_wts = model.state_dict()
	best_acc = 0.0
	iteration = 0

	train_acc_list = []
	accs_list = []
	train_loss_list = []
	val_loss_list = []

	with open(args.result_file, "w") as file:
		for epoch in range(epoches):
			file.write('Epoch {}/{}\n'.format(epoch+1, epoches))
			file.write('-'*10 + '\n')
			print('Epoch {}/{}'.format(epoch+1, epoches))
			print('-'*10)
			model.train()
			train_loss = 0.0
			train_corrects = 0.0
			train_preds = []
			targets = []
			probability = []
		
			for item in train_loader:
				image, labels = item['image'].to(f'cuda:{device_ids[0]}'), item['label'].to(f'cuda:{device_ids[0]}')
				iter_loss = 0.0
				iter_corrects = 0.0

				optimizer.zero_grad()
				outputs = model(image)

				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		
				iter_loss = loss.data.item()
				train_loss += iter_loss
				iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
				train_corrects += iter_corrects
				iteration += 1
				# if not (iteration % 20):
				# 	print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
					
				train_preds.extend(preds.detach().cpu().numpy()) #預測結果
				targets.extend(labels.data.detach().cpu().numpy()) #真實結果
		
			train_epoch_loss = train_loss / train_dataset_size
			train_epoch_acc = train_corrects / train_dataset_size
		
			train_preds = np.array(train_preds)
			train_targets = np.array(targets)
			train_acc = metrics.accuracy_score(train_preds,train_targets)
			f1 = f1_score(train_targets, train_preds, average='macro')
			re = recall_score(train_targets, train_preds, average='macro')
			pr = precision_score(train_targets, train_preds, average='macro')

			# file.write('\ntrain_preds:\n{}\n'.format(train_preds))
			# file.write('\ntrain_targets:\n{}\n'.format(targets))

			# print('train_preds:',train_preds)
			# print('train_targets:',targets)

			file.write('\nepoch: {} , train loss: {:.4f} , train acc: {:.4f} , train f1: {:.4f}, train re: {:.4f}, train pr: {:.4f}\n'.format(epoch+1, train_epoch_loss, train_epoch_acc, f1, re, pr))
			print('epoch: {} , train loss: {:.4f} , train acc: {:.4f} , train f1: {:.4f}, train re: {:.4f}, train pr: {:.4f}'.format(epoch+1, train_epoch_loss, train_epoch_acc, f1, re, pr))


			val_loss = 0.0
			val_corrects = 0.0
		
			pred_labels = []
			target_labels = []
			model.eval()
			with torch.no_grad():
				for item in val_loader:
					image, labels = item['image'].to(f'cuda:{device_ids[0]}'), item['label'].to(f'cuda:{device_ids[0]}')

					outputs = model(image)

					_, preds = torch.max(outputs.data, 1)
					loss = criterion(outputs, labels)
					val_loss += loss.data.item()
					val_corrects += torch.sum(preds == labels.data).to(torch.float32)

					pred_labels.extend(preds.detach().cpu().numpy())
					target_labels.extend(labels.detach().cpu().numpy())

				val_epoch_loss = val_loss / val_dataset_size
				val_epoch_acc = val_corrects / val_dataset_size
				
				if val_epoch_acc > best_acc:
					best_acc = val_epoch_acc
					best_model_wts = model.state_dict()

				preds_arr=np.array(pred_labels)
				labs_arr= np.array(target_labels)    
				accs = accuracy_score(labs_arr, preds_arr)
				f1 = f1_score(labs_arr, preds_arr, average='macro')
				re = recall_score(labs_arr, preds_arr, average='macro')
				pr = precision_score(labs_arr, preds_arr, average='macro')

				# file.write('\nval_preds: {}\n'.format(pred_labels))
				# file.write('\nval_targets: {}\n'.format(target_labels))

				# print('val_preds:',pred_labels)
				# print('val_targets:',target_labels)

				file.write('\nepoch: {} , val loss: {:.4f} , val acc: {:.4f} , val f1: {:.4f}, val re: {:.4f}, val pr: {:.4f}\n'.format(epoch+1, val_epoch_loss, val_epoch_acc, f1, re, pr))
				print('epoch: {} , val loss: {:.4f} , val acc: {:.4f} , val f1: {:.4f}, val re: {:.4f}, val pr: {:.4f}'.format(epoch+1, val_epoch_loss, val_epoch_acc, f1, re, pr))

			scheduler.step()
			#if not (epoch % 40):
			torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))

			train_acc_list.append(train_acc)
			accs_list.append(accs)
			train_loss_list.append(train_epoch_loss)
			val_loss_list.append(val_epoch_loss)

		file.write('\nBest val Acc: {:.4f}\n'.format(best_acc))
		print('Best val Acc: {:.4f}'.format(best_acc))

		model.load_state_dict(best_model_wts)
		torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))

	print(f"結果已經寫入到 {args.result_file} 文件中。")


	# 繪製Accuracy和Loss的變化圖
	epochs_range = range(1, epoches + 1)
	plt.figure(figsize=(12, 4))

	# 繪製Accuracy圖
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
	plt.plot(epochs_range, accs_list, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	# 繪製Loss圖
	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, train_loss_list, label='Training Loss')
	plt.plot(epochs_range, val_loss_list, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')

	plt.savefig(os.path.join(output_path, 'training_validation_plots.png'))
	plt.show()


if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--name', '-n', type=str, default='dy_vgg19_k=6')
	parse.add_argument('--epoches', '-e', type=int, default='10')
	parse.add_argument('--image_size', type=int, default='256')
	parse.add_argument('--model_name', '-mn', type=str, default='dy_vgg19_k=6.pkl')
	parse.add_argument('--continue_train', type=bool, default=False)
	parse.add_argument('--model_path', '-mp', type=str, default='/ssd6/Roy/DL/DL_hw2/Firstpart/output/dy_vgg19/best.pkl')
	parse.add_argument('--workers', type=int, default='8')
	parse.add_argument('--model', type=str, default='dy_vgg19()')
	parse.add_argument('--result_file', type=str, default='dy_vgg19_k=6.txt')
	parse.add_argument('--diff_channel', type=str, default="RGB")

	parse.add_argument('--batch_size', '-bz', type=int, default=1)
	parse.add_argument('--root_dir', type=str, default="/ssd6/Roy/DL/DL_hw2/dataset/images")
	parse.add_argument('--train_file_path', type=str, default="/ssd6/Roy/DL/DL_hw2/dataset/images/train.txt")
	parse.add_argument('--val_file_path', type=str, default="/ssd6/Roy/DL/DL_hw2/dataset/images/val.txt")
	parse.add_argument('--test_file_path', type=str, default="/ssd6/Roy/DL/DL_hw2/dataset/images/test.txt")

	os.environ['CUDA_VISIBLE_DEVICES']='4' 
	device_ids = [0]
	
	main()