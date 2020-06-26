from __future__ import division
import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
from imageio import imread, imwrite
from PIL import Image

import sys
sys.path.insert(0, '../preprocess/')
sys.path.insert(0, '../lib/')

from operations import *
from utils import *
from preprocess import *



F = tf.app.flags.FLAGS


# Function to save predicted images as .png file in results folder
def save_image(direc, i, num):
	img = (i * 255).astype(np.uint8)
	imgname = 'result_'+str(num)+'.png'
	im = Image.fromarray(img)
	im.save(os.path.join(direc,imgname))

def parallel_conv(c, patch, filter, name):

	h0 = lrelu(conv2d_WN(patch[0], filter[0], name=name+'_1_conv'))
	h0 = dropout(h0, name = name+"_1_dropout")
	patch[1] = tf.concat([h0, patch[1]], 3)
	h1 = lrelu(conv2d_WN(patch[1], filter[1], name=name+'_2_conv'))
	h1 = dropout(h1, name = name+"_2_dropout")
	patch[2] = tf.concat([h0, h1, patch[2]], 3)
	h2 = lrelu(conv2d_WN(patch[2], filter[2], name=name+'_3_conv'))
	h2 = dropout(h2, name = name+"_3_dropout")
	patch[3] = tf.concat([h0, h1, h2, patch[3]], 3)
	h3 = lrelu(conv2d_WN(patch[3], filter[3], name=name+'_4_conv'))
	h3 = dropout(h3, name = name+"_4_dropout")
	return h0, h1, h2, h3

def parallel_pool(patch):
	p0 = avg_pool2D(patch[0])
	p1 = avg_pool2D(patch[1])
	p2 = avg_pool2D(patch[2])
	p3 = avg_pool2D(patch[3])
	return p0, p1, p2, p3

def combine_pool(c, patch, filter, name):
	p = tf.concat([patch[0], patch[1], patch[2], patch[3]], 3)
	h0 = lrelu(conv2d_WN(p, filter[0], name=name+'_1_conv'))
	h0 = dropout(h0, name = name+"_1_dropout")
	h1 = lrelu(conv2d_WN(h0, filter[1], name=name+'_2_conv'))
	h1 = dropout(h1, name = name+"_2_dropout")
	return h1

def upsample(c, patch, layer, size, filter, name):
	u1 = deconv2d_WN(patch, size, name = name+'_deconv')
	concat_layer=[]
	for l in layer:
		concat_layer.append(l)
	concat_layer.append(u1)
	u1 = tf.concat(concat_layer, 3)
	h0 = lrelu(conv2d_WN(u1, filter, name=name+'_1_conv'))
	h0 = dropout(h0, name =name+"_1_dropout")
	h1 = lrelu(conv2d_WN(h0, filter, name=name+'_2_conv'))
	h1 = dropout(h1, name =name+"_2_dropout")
	return h1

# Same discriminator network as in model file
def trained_dis_network(patch, pshape, reuse = False):
	with tf.variable_scope('D') as scope:
		if reuse:
			scope.reuse_variables()

		sh, sh1, sh2, sh3 = int(pshape[0]/8), int(pshape[0]/4), int(pshape[0]/2), int(pshape[0])

		##----------------------------------------------DownSample------------------------------------------------
		#Layer1 with output size of 32
		h0, h1, h2, h3 = parallel_conv(0, [patch, patch, patch, patch], [20, 20, 20, 20], 'd_layer1')
		#Parallel Max Pooling1
		p1_0, p1_1, p1_2, p1_3 = parallel_pool([h0, h1, h2, h3])
		#Layer2 with output size of 16
		h4, h5, h6, h7 = parallel_conv(4, [p1_0, p1_1, p1_2, p1_3], [40, 40, 40, 40], 'd_layer2')
		#Parallel Max Pooling2
		p2_0, p2_1, p2_2, p2_3 = parallel_pool([h4, h5, h6, h7])
		#Layer3 with output size of 8
		h8, h9, h10, h11 = parallel_conv(8, [p2_0, p2_1, p2_2, p2_3], [80, 80, 80, 80], 'd_layer3')
		#Parallel Max Pooling3
		p3_0, p3_1, p3_2, p3_3 = parallel_pool([h8, h9, h10, h11])
		#Layer4 with output size of 4
		h12, h13, h14, h15 = parallel_conv(12, [p3_0, p3_1, p3_2, p3_3], [160, 160, 160, 160], 'd_layer4')
		#Parallel Max Pooling3
		p4_0, p4_1, p4_2, p4_3 = parallel_pool([h12, h13, h14, h15])
		##---------------------------------------------------------------------------------------------------------
		##-----------------------------------------------UpSample--------------------------------------------------
		#Combine All layers of size 16 and UpSample1
		c1 = combine_pool(16, [p1_0, p1_1, p1_2, p1_3], [80, 40], 'd_combine_layer1')
		u1 = upsample(18, c1, [h0, h1, h2, h3], sh3, 20, 'd_uplayer1')
		#Combine All layers of size 8 and UpSample2
		c2 = combine_pool(20, [p2_0, p2_1, p2_2, p2_3], [160, 80], 'd_combine_layer2')
		u2_1 = upsample(22, c2, [h4, h5, h6, h7], sh2, 40, 'd_uplayer2_1')
		u2_2 = upsample(24, u2_1, [h0, h1, h2, h3, u1], sh3, 20, 'd_uplayer2_2')
		#Combine All layers of size 4 and Upsample3
		c3 = combine_pool(26, [p3_0, p3_1, p3_2, p3_3], [320, 160], 'd_combine_layer3')
		u3_1 = upsample(28, c3, [h8, h9, h10, h11], sh1, 80, 'd_uplayer3_1')
		u3_2 = upsample(30, u3_1, [h4, h5, h6, h7, u2_1], sh2, 40, 'd_uplayer3_2')
		u3_3 = upsample(32, u3_2, [h0, h1, h2, h3, u1, u2_2], sh3, 20, 'd_uplayer3_3')
		#Combine All layers of size 2 and Upsample4
		c4 = combine_pool(34, [p4_0, p4_1, p4_2, p4_3], [640, 320], 'd_combine_layer4')
		u4_1 = upsample(36, c4, [h12, h13, h14, h15], sh, 160, 'd_uplayer4_1')
		u4_2 = upsample(38, u4_1, [h8, h9, h10, h11, u3_1], sh1, 80, 'd_uplayer4_2')
		u4_3 = upsample(40, u4_2, [h4, h5, h6, h7, u2_1, u3_2], sh2, 40, 'd_uplayer4_3')
		u4_4 = upsample(42, u4_3, [h0, h1, h2, h3, u1, u2_2, u3_3], sh3, 20, 'd_uplayer4_4')
		##---------------------------------------------------------------------------------------------------------
		#FCN Layer1
		fcn1 = lrelu(conv2d_WN(u4_4, 10, name='d_fcn1_conv'))
		#fcn1 = dropout(fcn1, name = "fcn1_dropout")
		fcn2 = conv2d_WN(fcn1, F.num_classes, name='d_fcn2_conv')

		return tf.nn.softmax(fcn2)

"""
 Function to test the model and evaluate the predicted images
 Parameters:
 * patch_shape - shape of the patch
 * extraction_step - stride while extracting patches
"""
def test(patch_shape, extraction_step):

	with tf.Graph().as_default():
		test_patches = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], patch_shape[1], F.num_mod], name='real_patches')

		# Define the network
		output_soft = trained_dis_network(test_patches, patch_shape, reuse=None)

		# To convert from one hat form
		output=tf.argmax(output_soft, axis=-1)
		print("Output Patch Shape:",output.get_shape())

		# To load the saved checkpoint
		saver = tf.train.Saver()
		with tf.Session() as sess:
			try:
				load_model(F.best_checkpoint_dir, sess, saver)
				print(" Checkpoint loaded succesfully!....\n")
			except:
				print(" [!] Checkpoint loading failed!....\n")
				return

			# Get patches from test images
			patches_test, labels_test = preprocess_dynamic_lab(F.data_directory, F.num_classes,extraction_step,patch_shape, F.number_train_images,validating=F.training, testing=F.testing,num_images_testing=F.number_test_images)
			total_batches = int(patches_test.shape[0]/F.batch_size)

			# Array to store the prediction results
			predictions_test = np.zeros((patches_test.shape[0], patch_shape[0], patch_shape[1]))

			print("max and min of patches_test:", np.min(patches_test), np.max(patches_test))

			print("Total number of Batches: ", total_batches)

			# Batch wise prediction
			for batch in range(total_batches):
				patches_feed = patches_test[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]
				preds = sess.run(output, feed_dict={test_patches:patches_feed})
				predictions_test[batch*F.batch_size:(batch+1)*F.batch_size,:,:]=preds
				print(("Processed_batch:[%8d/%8d]")%(batch,total_batches))

			print("All patches Predicted")

			print("Shape of predictions_test, min and max:", predictions_test.shape, np.min(predictions_test), np.max(predictions_test))

			# To stitch the image back
			images_pred = recompose3D_overlap(predictions_test, 512, 512, extraction_step[0], extraction_step[1])

			print("Shape of Predicted Output Groundtruth Images:",images_pred.shape, np.min(images_pred), np.max(images_pred), np.mean(images_pred),np.mean(labels_test))


			# To save the images
			for i in range(F.number_test_images):
				pred2d = np.reshape(images_pred[i], (512 * 512))
				lab2d = np.reshape(labels_test[i], (512 * 512))

				# Accuracy
				ind_acc = accuracy_score(lab2d, pred2d)
				with open('TestScore/Ind_Accuracy.txt', 'a') as f:
					f.write('%.2e \n' % ind_acc)

				ind_bal_acc = balanced_accuracy_score(lab2d, pred2d)
				with open('TestScore/Ind_balanced_accuracy.txt', 'a') as f:
					f.write('%.2e \n' % ind_bal_acc)

				#F1 Score
				ind_F1_score = f1_score(lab2d, pred2d, [0,1], average = None)
				with open('TestScore/Ind_F1_score.txt', 'a') as f:
					f.write('%.2e \n' % ind_F1_score[1])

				#Precision Score
				ind_precise = precision_score(lab2d, pred2d, [0,1], average = None)
				with open('TestScore/Ind_Precision.txt', 'a') as f:
					f.write('%.2e \n' % ind_precise[1])

				#Recall Score
				ind_recall = recall_score(lab2d, pred2d, [0,1], average = None)
				with open('TestScore/Ind_Recall.txt', 'a') as f:
					f.write('%.2e \n' % ind_recall[1])

				#AUC_ROC Score
				ind_auc = roc_auc_score(lab2d, pred2d)
				with open('TestScore/Ind_AUCScore.txt', 'a') as f:
					f.write('%.2e \n' % ind_auc)

				# Specificity and Sensitivity
				tn, fp, fn, tp = confusion_matrix(lab2d, pred2d).ravel()
				ind_Sensitivity = (tp) / (tp + fn)
				ind_Specificity = (tn) / (tn + fp)

				with open('TestScore/Ind_Specificity.txt', 'a') as f:
					f.write('%.2e \n' % ind_Specificity)

				with open('TestScore/Ind_Sensitivity.txt', 'a') as f:
					f.write('%.2e \n' % ind_Sensitivity)

				save_image(F.results_dir, images_pred[i], F.number_train_images+i+1)

			# Evaluation
			pred2d = np.reshape(images_pred, (images_pred.shape[0] * 512 * 512))
			lab2d = np.reshape(labels_test, (labels_test.shape[0] * 512 * 512))

			# Accuracy
			acc = accuracy_score(lab2d, pred2d)
			with open('TestScore/Accuracy.txt', 'a') as f:
				f.write('%.2e \n' % acc)

			bal_acc = balanced_accuracy_score(lab2d, pred2d)
			with open('TestScore/balanced_accuracy.txt', 'a') as f:
				f.write('%.2e \n' % bal_acc)

			#F1 Score
			F1_score = f1_score(lab2d, pred2d, [0,1], average = None)
			with open('TestScore/F1_score.txt', 'a') as f:
				f.write('%.2e \n' % F1_score[1])

			#Precision Score
			precise = precision_score(lab2d, pred2d, [0,1], average = None)
			with open('TestScore/Precision.txt', 'a') as f:
				f.write('%.2e \n' % precise[1])

			#Recall Score
			recall = recall_score(lab2d, pred2d, [0,1], average = None)
			with open('TestScore/Recall.txt', 'a') as f:
				f.write('%.2e \n' % recall[1])

			#AUC_ROC Score
			auc = roc_auc_score(lab2d, pred2d)
			with open('TestScore/AUCScore.txt', 'a') as f:
				f.write('%.2e \n' % auc)

			# Specificity and Sensitivity
			tn, fp, fn, tp = confusion_matrix(lab2d, pred2d).ravel()
			Sensitivity = (tp) / (tp + fn)
			Specificity = (tn) / (tn + fp)

			with open('TestScore/Specificity.txt', 'a') as f:
				f.write('%.2e \n' % Specificity)

			with open('TestScore/Sensitivity.txt', 'a') as f:
				f.write('%.2e \n' % Sensitivity)

			print("Testing Auc...... ", auc)
