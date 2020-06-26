from __future__ import division
import os
import pickle
import tensorflow as tf
from tqdm import tqdm

import sys
sys.path.insert(0, '../preprocess/')
sys.path.insert(0, '../lib/')

from operations import *
from utils import *
from preprocess import *
import numpy as np
from six.moves import xrange
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

F = tf.app.flags.FLAGS

"""
Model class

"""
class model(object):
	def __init__(self, sess, patch_shape, extraction_step):
		self.sess = sess
		self.patch_shape = patch_shape
		self.extraction_step = extraction_step
		self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(4)]

	def parallel_conv(self, c, patch, filter, name):

		h0 = lrelu(conv2d_WN(patch[0], filter[0], name=name+'_1_conv'))
		#h0 = dropout(h0, name = name+"_1_dropout")
		patch[1] = tf.concat([h0, patch[1]], 3)
		h1 = lrelu(conv2d_WN(patch[1], filter[1], name=name+'_2_conv'))
		#h1 = dropout(h1, name = name+"_2_dropout")
		patch[2] = tf.concat([h0, h1, patch[2]], 3)
		h2 = lrelu(conv2d_WN(patch[2], filter[2], name=name+'_3_conv'))
		#h2 = dropout(h2, name = name+"_3_dropout")
		patch[3] = tf.concat([h0, h1, h2, patch[3]], 3)
		h3 = lrelu(conv2d_WN(patch[3], filter[3], name=name+'_4_conv'))
		#h3 = dropout(h3, name = name+"_4_dropout")
		return h0, h1, h2, h3

	def parallel_pool(self, patch):
		p0 = avg_pool2D(patch[0])
		p1 = avg_pool2D(patch[1])
		p2 = avg_pool2D(patch[2])
		p3 = avg_pool2D(patch[3])
		return p0, p1, p2, p3

	def combine_pool(self, c, patch, filter, name):
		p = tf.concat([patch[0], patch[1], patch[2], patch[3]], 3)
		h0 = lrelu(conv2d_WN(p, filter[0], name=name+'_1_conv'))
		#h0 = dropout(h0, name = name+"_1_dropout")
		h1 = lrelu(conv2d_WN(h0, filter[1], name=name+'_2_conv'))
		#h1 = dropout(h1, name = name+"_2_dropout")
		return h1

	def upsample(self, c, patch, layer, size, filter, name):
		u1 = deconv2d_WN(patch, size, name = name+'_deconv')
		concat_layer=[]
		for l in layer:
			concat_layer.append(l)
		concat_layer.append(u1)
		u1 = tf.concat(concat_layer, 3)
		h0 = lrelu(conv2d_WN(u1, filter, name=name+'_1_conv'))
		#h0 = dropout(h0, name =name+"_1_dropout")
		h1 = lrelu(conv2d_WN(h0, filter, name=name+'_2_conv'))
		#h1 = dropout(h1, name =name+"_2_dropout")
		return h1

	def discriminator(self, patch, pshape, reuse = False):
		with tf.variable_scope('D') as scope:
			if reuse:
				scope.reuse_variables()

			sh, sh1, sh2, sh3 = int(pshape[0]/8), int(pshape[0]/4), int(pshape[0]/2), int(pshape[0])

			##----------------------------------------------DownSample------------------------------------------------
			#Layer1 with output size of 32
			h0, h1, h2, h3 = self.parallel_conv(0, [patch, patch, patch, patch], [20, 20, 20, 20], 'd_layer1')
			#Parallel Max Pooling1
			p1_0, p1_1, p1_2, p1_3 = self.parallel_pool([h0, h1, h2, h3])
			#Layer2 with output size of 16
			h4, h5, h6, h7 = self.parallel_conv(4, [p1_0, p1_1, p1_2, p1_3], [40, 40, 40, 40], 'd_layer2')
			#Parallel Max Pooling2
			p2_0, p2_1, p2_2, p2_3 = self.parallel_pool([h4, h5, h6, h7])
			#Layer3 with output size of 8
			h8, h9, h10, h11 = self.parallel_conv(8, [p2_0, p2_1, p2_2, p2_3], [80, 80, 80, 80], 'd_layer3')
			#Parallel Max Pooling3
			p3_0, p3_1, p3_2, p3_3 = self.parallel_pool([h8, h9, h10, h11])
			#Layer4 with output size of 4
			h12, h13, h14, h15 = self.parallel_conv(12, [p3_0, p3_1, p3_2, p3_3], [160, 160, 160, 160], 'd_layer4')
			#Parallel Max Pooling3
			p4_0, p4_1, p4_2, p4_3 = self.parallel_pool([h12, h13, h14, h15])
			##---------------------------------------------------------------------------------------------------------
			##-----------------------------------------------UpSample--------------------------------------------------
			#Combine All layers of size 16 and UpSample1
			c1 = self.combine_pool(16, [p1_0, p1_1, p1_2, p1_3], [80, 40], 'd_combine_layer1')
			u1 = self.upsample(18, c1, [h0, h1, h2, h3], sh3, 20, 'd_uplayer1')
			#Combine All layers of size 8 and UpSample2
			c2 = self.combine_pool(20, [p2_0, p2_1, p2_2, p2_3], [160, 80], 'd_combine_layer2')
			u2_1 = self.upsample(22, c2, [h4, h5, h6, h7], sh2, 40, 'd_uplayer2_1')
			u2_2 = self.upsample(24, u2_1, [h0, h1, h2, h3, u1], sh3, 20, 'd_uplayer2_2')
			#Combine All layers of size 4 and Upsample3
			c3 = self.combine_pool(26, [p3_0, p3_1, p3_2, p3_3], [320, 160], 'd_combine_layer3')
			u3_1 = self.upsample(28, c3, [h8, h9, h10, h11], sh1, 80, 'd_uplayer3_1')
			u3_2 = self.upsample(30, u3_1, [h4, h5, h6, h7, u2_1], sh2, 40, 'd_uplayer3_2')
			u3_3 = self.upsample(32, u3_2, [h0, h1, h2, h3, u1, u2_2], sh3, 20, 'd_uplayer3_3')
			#Combine All layers of size 2 and Upsample4
			c4 = self.combine_pool(34, [p4_0, p4_1, p4_2, p4_3], [640, 320], 'd_combine_layer4')
			u4_1 = self.upsample(36, c4, [h12, h13, h14, h15], sh, 160, 'd_uplayer4_1')
			u4_2 = self.upsample(38, u4_1, [h8, h9, h10, h11, u3_1], sh1, 80, 'd_uplayer4_2')
			u4_3 = self.upsample(40, u4_2, [h4, h5, h6, h7, u2_1, u3_2], sh2, 40, 'd_uplayer4_3')
			u4_4 = self.upsample(42, u4_3, [h0, h1, h2, h3, u1, u2_2, u3_3], sh3, 20, 'd_uplayer4_4')
			##---------------------------------------------------------------------------------------------------------
			#FCN Layer1
			fcn1 = lrelu(conv2d_WN(u4_4, 10, name='d_fcn1_conv'))
			#fcn1 = dropout(fcn1, name = "fcn1_dropout")
			fcn2 = conv2d_WN(fcn1, F.num_classes, name='d_fcn2_conv')

			return fcn2, tf.nn.softmax(fcn2), c1

	def generator(self, z, phase):
		"""
		Parameters:
		* z - Noise vector for generating 3D patches
		* phase - boolean variable to represent phase of operation of batchnorm
		Returns:
		* generated 3D patches
		"""
		with tf.variable_scope('G') as scope:
			sh1, sh2, sh3, sh4 = int(self.patch_shape[0]/16), int(self.patch_shape[0]/8), int(self.patch_shape[0]/4), int(self.patch_shape[0]/2)

			h0 = linear(z, sh1 * sh1 * 512, 'g_h0_lin')
			h0 = tf.reshape(h0, [F.batch_size, sh1, sh1, 512])
			h0 = relu(self.g_bns[0](h0, phase))

			h1 = relu(self.g_bns[1](deconv2d(h0, [F.batch_size, sh2, sh2, 256], name = 'g_h1_deconv'),phase))

			h2 = relu(self.g_bns[2](deconv2d(h1, [F.batch_size, sh3, sh3, 128], name = 'g_h2_deconv'),phase))

			h3 = relu(self.g_bns[3](deconv2d(h2, [F.batch_size, sh4, sh4, 64], name = 'g_h3_deconv'),phase))

			h4 = deconv2d_WN(h3, F.num_mod, name='g_h4_deconv')

			return tf.nn.tanh(h4)


	"""
	Defines the Few shot GAN U-Net model and the corresponding losses

	"""

	def build_model(self):
		self.patches_lab = tf.placeholder(tf.float32, [F.batch_size, self.patch_shape[0], self.patch_shape[1], F.num_mod], name='real_images_l')
		self.patches_unlab = tf.placeholder(tf.float32, [F.batch_size, self.patch_shape[0], self.patch_shape[1], F.num_mod], name='real_images_unl')

		self.z_gen = tf.placeholder(tf.float32, [None, F.noise_dim], name='noise')
		self.labels = tf.placeholder(tf.uint8, [F.batch_size, self.patch_shape[0], self.patch_shape[1]], name='image_labels')
		self.phase = tf.placeholder(tf.bool)

		#To make one hot of labels
		self.labels_1hot = tf.one_hot(self.labels, depth = F.num_classes)

		# To generate samples from noise
		self.patches_fake = self.generator(self.z_gen, self.phase)

		# Forward pass through network with different kinds of training patches
		self.D_logits_lab, self.D_probdist, _= self.discriminator(self.patches_lab, self.patch_shape, reuse = False)
		self.D_logits_unlab, _, self.features_unlab = self.discriminator(self.patches_unlab, self.patch_shape, reuse = True)
		self.D_logits_fake, _, self.features_fake = self.discriminator(self.patches_fake, self.patch_shape, reuse = True)

		# To obtain Validation Output
		self.Val_output = tf.argmax(self.D_probdist, axis=-1)

		# Supervised loss
		# Weighted cross entropy loss (You can play with these values)
		# Weights of different class are: Background- 0.33, vessel- 1.2
		class_weights = tf.constant([[0.33, 1.2]])
		weights = tf.reduce_sum(class_weights * self.labels_1hot, axis=-1)
		unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_logits_lab, labels=self.labels_1hot)
		weighted_losses = unweighted_losses * weights
		self.d_loss_lab = tf.reduce_mean(weighted_losses)

		# Unsupervised loss
		self.unl_lsexp = tf.reduce_logsumexp(self.D_logits_unlab, -1)
		self.fake_lsexp = tf.reduce_logsumexp(self.D_logits_fake, -1)
		# Unlabeled loss
		self.true_loss = - F.tlw * tf.reduce_mean(self.unl_lsexp) + F.tlw * tf.reduce_mean(tf.nn.softplus(self.unl_lsexp))
		# Fake loss
		self.fake_loss = F.flw * tf.reduce_mean(tf.nn.softplus(self.fake_lsexp))
		self.d_loss_unlab = self.true_loss + self.fake_loss

		#Total discriminator loss
		self.d_loss = self.d_loss_lab + self.d_loss_unlab

		#Feature matching loss
		self.g_loss_fm = tf.reduce_mean(tf.abs(tf.reduce_mean(self.features_unlab,0) - tf.reduce_mean(self.features_fake,0)))

		# Total Generator Loss
		self.g_loss = self.g_loss_fm


		t_vars = tf.trainable_variables()

		#define the trainable variables
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()


	"""
	Train function
	Defines learning rates and optimizers.
	Performs Network update and saves the losses
	"""
	def train(self):

		# Instantiate the dataset class
		data = dataset_badGAN(num_classes = F.num_classes,extraction_step = self.extraction_step, number_images_training = F.number_train_images, batch_size = F.batch_size, patch_shape = self.patch_shape, number_unlab_images_training = F.number_train_unlab_images, data_directory = F.data_directory)

		# Optimizer operations
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			d_optim = tf.train.AdamOptimizer(F.learning_rate_D, beta1=F.beta1D).minimize(self.d_loss,var_list=self.d_vars)
			g_optim = tf.train.AdamOptimizer(F.learning_rate_G, beta1=F.beta1G).minimize(self.g_loss,var_list=self.g_vars)
			
		tf.global_variables_initializer().run()

		# Load checkpoints if required
		if F.load_chkpt:
			try:
				load_model(F.checkpoint_dir, self.sess, self.saver)
				print("\n [*] Checkpoint loaded succesfully!")
			except:
				print("\n [!] Checkpoint loading failed!")
		else:
			print("\n [*] Checkpoint load not required.")

		# Load the validation data
		patches_val, labels_val_patch, labels_val = preprocess_dynamic_lab(F.data_directory, F.num_classes, self.extraction_step, self.patch_shape, F.number_train_images, validating = F.training, testing = F.testing, num_images_testing = F.number_test_images)

		predictions_val = np.zeros((patches_val.shape[0], self.patch_shape[0], self.patch_shape[1]),dtype="uint8")
		max_par = 0.0
		max_loss = 100
		for epoch in xrange(1, int(F.epoch)):
			idx = 0
			batch_iter_train = data.batch_train()
			total_val_loss = 0
			total_train_loss_CE = 0
			total_train_loss_UL = 0
			total_train_loss_FK = 0
			total_gen_FMloss = 0

			for patches_lab, patches_unlab, labels in batch_iter_train:
				# Network update
				sample_z_gen = np.random.uniform(-1, 1, [F.batch_size, F.noise_dim]).astype(np.float32)

				_ = self.sess.run(d_optim,feed_dict = {self.patches_lab:patches_lab, self.patches_unlab:patches_unlab, self.z_gen:sample_z_gen, self.labels:labels, self.phase: True})

				_ = self.sess.run(g_optim, feed_dict = {self.patches_unlab:patches_unlab, self.z_gen:sample_z_gen,
                                                                  self.z_gen:sample_z_gen, self.phase: True})

				feed_dict = {self.patches_lab:patches_lab, self.patches_unlab:patches_unlab, self.z_gen:sample_z_gen, self.labels:labels, self.phase: True}

				# Evaluate losses for plotting/printing purposes
				d_loss_lab = self.d_loss_lab.eval(feed_dict)
				d_loss_unlab_true = self.true_loss.eval(feed_dict)
				d_loss_unlab_fake = self.fake_loss.eval(feed_dict)
				g_loss_fm = self.g_loss_fm.eval(feed_dict)

				total_train_loss_CE = total_train_loss_CE + d_loss_lab
				total_train_loss_UL = total_train_loss_UL + d_loss_unlab_true
				total_train_loss_FK = total_train_loss_FK + d_loss_unlab_fake
				total_gen_FMloss = total_gen_FMloss + g_loss_fm

				idx += 1

				print(("Epoch:[%2d] [%4d/%4d] Labeled loss:%.2e Unlabeled loss:%.2e Fake loss:%.2e Generator loss:%.8f \n")% (epoch, idx,data.num_batches,d_loss_lab,d_loss_unlab_true,d_loss_unlab_fake,g_loss_fm))

				if (idx%7000 == 0):

					# Save the curret model
					save_model(F.checkpoint_dir, self.sess, self.saver)

					avg_train_loss_CE=total_train_loss_CE/(idx*1.0)
					avg_train_loss_UL=total_train_loss_UL/(idx*1.0)
					avg_train_loss_FK=total_train_loss_FK/(idx*1.0)
					avg_gen_FMloss=total_gen_FMloss/(idx*1.0)

					print('\n\n')

					total_batches = int(patches_val.shape[0]/F.batch_size)
					print("Total number of batches for validation: ", total_batches)

					# Prediction of validation patches
					for batch in range(total_batches):
						patches_feed = patches_val[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]
						labels_feed = labels_val_patch[batch*F.batch_size:(batch+1)*F.batch_size,:,:]
						feed_dict={self.patches_lab:patches_feed, self.labels:labels_feed, self.phase:False}
						preds = self.Val_output.eval(feed_dict)
						val_loss = self.d_loss_lab.eval(feed_dict)

						predictions_val[batch*F.batch_size:(batch+1)*F.batch_size,:,:]=preds
						print(("Validated Patch:[%8d/%8d]")%(batch,total_batches))
						total_val_loss=total_val_loss+val_loss

					# To compute average patchvise validation loss(cross entropy loss)
					avg_val_loss=total_val_loss/(total_batches*1.0)

					print("All validation patches Predicted")

					print("Shape of predictions_val, min and max:",predictions_val.shape,np.min(predictions_val), np.max(predictions_val))

					# To stitch back the patches into an entire image
					val_image_pred = recompose3D_overlap(predictions_val, 512, 512, self.extraction_step[0], self.extraction_step[1])
					val_image_pred = val_image_pred.astype('uint8')

					print("Shape of Predicted Output Groundtruth Images:",val_image_pred.shape, np.unique(val_image_pred), np.unique(labels_val), np.mean(val_image_pred),np.mean(labels_val))

					pred2d = np.reshape(val_image_pred, (val_image_pred.shape[0] * 512 * 512))
					lab2d = np.reshape(labels_val, (labels_val.shape[0] * 512 * 512))

					# For printing the validation results
					F1_score = f1_score(lab2d, pred2d,[0, 1], average = None)
					auc_score = roc_auc_score(lab2d, pred2d)
					print("Validation AUC_ROC Score.... ")
					print("Score:",auc_score)

					print("Validation Dice Coefficient.... ")
					print("Background:",F1_score[0])
					print("Blood Vessel:",F1_score[1])

					# To Save the best model
					if(max_par<(auc_score)):
						max_par = auc_score
						save_model(F.best_checkpoint_dir, self.sess, self.saver)
						print("Best checkpoint updated from validation results.")

					# To save the losses for plotting
					print("Average Validation Loss:",avg_val_loss)
					with open('TrainScore/Val_loss_GAN.txt', 'a') as f:
						f.write('%.2e \n' % avg_val_loss)
					with open('TrainScore/Train_loss_CE.txt', 'a') as f:
						f.write('%.2e \n' % avg_train_loss_CE)
					with open('TrainScore/Train_loss_UL.txt', 'a') as f:
						f.write('%.2e \n' % avg_train_loss_UL)
					with open('TrainScore/Train_loss_FK.txt', 'a') as f:
						f.write('%.2e \n' % avg_train_loss_FK)
					with open('TrainScore/Train_loss_FM.txt', 'a') as f:
						f.write('%.2e \n' % avg_gen_FMloss)
					with open('TrainScore/BloodVessel_F1Score.txt', 'a') as f:
						f.write('%.2e \n' % F1_score[1])
					with open('TrainScore/Background_F1Score.txt', 'a') as f:
						f.write('%.2e \n' % F1_score[0])
					with open('TrainScore/AUCScore.txt', 'a') as f:
						f.write('%.2e \n' % auc_score)

					idx = 0
					total_val_loss = 0
					total_train_loss_CE = 0
					total_train_loss_UL = 0
					total_train_loss_FK = 0
					total_gen_FMloss = 0
		return
