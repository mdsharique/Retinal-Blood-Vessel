import os
import glob
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil
from nipype.interfaces.ants import N4BiasFieldCorrection
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from sklearn.utils import shuffle
import tensorflow as tf
import scipy.misc
from tqdm import tqdm
from PIL import Image

#import pdb

F = tf.app.flags.FLAGS


seed = 7
np.random.seed(seed)

all_modalities={'T1','T2'}


def get_filename(set_name, case_idx, input_name, loc):
    pattern = '{0}/{1}/{3}/{2}.png'
    return pattern.format(loc, set_name, case_idx, input_name)

def get_set_name(case_idx):
    return 'Training' if case_idx < 114 else 'Testing'

def read_data(case_idx, input_name, loc):
    set_name = get_set_name(case_idx)
    image_path = get_filename(set_name, case_idx, input_name, loc)
    print(image_path)
    return Image.open(image_path)

def read_vol(case_idx, input_name, dir):
    image_data = read_data(case_idx, input_name, dir)
    return np.array(image_data)

def correct_bias(in_file, out_file):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image

def normalise(case_idx, input_name, in_dir, out_dir, copy = False):

	set_name = get_set_name(case_idx)
	image_in_path = get_filename(set_name, case_idx, input_name, in_dir)
	image_out_path = get_filename(set_name, case_idx, input_name, out_dir)
	if copy:
		shutil.copy(image_in_path, image_out_path)
	else:
		correct_bias(image_in_path, image_out_path)
	print(image_in_path + " done.")



"""
To extract patches from a 3D image
"""
def extract_imgpatches(volume, patch_shape, extraction_step,datype='float32'):
  
    patch_h, patch_w = patch_shape[0], patch_shape[1]
    stride_h, stride_w = extraction_step[0], extraction_step[1]
    img_h, img_w = volume.shape[0],volume.shape[1]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1

    N_patches_img = N_patches_h * N_patches_w
    raw_patch_martrix = np.zeros((N_patches_img, patch_h, patch_w, 3),dtype=datype)
    k=0
    #iterator over all the patches
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            raw_patch_martrix[k]=volume[h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w, :]
            k+=1
    assert(k==N_patches_img)
    return raw_patch_martrix

"""
To extract patches from a 3D label image
"""
def extract_patches(volume, patch_shape, extraction_step,datype='float32'):
 
    patch_h, patch_w = patch_shape[0], patch_shape[1]
    stride_h, stride_w = extraction_step[0], extraction_step[1]
    img_h, img_w = volume.shape[0],volume.shape[1]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1

    N_patches_img = N_patches_h * N_patches_w
    raw_patch_martrix = np.zeros((N_patches_img, patch_h, patch_w),dtype=datype)
    k=0
    #iterator over all the patches
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            raw_patch_martrix[k]=volume[h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w]
            k+=1
    assert(k==N_patches_img)
    return raw_patch_martrix

"""
To extract labeled patches from array of 3D labeled images
"""
def get_patches_lab(img_vols, label_vols, extraction_step, patch_shape, validating, testing, num_images_training):

    patch_shape_1d=patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, 3),dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d),dtype="uint8")
    for idx in tqdm(range(len(img_vols))) :
  
        y_length = len(y)
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step, datype="uint8")

        # Select only those who are important for processing
        if testing or validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2)) > 0)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d, 3), dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d),dtype="uint8")))

        y[y_length:, :, :] = label_patches

        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements

        img_train = extract_imgpatches(img_vols[idx], patch_shape, extraction_step, datype="float32")
        x[y_length:, :, :, :] = img_train[valid_idxs]

    return x, y

"""
To preprocess the labeled training data
"""
def preprocess_dynamic_lab(dir, num_classes, extraction_step, patch_shape, num_images_training=90, validating=False, testing=False, num_images_testing=7):
    if testing:
        print("Testing")
        r1 = 1
        r2 = num_images_testing + 1
        c = 0
        img_vols = np.empty((num_images_testing, 512, 512, 3), dtype = "float32")
        label_vols = np.empty((num_images_testing, 512, 512), dtype = "uint8")
    elif validating:
        print("Validating")
        r1 = num_images_training + 1
        r2 = num_images_training + 24 #23 Images for Validation
        c = num_images_training
        img_vols = np.empty(((r2 - r1), 512, 512, 3), dtype="float32")
        label_vols = np.empty(((r2 - r1), 512, 512), dtype="uint8")
    else:
        print("Training")
        r1 = 1
        r2 = num_images_training + 1
        c = 0
        img_vols = np.empty((num_images_training, 512, 512, 3),dtype="float32")
        label_vols = np.empty((num_images_training, 512, 512),dtype="uint8")

    for case_idx in range(r1, r2):
        print(case_idx)
        img_vols[(case_idx-c-1), :, :, :] = read_vol(case_idx, 'train', dir)
        label_vols[(case_idx-c-1), :, :] = read_vol(case_idx, 'label', dir)

    img_mean = img_vols.mean()
    img_std = img_vols.std()
    img_vols = (img_vols - img_mean) / img_std

    for i in tqdm(range(img_vols.shape[0])):
        img_vols[i] = ((img_vols[i] - np.min(img_vols[i])) / (np.max(img_vols[i])-np.min(img_vols[i]))) * 255

    img_vols = img_vols / 127.5 -1.
    label_vols = label_vols / 255.

    x,y = get_patches_lab(img_vols, label_vols, extraction_step, patch_shape, validating = validating, testing = testing, num_images_training = num_images_training)
    print("Total Extracted Labelled Patches Shape:", x.shape, y.shape)
    if testing:
        return x, label_vols
    elif validating:
        return x, y, label_vols
    else:
        return x, y


"""
To extract labeled patches from array of 3D ulabeled images
"""
def get_patches_unlab(img_vols, extraction_step, patch_shape, dir):

    patch_shape_1d = patch_shape[0]
    # Extract patches from input volumes and ground truth
    label_ref= np.empty((1, 512, 512),dtype="uint8")
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, 3))
    label_ref = read_vol(1, 'label', dir)
    for idx in tqdm(range(len(img_vols))) :

        x_length = len(x)
        label_patches = extract_patches(label_ref, patch_shape, extraction_step)

        # Select only those who are important for processing
        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2)) > 0)

        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, patch_shape_1d, 3))))

        img_train = extract_imgpatches(img_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :] = img_train[valid_idxs]

    return x

"""
To preprocess the unlabeled training data
"""
def preprocess_dynamic_unlab( dir,extraction_step, patch_shape, num_images_training, num_images_training_unlab):

    img_vols = np.empty((num_images_training_unlab, 512, 512, 3),dtype = "float32")
    r1 = num_images_training + 24 # 23 Validation Data
    r2 = num_images_training + 24 + num_images_training_unlab
    c = num_images_training + 23

    for case_idx in range(r1, r2):
        img_vols[(case_idx-c-1), :, :, :] = read_vol(case_idx, 'train', dir)

    img_mean = img_vols.mean()
    img_std = img_vols.std()
    img_vols = (img_vols - img_mean) / img_std

    for i in range(img_vols.shape[0]):
        img_vols[i] = ((img_vols[i] - np.min(img_vols[i])) / (np.max(img_vols[i])-np.min(img_vols[i])))*255

    img_vols = img_vols/127.5 -1.

    x = get_patches_unlab(img_vols, extraction_step, patch_shape,dir)
    print("Total Extracted Unlabelled Patches Shape:",x.shape)
    return x

"""
dataset class for preparing training data of basic U-Net
"""
class dataset(object):
    def __init__(self,num_classes, extraction_step, number_images_training, batch_size, patch_shape,data_directory):
        # Extract labelled and unlabelled patches
        self.batch_size=batch_size
        self.data_lab, self.label = preprocess_dynamic_lab(data_directory,num_classes,extraction_step, patch_shape,number_images_training)

        self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state = 2020)
        print("Data_shape:",self.data_lab.shape)
        print("Data lab max and min:",np.max(self.data_lab),np.min(self.data_lab))
        print("Label unique:",np.unique(self.label))

    def batch_train(self):
        self.num_batches = len(self.data_lab) // self.batch_size
        for i in range(self.num_batches):
            yield self.data_lab[i*self.batch_size:(i+1)*self.batch_size], self.label[i*self.batch_size:(i+1)*self.batch_size]


"""
dataset_badGAN class for preparing data of our model
"""
class dataset_badGAN(object):
    def __init__(self,num_classes, extraction_step, number_images_training, batch_size, patch_shape, number_unlab_images_training,data_directory):
        # Extract labelled and unlabelled patches,
        self.batch_size=batch_size
        self.data_lab, self.label = preprocess_dynamic_lab(data_directory, num_classes, extraction_step, patch_shape, number_images_training)

        self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state = 2020)
        self.data_unlab = preprocess_dynamic_unlab(data_directory, extraction_step, patch_shape, number_images_training, number_unlab_images_training)
        self.data_unlab = shuffle(self.data_unlab, random_state = 2020)

        # If training, repeat labelled data to make its size equal to unlabelled data
        factor = len(self.data_unlab) // len(self.data_lab)
        print("Factor for labeled images:",factor)
        rem = len(self.data_unlab)%len(self.data_lab)
        temp = self.data_lab[:rem]
        self.data_lab = np.concatenate((np.repeat(self.data_lab, factor, axis = 0), temp), axis=0)
        temp = self.label[:rem]
        self.label = np.concatenate((np.repeat(self.label, factor, axis=0), temp), axis=0)
        assert(self.data_lab.shape == self.data_unlab.shape)
        print("Data_shape:",self.data_lab.shape,self.data_unlab.shape)
        print("Data lab max and min:",np.max(self.data_lab),np.min(self.data_lab))
        print("Data unlab max and min:",np.max(self.data_unlab),np.min(self.data_unlab))
        print("Label unique:",np.unique(self.label))

    def batch_train(self):
        self.num_batches = len(self.data_lab) // self.batch_size
        for i in range(self.num_batches):
            yield self.data_lab[i*self.batch_size:(i+1)*self.batch_size], self.data_unlab[i*self.batch_size:(i+1)*self.batch_size], self.label[i*self.batch_size:(i+1)*self.batch_size]
