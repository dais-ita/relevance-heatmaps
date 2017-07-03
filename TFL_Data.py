from DanTFDeepConvNet.DanTF_DeepNet import TFReadyData
import tensorflow as tf
import cv2
import numpy as np

import os

import random

import matplotlib.pyplot as plt


class TFLData(TFReadyData):

	def __init__(self,data):
		#pass tuple in the form (train_labels_file_path,test_labels_file_path)
		self.data=data

		self.train_labels = []
		self.test_labels = []

		self.train_image_paths = []
		self.test_image_paths = []

		self.y_key = {"congestion":0,"no-congestion":1}

		self.SetUpImages()

	def SetUpImages(self):
		num_categories = len(self.y_key)

		train_label_lines = [line.rstrip('\n') for line in open(self.data[0], "r")]
		for train_label_line in train_label_lines:
			annotation_data = train_label_line.rstrip().split(",")
			y_labels = [0] * num_categories
			y_labels[self.y_key[annotation_data[1]]] = 1
			self.train_labels.append(y_labels)

			self.train_image_paths.append(annotation_data[0])

		test_label_lines = [line.rstrip('\n') for line in open(self.data[1], "r")]
		for test_label_line in test_label_lines:
			annotation_data = test_label_line.rstrip().split(",")
			y_labels = [0] * num_categories
			y_labels[self.y_key[annotation_data[1]]] = 1
			self.test_labels.append(y_labels)

			self.test_image_paths.append(annotation_data[0])



	def NextTrainBatch(self,batch_size):
		#should return x and y data for next batch of training data
		# (np.array( [ [[Xi]] ] ),np.array([ [Yi] ]) )
		
		batch_image_index = random.sample(list(range(0,len(self.train_image_paths))), batch_size)

		y_values = [self.train_labels[i] for i in batch_image_index]
		image_paths = [self.train_image_paths[i] for i in batch_image_index]

		images = self.OpenPathsAsArray(image_paths)

		return (images, np.array(y_values,dtype='float32')) 
		


	def NextTestBatch(self,batch_size):
		#should return x and y data for next batch of test data
		# (np.array( [ [[Xi]] ] ),np.array([ [Yi] ]) )
		
		batch_image_index = random.sample(list(range(0,len(self.test_image_paths))), batch_size)

		y_values = [self.test_labels[i] for i in batch_image_index]
		image_paths = [self.test_image_paths[i] for i in batch_image_index]
		self.OutputPathListToFile(image_paths)
		images = self.OpenPathsAsArray(image_paths)
		
		return (images, np.array(y_values)) 

	def OpenPathsAsArray(self,image_paths):

		images = []
		for image_path in image_paths:
			path = os.path.join( *image_path.split("\\") )
			images.append( cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE).flatten() )
			
		return np.array( images,dtype='float32')


	def OutputPathListToFile(self,paths,output_file="last_batch.csv"):
		with open(output_file, "w") as output_file:
			for path in paths:
				output_file.write(path + "\n")


if __name__ == '__main__':

	train_file_path = "112_train_y.csv"
	test_file_path = "112_test_y.csv"

	tfl_data = TFLData( (train_file_path,test_file_path) )

	train_data = tfl_data.NextTrainBatch(10)
	print(train_data[0].shape)
	print(train_data[1])
	show_image = False
	if(show_image):
		cv2.imshow('image',train_data[0][0])
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	
	