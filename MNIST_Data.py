from DanTFDeepConvNet.DanTF_DeepNet import LayerType,NetLayer,TFReadyData,DanTFDeepNet
import tensorflow as tf
import cv2
import numpy as np

import os

import random

import matplotlib.pyplot as plt


class MnistData(TFReadyData):

	def __init__(self,data,show_sample=False):
		#pass tuple in the form (train_labels_file_path,test_labels_file_path,OPTIONAL_(num_train,num_test))
		self.data=data

		self.train_labels = []
		self.test_labels = []

		self.train_images = []
		self.test_images = []

		self.y_key = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9}

		self.SetUpImages()
		if(show_sample):
			print("Showing sample...")
			self.ShowSampleImage()

	def SetUpImages(self):
		num_categories = len(self.y_key)

		load_limits = None
		if(len(self.data) > 2):
			load_limits = self.data[2]
		print("Loading Train Images")
		train_label_lines = [line.rstrip('\n') for line in open(self.data[0], "r")]
		
		total_train = len(train_label_lines)
		load_train = total_train
		if(load_limits != None):
			load_train = min(load_limits[0],total_train)
			print("Train image limit:"+ str(load_train))

		print("Train Images Found: "+str(total_train))
		
		for train_label_line_index in range(0,load_train):
			train_label_line = train_label_lines[train_label_line_index]
			annotation_data = train_label_line.rstrip().split(",")
			
			label = annotation_data.pop(0)
			y_labels = [0] * num_categories
			y_labels[self.y_key[label]] = 1
			self.train_labels.append(y_labels)

			annotation_data = [int(data) for data in annotation_data]
			self.train_images.append(np.array(annotation_data, dtype='float32')/255) 
			

		print("Loading Test Images")
		test_label_lines = [line.rstrip('\n') for line in open(self.data[1], "r")]

		total_test = len(test_label_lines)
		load_test = total_test
		if(load_limits):
			load_test = min(load_limits[1],total_test)
			print("Train image limit:"+ str(load_test))
		print("Test Images Found: "+str(total_test))
		
		for test_label_line_index in range(0,load_test):
			test_label_line = test_label_lines[test_label_line_index]
			annotation_data = test_label_line.rstrip().split(",")
			
			label = annotation_data.pop(0)
			y_labels = [0] * num_categories
			y_labels[self.y_key[label]] = 1
			self.test_labels.append(y_labels)

			annotation_data = [int(data) for data in annotation_data]
			self.test_images.append(np.array(annotation_data, dtype='float32')/255) 



	def NextTrainBatch(self,batch_size):
		#should return x and y data for next batch of training data
		# (np.array( [ [[Xi]] ] ),np.array([ [Yi] ]) )
		
		batch_image_index = random.sample(list(range(0,len(self.train_images))), batch_size)

		y_values = [self.train_labels[i] for i in batch_image_index]
		images = np.array([self.train_images[i] for i in batch_image_index])

		return (images, np.array(y_values)) 
		


	def NextTestBatch(self,batch_size):
		#should return x and y data for next batch of test data
		# (np.array( [ [[Xi]] ] ),np.array([ [Yi] ]) )
		
		batch_image_index = random.sample(list(range(0,len(self.test_images))), batch_size)

		y_values = [self.test_labels[i] for i in batch_image_index]
		images = np.array([self.test_images[i] for i in batch_image_index])
		self.OutputPathListToFile(batch_image_index)
		
		return (images, np.array(y_values)) 


	def OutputPathListToFile(self,paths,output_file="last_batch.csv"):
		with open(output_file, "w") as output_file:
			for path in paths:
				output_file.write(str(path) + "\n")

	def ShowSampleImage(self):
		train_data = self.NextTrainBatch(10)
		cv2.imshow('image',train_data[0][0].reshape(28,28))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':

	train_file_path = os.path.join("mnist_csvs","mnist_train.csv")
	test_file_path = os.path.join("mnist_csvs","mnist_test.csv")

	mnist = MnistData( (train_file_path,test_file_path,(100,100)),True )

	train_data = mnist.NextTrainBatch(10)
	print(train_data[0].shape)
	print(train_data[1])
	print(train_data[0][0].reshape(28,28))
	show_image = False
	if(show_image):
		for i in range(10):
			print(train_data[1][i])
			cv2.imshow('image',train_data[0][i].reshape(28,28))
			cv2.waitKey(0)
			cv2.destroyAllWindows()
	
	
	