
'''
Script to organise images into appropriate directories
'''

import os
import math 
from shutil import copyfile 




image_y = [line.rstrip('\n') for line in open("y_values.csv", "r")]

image_dict = {}

for line in image_y:
	line_split = line.split(",")

	path_split = line_split[0].split("/")

	camera_id = path_split[0]
	filename = path_split[1]

	if(not camera_id in image_dict):
		image_dict[camera_id] = [ [], [] ]

	if(line_split[1] == "congestion"):
		image_dict[camera_id][0].append(filename)
	else:
		image_dict[camera_id][1].append(filename)

train_path = os.path.join("input_images","train")
test_path = os.path.join("input_images","test")

train_y_values = []
test_y_values = []

for key,value in image_dict.items():
	output_train_path = os.path.join(train_path,key)
	output_test_path = os.path.join(test_path,key)
	
	if not os.path.exists(output_train_path):
		os.makedirs(output_train_path)	 

	if not os.path.exists(output_test_path):
		os.makedirs(output_test_path)	 

	camera_path = os.path.join("annotated_images",key)
	

	congestion_train_divide_index = math.floor(len(value[0]) * 0.9)

	train_congestion_images = value[0][0:congestion_train_divide_index]
	test_congestion_images = value[0][congestion_train_divide_index:]

	for train_congestion_image in train_congestion_images:
		image_path = os.path.join(camera_path,train_congestion_image)
		output_path = os.path.join(output_train_path,train_congestion_image)
		copyfile(image_path, output_path)
		train_y_values.append( (output_path,"congestion") )

	for test_congestion_image in test_congestion_images:
		image_path = os.path.join(camera_path,test_congestion_image)
		output_path = os.path.join(output_test_path,test_congestion_image)
		copyfile(image_path, output_path)
		test_y_values.append( (output_path,"congestion") )

	no_congestion_train_divide_index = math.floor(len(value[1]) * 0.9)

	train_no_congestion_images = value[0][0:no_congestion_train_divide_index]
	test_no_congestion_images = value[0][no_congestion_train_divide_index:]

	for train_no_congestion_image in train_no_congestion_images:
		image_path = os.path.join(camera_path,train_no_congestion_image)
		output_path = os.path.join(output_train_path,train_no_congestion_image)
		copyfile(image_path, output_path)
		train_y_values.append( (output_path,"no-congestion") )

	for test_no_congestion_image in test_no_congestion_images:
		image_path = os.path.join(camera_path,test_no_congestion_image)
		output_path = os.path.join(output_test_path,test_no_congestion_image)
		copyfile(image_path, output_path)
		test_y_values.append( (output_path,"no-congestion") )


with open("train_y.csv", "w") as train_output_file:
	for train_y in train_y_values:
		train_output_file.write( str(train_y[0]) + "," + str(train_y[1]) + "\n")

with open("test_y.csv", "w") as test_output_file:
	for test_y in test_y_values:
		test_output_file.write( str(test_y[0]) + "," + str(test_y[1]) + "\n")




