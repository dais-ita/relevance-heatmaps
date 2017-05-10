from __future__ import division
import cv2 as cv
import os

def ResizeSquareAndSaveImage(image_path,target_width,output_path,pad = True, grayscale = True):
	image = cv.imread(image_path)
	
	image_height = image.shape[0]
	image_width = image.shape[1]

	max_length = max(image_height,image_width)
	resize_ratio = target_width/max_length

	new_height = int(resize_ratio*image_height)
	new_width = int(resize_ratio*image_width)

	new_image = cv.resize(image, (0,0), fx=new_width/image_width, fy=new_height/image_height) 

	if(pad):
		pad_y,y_remainder = divmod(target_width-new_height,2)
		pad_x,x_remainder = divmod(target_width-new_width,2)

		new_image= cv.copyMakeBorder(new_image,pad_y+y_remainder,pad_y,pad_x+x_remainder,pad_x,cv.BORDER_CONSTANT,value=[0,0,0])

	if(grayscale):
		new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)

	cv.imwrite(output_path,new_image)


if __name__ == '__main__':
	resize_length = 112

	input_dir = "input_images"
	output_top_dir = "resized_images_"+str(resize_length)
	image_type_dirs = [f for f in os.listdir(input_dir) if not f.startswith('.')]
	for type_dir in image_type_dirs:
		type_dir_path = os.path.join(input_dir,type_dir)

		camera_id_dirs = [f for f in os.listdir(type_dir_path) if not f.startswith('.')]

		for id_dir in camera_id_dirs:
			id_path = os.path.join(type_dir_path,id_dir)

			images = [f for f in os.listdir(id_path) if not f.startswith('.')]


			output_dir = os.path.join(output_top_dir,type_dir,id_dir)
			if(not os.path.exists(output_dir)):
				os.makedirs(output_dir)
			for image in images:
				image_path = os.path.join(id_path,image)
				output_path = os.path.join(output_dir,image)
				ResizeSquareAndSaveImage(image_path,resize_length,output_path)

	train_label_lines = [line.rstrip('\n') for line in open("28_train_y.csv", "r")]

	with open(str(resize_length)+"_train_y.csv","w") as output_train:
		for line in train_label_lines:
			output_train.write( line.replace("images_28","images_"+str(resize_length)) + "\n" )

	test_label_lines = [line.rstrip('\n') for line in open("28_test_y.csv", "r")]

	with open(str(resize_length)+"_test_y.csv","w") as output_test:
		for line in test_label_lines:
			output_test.write( line.replace("images_28","images_"+str(resize_length)) + "\n" )

