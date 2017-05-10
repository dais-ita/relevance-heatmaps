import tensorflow as tf
import os

from enum import Enum

class LayerType(Enum):
	conv = 1
	maxpool = 2
	reshape = 3
	fully_connected = 4
	dropout = 5
	resize = 6

class NetLayer():
	def __init__(self,layer_type,shape=[],strides=[],k_size=[],dropout = 0.5):
		self.layer_type = layer_type
		self.shape = shape
		self.strides = strides
		self.k_size = k_size
		self.dropout = dropout

class TFReadyData():

	def __init__(self,data):
		#should store some reference to the data (TF reference, file path etc. This will depend on the intended access methods)
		self.data=data
		

	def NextTrainBatch(self,batch_size):
		#should return x and y data for next batch of training data
		pass

	def NextTestBatch(self):
		#should return x and y data for next batch of test data
		pass


class MnistData(TFReadyData):
	def __init__(self,data):
		self.data=data
		

	def NextTrainBatch(self,batch_size):
		return self.data.train.next_batch(batch_size)


	def NextTestBatch(self):

		return self.data.test.images, self.data.test.labels


class DanTFDeepNet():

	def __init__(self, layer_list, tf_ready_data):
		self.layers = layer_list

		self.X,self.Y_,self.W_dict,self.B_dict,self.y_list,self.dropout_dict = self.CreateTFVariables(layer_list)
		self.data = tf_ready_data

		self.saver = tf.train.Saver()


	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)


	def bias_variable(self,shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)


	def ProcessLayer(self,W_dict,B_dict,y_list,dropout_dict,layer,layer_index,inc_relu_step=False):
		if layer.layer_type is LayerType.conv:
			print("Conv Layer")
			print("")

			W_dict[layer_index] =  self.weight_variable(layer.shape) 
			B_dict[layer_index] =  self.bias_variable( [layer.shape[-1]] ) 

			y_list.append(  tf.nn.conv2d(y_list[-1], W_dict[layer_index], layer.strides, padding='SAME') + B_dict[layer_index])
			
			if(inc_relu_step):
				tf.nn.relu(y_list[-1])

		
		elif  layer.layer_type is LayerType.maxpool:
			print("MaxPool Layer")
			print("")

			y_list.append( tf.nn.max_pool(y_list[-1], layer.k_size, layer.strides, padding='SAME') )


		elif  layer.layer_type is LayerType.reshape:
			print("Reshape Layer")
			print("")

			y_list.append( tf.reshape(y_list[-1], layer.shape) )

		
		elif  layer.layer_type is LayerType.fully_connected:
			print("Fully Connected Layer")
			print("")

			W_dict[layer_index] =  self.weight_variable(layer.shape)
			B_dict[layer_index] =  self.bias_variable( [layer.shape[-1]] )

			y_list.append( tf.matmul(y_list[-1], W_dict[layer_index]) + B_dict[layer_index])

			if(inc_relu_step):
				tf.nn.relu(y_list[-1])
		
		elif  layer.layer_type is LayerType.dropout:
			print("Dropout Layer")
			print("")

			dropout_dict[layer_index] = tf.constant(layer.dropout)
			y_list.append( tf.nn.dropout(y_list[-1], dropout_dict[layer_index]) )

		elif  layer.layer_type is LayerType.resize:
			print("Resize Layer")
			print("")

			y_list.append( tf.image.resize_images(y_list[-1], layer.shape) )
		
		else:
			print("Unhandled Layer Type")


	def CreateTFVariables(self,layers):
		X = tf.placeholder(tf.float32, [None, layers[0].shape[1]*layers[0].shape[2]])
		Y_ = tf.placeholder(tf.float32, [None, layers[-1].shape[-1]])

		W_dict = {}
		B_dict = {}

		dropout_dict = {}

		y_list = []

		y_list.append(X)

		for layer_index in range(len(layers)-1):
			print("Layer:"+str(layer_index))
			
			self.ProcessLayer(W_dict,B_dict,y_list,dropout_dict,layers[layer_index],layer_index,inc_relu_step=True)

		print("Layer:"+str(layer_index + 1)+" (Final Layer)")
		self.ProcessLayer(W_dict,B_dict,y_list,dropout_dict,layers[-1],layer_index+1,inc_relu_step=False)

		return X,Y_,W_dict,B_dict,y_list,dropout_dict


	def StartSession(self):
		self.sess = tf.Session()
		

	def CloseSession(self):
		self.sess.close()


	def Train(self,epochs,n_batches,batch_size):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_list[-1], self.Y_))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		tf.global_variables_initializer().run(session=self.sess)	
		
		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(n_batches):
				batch_xs, batch_ys = self.data.NextTrainBatch(batch_size)
				_,bloss =  self.sess.run([train_step,cross_entropy], feed_dict={self.X: batch_xs, self.Y_: batch_ys})
				epoch_loss += bloss
				
			print("Epoch "+str(epoch)+" Loss:")
			print(epoch_loss)


	def Test(self):
		
		correct_prediction = tf.equal(tf.argmax(self.y_list[-1], 1), tf.argmax(self.Y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		test_x,test_y = self.data.NextTestBatch()
		return self.sess.run( accuracy, feed_dict={self.X: test_x, self.Y_: test_y})

	def RunLayer(self,W_dict,B_dict,y_list,dropout_dict,layer,layer_index,inc_relu_step=False):
		if layer.layer_type is LayerType.conv:
			print("Conv Layer")
			print("")
			
			y_list.append(  tf.nn.conv2d(y_list[-1], W_dict[layer_index], layer.strides, padding='SAME') + B_dict[layer_index])
			
			if(inc_relu_step):
				tf.nn.relu(y_list[-1])

		
		elif  layer.layer_type is LayerType.maxpool:
			print("MaxPool Layer")
			print("")

			y_list.append( tf.nn.max_pool(y_list[-1], layer.k_size, layer.strides, padding='SAME') )


		elif  layer.layer_type is LayerType.reshape:
			print("Reshape Layer")
			print("")

			y_list.append( tf.reshape(y_list[-1], layer.shape) )

		
		elif  layer.layer_type is LayerType.fully_connected:
			print("Fully Connected Layer")
			print("")

			y_list.append( tf.matmul(y_list[-1], W_dict[layer_index]) + B_dict[layer_index])

			if(inc_relu_step):
				tf.nn.relu(y_list[-1])
		
		elif  layer.layer_type is LayerType.dropout:
			print("Dropout Layer")
			print("")

			dropout_dict[layer_index] = tf.constant(1.0)
			y_list.append( tf.nn.dropout(y_list[-1], dropout_dict[layer_index]) )

		elif  layer.layer_type is LayerType.resize:
			print("Resize Layer")
			print("")

			y_list.append( tf.image.resize_images(y_list[-1], layer.shape) )
		
		else:
			print("Unhandled Layer Type")

		
	
	def Run(self,x_input):
		self.y_val_list = []

		self.y_val_list.append(x_input)

		for layer_index in range(len(self.layers)-1):
			self.RunLayer(self.W_dict,self.B_dict,self.y_val_list,self.dropout_dict,self.layers[layer_index],layer_index,inc_relu_step=True)

		print("Layer:"+str(layer_index + 1)+" (Final Layer)")
		self.RunLayer(self.W_dict,self.B_dict,self.y_val_list,self.dropout_dict,self.layers[layer_index+1],layer_index+1,inc_relu_step=False)


		return tf.argmax(self.y_val_list[-1],1)


	def SaveSess(self,session_name):
		cwd = os.getcwd()
		sessions_dir = os.path.join(cwd,"sessions")
		session_dir = os.path.join(sessions_dir,session_name)
		
		if not os.path.exists(session_dir):
			os.makedirs(session_dir)

		save_path = os.path.join(session_dir,session_name+".ckpt")
		self.saver.save(self.sess, save_path)


	def LoadSess(self,session_name):
		cwd = os.getcwd()
		sessions_dir = os.path.join(cwd,"sessions")
		session_dir = os.path.join(sessions_dir,session_name)
		load_path = os.path.join(session_dir,session_name+".ckpt")
		
		if(not os.path.exists(session_dir)):
			print("Session checkpoint not found")
		else:
			self.saver.restore(self.sess, load_path)

			