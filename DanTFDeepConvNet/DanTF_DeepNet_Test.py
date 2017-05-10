from DanTF_DeepNet import DanTFDeepNet
from DanTF_DeepNet import MnistData
from DanTF_DeepNet import NetLayer
from DanTF_DeepNet import LayerType

from tensorflow.examples.tutorials.mnist import input_data

layers = []

layers.append( NetLayer(layer_type=LayerType.reshape, shape=[-1,28,28,1]) )

layers.append( NetLayer(layer_type=LayerType.conv, shape=[5,5,1,32], strides=[1, 1, 1, 1]) )
layers.append( NetLayer(layer_type=LayerType.maxpool, strides=[1, 2, 2, 1], k_size=[1, 2, 2, 1]) )

layers.append( NetLayer(layer_type=LayerType.conv, shape=[5,5,32,64],strides=[1, 1, 1, 1])  )
layers.append( NetLayer(layer_type=LayerType.maxpool, strides=[1, 2, 2, 1], k_size=[1, 2, 2, 1]) )

layers.append( NetLayer(layer_type=LayerType.reshape, shape=[-1,7*7*64]) )

layers.append( NetLayer(layer_type=LayerType.fully_connected, shape=[7*7*64,1024]) )

layers.append( NetLayer(layer_type=LayerType.dropout) )

layers.append( NetLayer(layer_type=LayerType.fully_connected, shape=[1024,10]) )


mnist = input_data.read_data_sets("mnist/", one_hot=True)
mnist_data = MnistData(mnist)

tf_net = DanTFDeepNet(layers, mnist_data)

print("Starting Session")
print("")
tf_net.StartSession()


epochs=20
n_batches=500
batch_size=50


tf_net.Train(epochs,n_batches,batch_size)

# print(tf_net.Test())

# tf_net.SaveSess("mnist-C-MP-C-MP-FCL-FCL"+"-"+str(epochs)+"-"+str(n_batches)+"-"+str(batch_size))

#tf_net.LoadSess("mnist-C-MP-C-MP-FCL-FCL")

run_x, run_y = tf_net.data.NextTrainBatch(1)

print("Prediction:") 
print(tf_net.sess.run(tf_net.Run(run_x))[0])
print("Actual:")
print(list(run_y[0]).index(1))


print(len(tf_net.y_val_list))

for i,element in enumerate(tf_net.y_val_list):
	print(element)
	print(type(element))

	if(i > 0 and i < len(tf_net.y_val_list)-1):
		print(tf_net.y_val_list[i].eval(session=tf_net.sess))

for key,value in tf_net.W_dict.items():
	print("Key: " + str(key))
	print(value.eval(session=tf_net.sess).shape)	