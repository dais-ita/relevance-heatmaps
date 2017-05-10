'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin 
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

#Code modified by Daniel Harborne

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render


import tensorflow as tf
import numpy as np
import pdb
import scipy.io as sio

from MNIST_Data import MnistData

import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 1,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 200,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 10,'Number of steps to run trainer.')
flags.DEFINE_integer("image_dim", 28,'Width of square input images')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_convolutional_logs','Summaries directory')
flags.DEFINE_boolean("relevance", True,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", True,'Restore the trained model')
#flags.DEFINE_string("checkpoint_dir", 'mnist_convolution_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_dir", 'mnist_heat_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", 'mnist_heat_model','Checkpoint dir')

FLAGS = flags.FLAGS


def nn():
    
    return Sequential([Convolution(output_depth=10,input_depth=1,batch_size=FLAGS.batch_size, input_dim=FLAGS.image_dim, act ='relu', stride_size=1, pad='VALID'),
                       AvgPool(),

                       Convolution(output_depth=25,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=4,output_depth=100,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),
                       Softmax()])


def feed_dict(tfl_data, train):
    if train:
        xs, ys = tfl_data.NextTrainBatch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = tfl_data.NextTestBatch(FLAGS.batch_size)
        k = 1.0


    # print(type(xs))
    # print(xs.shape)
    # print(type(xs[0,0]))

    # print(type(ys))
    # print(ys.shape)
    
    #return (2*xs)-1, ys, k
    return xs, ys, k


def train():
  # Import data
  # train_file_path = str(FLAGS.image_dim)+"_train_y.csv"
  # test_file_path = str(FLAGS.image_dim)+"_test_y.csv"

  # mnist = TFLData( (train_file_path,test_file_path) )

  train_file_path = os.path.join("mnist_csvs","mnist_train.csv")
  test_file_path = os.path.join("mnist_csvs","mnist_test.csv")

  mnist = MnistData( (train_file_path,test_file_path,(1000,1000)) )

  

  config = tf.ConfigProto(allow_soft_placement = True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:

    #with tf.Session() as sess:
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.image_dim*FLAGS.image_dim], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
    
    with tf.variable_scope('model'):
        net = nn()
        inp = tf.pad(tf.reshape(x, [FLAGS.batch_size,FLAGS.image_dim,FLAGS.image_dim,1]), [[0,0],[2,2],[2,2],[0,0]])
        op = net.forward(inp)
        y = tf.squeeze(op)
        
        trainer = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])
    with tf.variable_scope('relevance'):
        if FLAGS.relevance:
            LRP = net.lrp(op, FLAGS.relevance_method, 1e-8)

            # LRP layerwise 
            relevance_layerwise = []
            # R = y
            # for layer in net.modules[::-1]:
            #     R = net.lrp_layerwise(layer, R, 'simple')
            #     relevance_layerwise.append(R)

        else:
            LRP=[]
            relevance_layerwise = []
            
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()
    
    utils = Utils(sess, FLAGS.checkpoint_reload_dir)
    if FLAGS.reload_model:
        utils.reload_model()

    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:  # test-set accuracy
            d = feed_dict(mnist, False)
            test_inp = {x:d[0], y_: d[1], keep_prob: d[2]}
            #pdb.set_trace()
            summary, acc , relevance_test, rel_layer= sess.run([merged, accuracy, LRP, relevance_layerwise], feed_dict=test_inp)
            
            print_y = tf.argmax(y,1)
            y_labels = print_y.eval(feed_dict=test_inp)
           

            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
            # print([np.sum(rel) for rel in rel_layer])
            # print(np.sum(relevance_test))
            
            # save model if required
            if FLAGS.save_model:
                utils.save_model()

        else:  
            d = feed_dict(mnist, True)
            inp = {x:d[0], y_: d[1], keep_prob: d[2]}
            summary, _ , relevance_train,op, rel_layer= sess.run([merged, trainer.train, LRP,y, relevance_layerwise], feed_dict=inp)
            train_writer.add_summary(summary, i)
            
            
    # relevances plotted with visually pleasing color schemes
    if FLAGS.relevance:
        #pdb.set_trace()
        relevance_test = relevance_test[:,2:FLAGS.image_dim+2,2:FLAGS.image_dim+2,:]
        # plot test images with relevances overlaid
        images = test_inp[test_inp.keys()[0]].reshape([FLAGS.batch_size,FLAGS.image_dim,FLAGS.image_dim,1])
        #images = (images + 1)/2.0
        plot_relevances(relevance_test.reshape([FLAGS.batch_size,FLAGS.image_dim,FLAGS.image_dim,1]), images, test_writer, y_labels )
        
        # plot train images with relevances overlaid
        # relevance_train = relevance_train[:,2:30,2:30,:]
        # images = inp[inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
        # plot_relevances(relevance_train.reshape([FLAGS.batch_size,28,28,1]), images, train_writer )


    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
