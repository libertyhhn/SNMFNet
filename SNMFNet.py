# import argparse
# import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

#import scipy
import scipy.io as scio
import math
#from math import e
import numpy as np
import cross_entropy_def
#import random
import time
#from tflearn.layers.normalization import batch_normalization
from train_next_batch import DataSet

BatchSize = int(200)
DimensionSize = int(100)
G1 = np.load('G1.npy')
G1 = tf.abs(G1) 

# 读入健康样本和不健康样本的MFCC特征
def InputExperimentData(Datai):
  path = '../../../DataCode/Rubin_Train_Test_Data/10fold_9/'
  PathFile_name = path+'trainsetF'+str(Datai)+'.mat'      # 带病的训练样本
  trainsetF = scio.loadmat(PathFile_name)
  trainsetF = trainsetF['trainsetF']
  PathFile_name = path+'trainsetT'+str(Datai)+'.mat'      # 健康的训练样本
  trainsetT = scio.loadmat(PathFile_name)
  trainsetT = trainsetT['trainsetT']
  print(trainsetF.shape,trainsetT.shape)

  PathFile_name = path+'testsetF'+str(Datai)+'.mat'       # 带病的测试样本
  testsetF = scio.loadmat(PathFile_name)
  testsetF = testsetF['testsetF']
  PathFile_name = path+'testsetT'+str(Datai)+'.mat'       # 健康的测试样本
  testsetT = scio.loadmat(PathFile_name)
  testsetT = testsetT['testsetT']
  print(testsetF.shape,testsetT.shape)

  PathFile_name = path+'TestSmpF_cnt'+str(Datai)+'.mat'   # 带病的测试样本的周期数
  TestSmpF_cnt = scio.loadmat(PathFile_name)
  TestSmpF_cnt = TestSmpF_cnt['TestSmpF_cnt']
  PathFile_name = path+'TestSmpT_cnt'+str(Datai)+'.mat'   # 健康的测试样本的周期数
  TestSmpT_cnt = scio.loadmat(PathFile_name)
  TestSmpT_cnt = TestSmpT_cnt['TestSmpT_cnt']
  print(TestSmpF_cnt.shape,TestSmpT_cnt.shape)
  
  # 不健康样本标签[0 1]
  trainF_label = np.zeros([trainsetF.shape[0],2], dtype = float)
  trainF_label[::,1] = 1.0
  # 健康样本标签[1 0]
  trainT_label = np.ones([trainsetT.shape[0],2], dtype = float)
  trainT_label[::,1] = 0.0
  print(trainF_label.shape, trainT_label.shape)

  # 训练集
  trainset = np.concatenate((trainsetF,trainsetT), axis=0)
  trainlabel = np.concatenate((trainF_label,trainT_label), axis=0)
  print(trainset.shape,trainlabel.shape)
  # 打乱样本顺序
  seed = Datai
  np.random.seed(seed)
  train_index = np.arange(trainset.shape[0])
  np.random.shuffle(train_index)
  trainset = np.array(trainset)[train_index]
  trainlabel = np.array(trainlabel)[train_index]
  print(trainset.shape,trainlabel.shape)

  #不健康测试样本标签[0 1]
  testF_label = np.zeros([testsetF.shape[0],2], dtype = float)
  testF_label[::,1] = 1.0
  #健康测试样本标签[1 0]
  testT_label = np.ones([testsetT.shape[0],2], dtype = float)
  testT_label[::,1] = 0.0
  print(testF_label.shape,testT_label.shape)

  # 测试集样本和标签
  testset = np.concatenate((testsetF, testsetT), axis=0)
  testlabel = np.concatenate((testF_label, testT_label), axis=0)
  print(testset.shape,testlabel.shape)

  return trainset,testset,trainlabel,testlabel,TestSmpF_cnt,TestSmpT_cnt

def deepnn(x,y_,keep_prob1):
  global G1

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 300, 6])
    x_image = tf.reshape(x_image, [-1, 1800])
    # x_image = batch_normalization(x_image)
    print(x_image)

  with tf.name_scope('SNMF1'):
    F1 = weight_variable([100,1800])/tf.sqrt(100.0)
    F11 = tf.transpose(F1)
    print(F1,F11)
    for _ in range(0,100):
      A1 = tf.matmul(x_image, F11)
      Ap1=(tf.add(tf.abs(A1),A1))/tf.constant(2.0)
      An1=(tf.subtract(tf.abs(A1),A1))/tf.constant(2.0)
      # print(A,Ap,An)

      B1= tf.matmul(F1, F11)
      Bp1=(tf.add(tf.abs(B1),B1))/tf.constant(2.0)
      Bn1=(tf.subtract(tf.abs(B1),B1))/tf.constant(2.0)

      G11 = tf.add(Ap1, tf.matmul(G1,Bn1))
      G12 = tf.add(An1, tf.matmul(G1,Bp1))
      # print(G11,G12)

      G1 = G1*(tf.sqrt(tf.divide(G11,G12)))

  with tf.name_scope('dropout'):
    G1_drop = tf.nn.dropout(G1, keep_prob1, seed=2)

  # Map the 200 features to 2 classes, one for each digit
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([100, 256])/tf.sqrt(50.0)
    b_fc1 = bias_variable([256])
    o_fc1 = tf.nn.relu(tf.matmul(G1_drop, W_fc1) + b_fc1)
    o_fc1_drop = tf.nn.dropout(o_fc1, keep_prob1, seed=2)
  with tf.name_scope('out'):
    W_out = weight_variable([256, 2])/tf.sqrt(50.0)
    b_out = bias_variable([2])
    y_conv = tf.nn.softmax(tf.matmul(o_fc1_drop, W_out) + b_out)
      
  with tf.name_scope('loss'):
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv1)
    cross_entropy = cross_entropy_def.cross_entropy(y_, y_conv,1.0,1.0)
    cross_entropy = tf.reduce_mean(cross_entropy)

    matrix_loss = tf.norm(x_image-tf.matmul(G1,F1), ord=2)
    matrix_loss = tf.reduce_mean(matrix_loss)
    total_loss = 1.0*matrix_loss+1.0*cross_entropy
  return y_conv, total_loss


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, ksize, strides):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize, strides, padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1, seed=1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def errweight_variable(shape,minv,maxv):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.random_uniform(shape=shape,minval=minv,maxval=maxv,dtype=tf.float32)
    return tf.Variable(initial)


def main(_):

  # Create the model
  x = tf.placeholder(tf.float32, [None, 300, 6], name='x')
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])
  keep_prob1 = tf.placeholder(tf.float32)
  # Build the graph for the deep net
  y_conv, total_loss = deepnn(x,y_,keep_prob1)

  with tf.name_scope('adam_optimizer'):
    learn_rate = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

#  graph_location = 'logs/train'
#  print('Saving graph to: %s' % graph_location)
#  train_writer = tf.summary.FileWriter(graph_location)
#  train_writer.add_graph(tf.get_default_graph())  

  global G1
  G0 = G1
  for Datai in range(0,10):
    G1 = G0
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # 读取心音信号样本(MFCC特征)
      trainset,testset,trainlabel,testlabel,TestSmpF_cnt,TestSmpT_cnt = InputExperimentData(Datai)
      trainset_size = trainset.shape
      testset_size = testset.shape
      print(trainset_size,trainlabel.shape,testset_size,testlabel.shape)

      filename = "TestSmpF_cnt%01d.npy" % Datai
      np.save(filename, TestSmpF_cnt)
      filename = "TestSmpT_cnt%01d.npy" % Datai
      np.save(filename,TestSmpT_cnt)

      ds = DataSet(trainset, trainlabel, trainset_size[0])

      Train_K = int(math.ceil(trainset_size[0]/float(BatchSize)))
      Test_K1 = testset_size[0]//BatchSize
      Test_K2 = testset_size[0]%BatchSize
      print(Train_K, Test_K1, Test_K2)
      iter_cnt = 100

      totalloss = np.zeros([iter_cnt*Train_K,1],dtype = float)
      ii = int(0)

      Model_PathName = 'ckpt'+str(Datai)+'/model.ckpt'
      saver=tf.train.Saver(max_to_keep=1)
      for epoch in range(0,iter_cnt):
        for _ in range(0,Train_K):
          train_batch, label_batch = ds.next_batch(BatchSize,epoch)
          [o,c]=sess.run([train_step,total_loss],feed_dict={x: train_batch, y_: label_batch, learn_rate: 0.0005, keep_prob1: 0.75})
          totalloss[ii, 0] = c
          SF1_val[ii, 0] = SF1
          SG1_val[ii, 0] = SG1
        print('ExperimentNo.:%d'%(Datai+1),'--Epoches:%d'%iter_cnt,'--CurrentEpoch:%d'%(epoch+1))
      saver.save(sess, Model_PathName)

      filename = "totalloss%01d.mat" % Datai
      scio.savemat(filename, {'totalloss':totalloss})
      # 网络预测
      G1 = G0
      testset_predict = y_conv.eval(feed_dict={x: testset[:BatchSize], y_: testlabel[:BatchSize], keep_prob1: 1.0})    # make prediction
      print(testset_predict.shape)
      for i in range(1,Test_K1):
        G1 = G0
        temp_predict = y_conv.eval(feed_dict={x: testset[i*BatchSize:(i+1)*BatchSize], y_: testlabel[i*BatchSize:(i+1)*BatchSize], keep_prob1: 1.0})   # make prediction
        testset_predict = np.concatenate((testset_predict, temp_predict), axis=0)
      print(testset_predict.shape)
      if(Test_K2!=0):
        G1 = G0
        temp_predict = y_conv.eval(feed_dict={x: testset[(testset_size[0]-BatchSize)::], y_: testlabel[(testset_size[0]-BatchSize)::], keep_prob1: 1.0})   # make prediction
        testset_predict = np.concatenate((testset_predict, temp_predict[(BatchSize-Test_K2)::]), axis=0)
      print(testset_predict.shape)

      filename = "testset_predict%01d.npy" % Datai
      np.save(filename, testset_predict)
      filename = "testset_predict%01d.mat" % Datai
      scio.savemat(filename,{'testset_predict':testset_predict})

time_start=time.time()
if __name__ == '__main__':
    tf.app.run(main=main)
time_end=time.time()
print('training_time',time_end-time_start)

