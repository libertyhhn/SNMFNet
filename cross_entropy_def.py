
import tensorflow as tf
#import numpy as np
#import scipy
#import scipy.io as scio

def cross_entropy(y, y_pred, class1_weight, class2_weight):
    # class1_weight: 第一类的权重系数
    # class2_weight: 第二类的权重系数
    # y: True Labels

    cross_entropy = -class1_weight*(y[:,0]*tf.log(y_pred[:,0]))-class2_weight*(y[:,1]*tf.log(y_pred[:,1]))
    return cross_entropy
