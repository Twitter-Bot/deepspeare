import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math
import time
from collections import Counter

print("it is working")
#lm_dec_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(600, use_peepholes=True, forget_bias=1.0)
#lm_dec_cell = tf.keras.experimental.PeepholeLSTMCell(600)
#lm_dec_cell = tfa.rnn.PeepholeLSTMCell(600)
#print("working",lm_dec_cell,[lm_dec_cell] * 2)
# x=np.array([[[0, 0, 0], [0, 0, 0], [4, 0, 0]],
#  [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#  [[0, 0, 0], [0, 0, 0], [0, 0, 7]]])
# #print(x)
# x = x[:,-1,:]
# print(x)
# word_embedding = tf.compat.v1.get_variable("word_embedding", [10, 100],
#     initializer=tf.random_uniform_initializer(-0.05/100, 0.05/100))
# print('embeddings',word_embedding)
# rnn = tf.compat.v1.nn.dynamic_rnn(lm_dec_cell)
#     #update the cell. if keep_prob is at 1 then don't need a dropout layer
# lm_dec_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lm_dec_cell, output_keep_prob=0.7)
# #create RNN of LSTM cells
# lm_dec_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lm_dec_cell] * 1)

# attend_b = tf.compat.v1.get_variable("attend_b", [10], initializer=tf.constant_initializer())
# print("printing here",attend_b[0])


# attend_v = tf.compat.v1.get_variable("attend_v", [10, 1])
# print(attend_v)

# file = open('file_path', 'w')
# file.write('hello world !')
# file.close()

# # 2) without using with statement
# file = open('file_path', 'w')
# try:
#     file.writes('hello world')
# finally:
#     file.close()

# def selective_encoding(x):
#   y="hello"
#   print(x)

# with tf.compat.v1.variable_scope("selective_encoding"):
#   print(y)


class Name():
  def selective_encoding(self,x):
    print(x)
  def anotherfxn(self,y):
    x="hello"
    print(x)
    with tf.compat.v1.variable_scope("changeX"):
      z=x+"zzzz"
      print(x)
    print(z.name)


newClassTest=Name()
newClassTest.anotherfxn("hlala")


# hist_outputs_t = tf.tile([5,5,5], [1])

# tiledTensor=tf.tile(hist_outputs_t,[2])
# print(tiledTensor)

# a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
# b = tf.constant([2,1], tf.int32)
# print(tf.tile(a, b))
#output should be
# tf.Tensor(
# [[1 2 3]
#  [4 5 6]
#  [1 2 3]
#  [4 5 6]], shape=(4, 3), dtype=int32)
