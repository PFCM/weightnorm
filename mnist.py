# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

import weightnorm.weightnorm as weightnorm

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def layer(input_var, input_size, outputs, name, do_weightnorm=False,
          do_batchnorm=False, train=True):
  """Just do a single layer, return logits. Adds some summary nodes as well"""
  if do_weightnorm:
    weights, g, v = weightnorm.get_normed_weights([input_size, outputs],
                                                  scope=name,
                                                  return_all=True,
                                                  axis=0)
    if train:
      tf.histogram_summary(name+'unnormalised-weights', v)
      tf.histogram_summary(name+'scales', g)
  else:
    weights = tf.get_variable('w', [input_size, outputs])
  if train:
    tf.histogram_summary(name+'weights', weights)
  biases = tf.get_variable('b', [outputs])
  if do_batchnorm:
    return weightnorm.meanonly_batchnormalise(weights,
                                              input_var,
                                              biases,
                                              axis=0,
                                              train=train)
  return tf.matmul(input_var, weights) + biases


def inference(images, hidden_size, num_layers, do_weightnorm=False,
              do_batchnorm=False, train=True):
  """Build the MNIST model up to where it may be used for inference.

  If you want to use mean only batch normalisation, then you will need
  a separate model for training and evaluation.

  Args:
    images: Images placeholder, from inputs().
    hidden_size: Size of the first hidden layer.
    do_weightnorm: whether to use normalised weights.
    do_batchnorm: whether to do mean only batch normalisation.
    train: whether this model should be for training or evaluation.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hiddens
  last_out = images
  for i in range(num_layers):
    with tf.variable_scope('hidden{}'.format(i+1), reuse=not train):
      last_out = tf.nn.relu(layer(last_out,
                                  IMAGE_PIXELS if last_out is images else hidden_size,
                                  hidden_size,
                                  'hidden{}'.format(i+1),
                                  do_weightnorm,
                                  do_batchnorm,
                                  train=train))
  # Linear
  with tf.variable_scope('softmax_linear', reuse=not train):
    logits = layer(last_out, hidden_size, NUM_CLASSES, 'softmax', train=train)
  return logits


def loss(logits, labels, summary_name='xentropy_mean'):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name=summary_name)
  return loss


def training(loss, learning_rate, momentum=0.9):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer = tf.train.AdamOptimizer(learning_rate)
  #optimizer = tf.train.RMSPropOptimizer(learning_rate)
  optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
