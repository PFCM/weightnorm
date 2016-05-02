"""Weight normalisation, provides a helper method for getting normalised
weight matrices. 

Variables that actually need to get trained are added to the Tensorflow
trainable variable list which means they should be grabbed by the optimiser.

Provides functions:
  - get_normed_weights: returns a tensor that can be used as weights, except
      when it is trained it should remain normalised.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# TODO (pfcm): how does this work with a 3-tensor
def get_normed_weights(shape, axis=None, scope=None, return_all=True,
                       reuse=False,
                       init=tf.random_normal_initializer(stddev=0.05)):
    """
    Returns a normalised tensor of the given shape.

    Args:
      shape: the shape of the desired weights. At the moment we assume
        this is [num_inputs x num_outputs] and we have a gain/scale per
        output.
      axis: the axis or axes over which to normalise. If None (default), then
        each element is divided by the norm of the entire tensor.
      scope: scope in which to get the variables required. Defaults to None,
        which means `weightnorm` will be used.
      return_all: if true, returns the allocated trainable variable as well as
        the resulting weights.
      reuse: whether or not to attempt to reuse variables. Default is False.
      init: the initializer to use to initialise the variables. Defaults to the
        values from the paper, ie. normally distributed with mean 0 and
        standard deviation 0.05.

    Returns:
      - if `return_all` is true it will return `(w, g, v)` where `w` is the
          required weights, `g` and `v` are the scale and the unnormalised
          weights respectively.
      - otherwise, just return `w`.
    """
    with tf.variable_scope(scope or 'weightnorm', reuse=reuse,
                           initializer=init):
        v = tf.get_variable('v', shape=shape)
        g = tf.get_variable('g', shape=shape[-1], initializer=tf.constant_initializer(1),
                            trainable=False)
        inv_norm = tf.rsqrt(tf.reduce_sum(tf.square(v), reduction_indices=axis))
        w = v * g * inv_norm
        #w = g * tf.nn.l2_normalize(v, 1)
        if return_all:
            return w, g, v
        return w


def meanonly_batchnormalise(weights, inputs, biases, axis=0, train=True):
    """Does mean-only batch normalisation. Returns something you can use for 
    your activations.
    """
    t = tf.matmul(inputs, weights)
    running_average = tf.get_variable('running_average', t.get_shape())
    mean = tf.reduce_mean(t, reduction_indices=axis)
    assign_average = running_average.assign((running_average + mean)/2)
    if train:
        with tf.control_dependencies([assign_average]):
            t_tilde = t - mean + biases
    else:
        t_tilde = t - running_average + biases
        
    return t_tilde

def full_batchnorm(pre_activations, batch, epsilon=1e-8, train=True,
                   beta_init=tf.constant_initializer(0),
                   gamma_init=tf.constant_initializer(1)):
    """Does full batch normalisation of pre activations.
    Expects to get given something pre-nonlinearity.

    This is only set up for feed forward nets, in order to work properly for
    recurrent nets we will need to know what step we are up to, as in the 
    paper they calculate population statistics at every time step.

    Args:
      pre_activations: the logits who will be normalised. We assume this is
        of shape [batch_size, num_units]
      batch: the data which generated the logits, which we need to calculate
        statistics used to normalise.
      train: if true, the statistics will be recalculated for each batch. If not,
        then the average from the training phase will be used.

    Returns:
      batch normalised activations.
    """
    # get beta and gamma
    num_units = pre_activations.get_shape()[0]
    beta = tf.get_variable('beta', [num_units])
    gamma = tf.get_variable('gamma', [num_units])
    mean, variance = tf.nn.moments(pre_activations, [0])
    isqr = tf.rsqrt(variance+epsilon)
    centered = pre_activations - mean
    return beta + gamma * centered * isqr
    
