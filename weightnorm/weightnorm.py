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
        g = tf.get_variable('g', shape=shape[-1])
        #inv_norm = tf.rsqrt(tf.reduce_sum(tf.square(v), reduction_indices=axis))
        #w = v * g * inv_norm
        w = g * tf.nn.l2_normalize(v, 1)
        if return_all:
            return w, g, v
        return w
