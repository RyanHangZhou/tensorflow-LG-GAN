""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2017
"""

import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import math_ops

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    assert(data_format=='NHWC' or data_format=='NCHW')
    if data_format == 'NHWC':
      num_in_channels = inputs.get_shape()[-1].value
    elif data_format=='NCHW':
      num_in_channels = inputs.get_shape()[1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn',
                                      data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs




def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.get_shape()[-1].value
      elif data_format=='NCHW':
        num_in_channels = inputs.get_shape()[1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn',
                                        data_format=data_format)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_bn(labels_onehot, inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):

  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.get_shape()[-1].value
      elif data_format=='NCHW':
        num_in_channels = inputs.get_shape()[1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

      if bn:
        outputs = batch_norm_for_conv2d_bn(labels_onehot, outputs, is_training,
                                        bn_decay=bn_decay, scope='bn',
                                        data_format=data_format)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

   

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

# def batch_norm_template_bn(labels_onehot, inputs,
#                decay=0.999,
#                center=True,
#                scale=False,
#                epsilon=0.001,
#                activation_fn=None,
#                updates_collections=ops.GraphKeys.UPDATE_OPS,
#                is_training=True,
#                reuse=None,
#                variables_collections=None,
#                outputs_collections=None,
#                trainable=True,
#                scope=None):
#   """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.
#     "Batch Normalization: Accelerating Deep Network Training by Reducing
#     Internal Covariate Shift"
#     Sergey Ioffe, Christian Szegedy
#   Can be used as a normalizer function for conv2d and fully_connected.
#   Args:
#     inputs: a tensor of size `[batch_size, height, width, channels]`
#             or `[batch_size, channels]`.
#     decay: decay for the moving average.
#     center: If True, subtract `beta`. If False, `beta` is ignored.
#     scale: If True, multiply by `gamma`. If False, `gamma` is
#       not used. When the next layer is linear (also e.g. `nn.relu`), this can be
#       disabled since the scaling can be done by the next layer.
#     epsilon: small float added to variance to avoid dividing by zero.
#     activation_fn: Optional activation function.
#     updates_collections: collections to collect the update ops for computation.
#       If None, a control dependency would be added to make sure the updates are
#       computed.
#     is_training: whether or not the layer is in training mode. In training mode
#       it would accumulate the statistics of the moments into `moving_mean` and
#       `moving_variance` using an exponential moving average with the given
#       `decay`. When it is not in training mode then it would use the values of
#       the `moving_mean` and the `moving_variance`.
#     reuse: whether or not the layer and its variables should be reused. To be
#       able to reuse the layer scope must be given.
#     variables_collections: optional collections for the variables.
#     outputs_collections: collections to add the outputs.
#     trainable: If `True` also add variables to the graph collection
#       `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
#     scope: Optional scope for `variable_op_scope`.
#   Returns:
#     a tensor representing the output of the operation.
#   """
#   with variable_scope.variable_op_scope([inputs],
#                                         scope, 'BatchNorm', reuse=reuse) as sc:
#     inputs_shape = inputs.get_shape()
#     dtype = inputs.dtype.base_dtype
#     axis = list(range(len(inputs_shape) - 1))
#     params_shape = inputs_shape[-1:]
#     # Allocate parameters for the beta and gamma of the normalization.
#     beta, gamma = None, None
#     if center:
#       beta_collections = utils.get_variable_collections(variables_collections,
#                                                         'beta')
#       beta = variables.model_variable('beta',
#                                       shape=params_shape,
#                                       dtype=dtype,
#                                       initializer=init_ops.zeros_initializer,
#                                       collections=beta_collections,
#                                       trainable=trainable)
#     if scale:
#       gamma_collections = utils.get_variable_collections(variables_collections,
#                                                          'gamma')
#       gamma = variables.model_variable('gamma',
#                                        shape=params_shape,
#                                        dtype=dtype,
#                                        initializer=init_ops.ones_initializer,
#                                        collections=gamma_collections,
#                                        trainable=trainable)
#     # Create moving_mean and moving_variance variables and add them to the
#     # appropiate collections.
#     moving_mean_collections = utils.get_variable_collections(
#         variables_collections, 'moving_mean')
#     moving_mean = variables.model_variable(
#         'moving_mean',
#         shape=params_shape,
#         dtype=dtype,
#         initializer=init_ops.zeros_initializer,
#         trainable=False,
#         collections=moving_mean_collections)
#     moving_variance_collections = utils.get_variable_collections(
#         variables_collections, 'moving_variance')
#     moving_variance = variables.model_variable(
#         'moving_variance',
#         shape=params_shape,
#         dtype=dtype,
#         initializer=init_ops.ones_initializer,
#         trainable=False,
#         collections=moving_variance_collections)
#     if is_training:
#       # Calculate the moments based on the individual batch.
#       mean, variance = nn.moments(inputs, axis, shift=moving_mean)
#       # Update the moving_mean and moving_variance moments.
#       update_moving_mean = moving_averages.assign_moving_average(
#           moving_mean, mean, decay)
#       update_moving_variance = moving_averages.assign_moving_average(
#           moving_variance, variance, decay)
#       if updates_collections is None:
#         # Make sure the updates are computed here.
#         with ops.control_dependencies([update_moving_mean,
#                                        update_moving_variance]):
#           outputs = nn.batch_normalization(
#               inputs, mean, variance, beta, gamma, epsilon)
#       else:
#         # Collect the updates to be computed later.
#         ops.add_to_collections(updates_collections, update_moving_mean)
#         ops.add_to_collections(updates_collections, update_moving_variance)
#         outputs = nn.batch_normalization(
#             inputs, mean, variance, beta, gamma, epsilon)
#     else:
#       outputs = nn.batch_normalization(
#           inputs, moving_mean, moving_variance, beta, gamma, epsilon)
#     outputs.set_shape(inputs.get_shape())
#     if activation_fn:
#       outputs = activation_fn(outputs)
#     return utils.collect_named_outputs(outputs_collections, sc.name, outputs)



# def batch_norm_template_bn(labels_onehot, inputs, is_training, scope, moments_dims, bn_decay, data_format='NHWC'):
#   """ NOTE: this is older version of the util func. it is deprecated.
#   Batch normalization on convolutional maps and beyond...
#   Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
#   Args:
#       inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
#       is_training:   boolean tf.Varialbe, true indicates training phase
#       scope:         string, variable scope
#       moments_dims:  a list of ints, indicating dimensions for moments calculation
#       bn_decay:      float or float tensor variable, controling moving average weight
#   Return:
#       normed:        batch-normalized maps
#   """
#   with tf.variable_scope(scope) as sc:
#     num_channels = inputs.get_shape()[-1].value
#     beta = _variable_on_cpu(name='beta',shape=[num_channels],
#                             initializer=tf.constant_initializer(0))
#     gamma = _variable_on_cpu(name='gamma',shape=[num_channels],
#                             initializer=tf.constant_initializer(1.0))
#     batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
#     decay = bn_decay if bn_decay is not None else 0.9
#     ema = tf.train.ExponentialMovingAverage(decay=decay)
#     # Operator that maintains moving averages of variables.
#     # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
#     # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
#     with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#         ema_apply_op = tf.cond(tf.cast(is_training, tf.bool),
#                                lambda: ema.apply([batch_mean, batch_var]),
#                                lambda: tf.no_op())
    
#     # Update moving average and return current batch's avg and var.
#     def mean_var_with_update():
#       with tf.control_dependencies([ema_apply_op]):
#         return tf.identity(batch_mean), tf.identity(batch_var)
    
#     # ema.average returns the Variable holding the average of var.
#     mean, var = tf.cond(tf.cast(is_training, tf.bool),
#                         mean_var_with_update,
#                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#     normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
#   return normed


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  bn_decay = bn_decay if bn_decay is not None else 0.9
  return tf.contrib.layers.batch_norm(inputs, 
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope=scope,
                                      data_format=data_format)


# def batch_norm_template_bn(labels_onehot, inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
#   """ Batch normalization on convolutional maps and beyond...
#   Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
#   Args:
#       inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
#       is_training:   boolean tf.Varialbe, true indicates training phase
#       scope:         string, variable scope
#       moments_dims:  a list of ints, indicating dimensions for moments calculation
#       bn_decay:      float or float tensor variable, controling moving average weight
#       data_format:   'NHWC' or 'NCHW'
#   Return:
#       normed:        batch-normalized maps
#   """
#   bn_decay = bn_decay if bn_decay is not None else 0.9
#   return tf.contrib.layers.batch_norm(inputs, 
#                                       center=True, scale=True,
#                                       is_training=is_training, decay=bn_decay,updates_collections=None,
#                                       scope=scope,
#                                       data_format=data_format)

def batch_norm_template_bn(labels_onehot, inputs, is_training, scope, moments_dims, bn_decay, data_format='NHWC'):
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = _variable_on_cpu(name='beta',shape=[num_channels],
                            initializer=tf.constant_initializer(0))
    print('I need the beta, ', np.shape(beta))
    print('input number of channels, ', num_channels)

    out_channels = np.shape(labels_onehot)[1]
    labels_onehot = tf.expand_dims(labels_onehot, 2)
    print(out_channels)
    kernel_shape = [labels_onehot.get_shape()[1], 1, num_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=True,
                                         stddev=1e-3,
                                         wd=None)
    print('labels_onehot', np.shape(labels_onehot))
    offset = tf.nn.tanh(tf.nn.conv1d(labels_onehot, kernel,
                           stride=1,
                           padding='VALID',
                           data_format='NHWC'))
    print('output shapesss', np.shape(offset))


    kernel2 = _variable_with_weight_decay('weights2',
                                         shape=kernel_shape,
                                         use_xavier=True,
                                         stddev=1e-3,
                                         wd=None)
    scale = tf.nn.tanh(tf.nn.conv1d(labels_onehot, kernel2,
                           stride=1,
                           padding='VALID',
                           data_format='NHWC'))

    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op()) ###########lambda 改掉
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    print('inputs size', np.shape(inputs))
    print('mean size', np.shape(mean))
    print('var size', np.shape(var))
    print('beta size', np.shape(beta))
    # print('gamma size', np.shape(gamma))
    normed = batch_normalization_my(inputs, mean, var, offset, scale, 1e-3)
  return normed


def batch_normalization_my(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None):

  with ops.name_scope(name, "batchnorm", [x, mean, variance, scale, offset]):
    inv = math_ops.rsqrt(variance + variance_epsilon)

    scale = tf.cast(scale, tf.float32)
    scale = tf.expand_dims(scale, 1)
    scale = tf.tile(scale, [1, x.get_shape()[1], x.get_shape()[2], 1])
    offset = tf.cast(offset, tf.float32)
    offset = tf.expand_dims(offset, 1)
    offset = tf.tile(offset, [1, x.get_shape()[1], x.get_shape()[2], 1])

    if scale is not None:
      inv *= scale
    # Note: tensorflow/contrib/quantize/python/fold_batch_norms.py depends on
    # the precise order of ops that are generated by the expression below.
    return x * math_ops.cast(inv, x.dtype) + math_ops.cast(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype)



def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, data_format):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay, data_format)



  
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay, data_format)


def batch_norm_for_conv2d_bn(labels_onehot, inputs, is_training, bn_decay, scope, data_format):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template_bn(labels_onehot, inputs, is_training, scope, [0,1,2], bn_decay, data_format)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs
