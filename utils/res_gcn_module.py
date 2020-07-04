import tensorflow as tf
import numpy as np

from tf_ops.grouping.tf_grouping import group_point, knn_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point

from tensorflow.python.training.moving_averages import assign_moving_average


def batch_norm_orig(x, train, beta, gamma, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tf.shape(x)[-1:]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x


# def batch_norm2(x, train, beta, gamma, eps=1e-05, decay=0.9, affine=True, name=None):
#     with tf.variable_scope(name, default_name='BatchNorm2d'):
#         params_shape = tf.shape(x)[-1:]
#         moving_mean = tf.get_variable('mean', params_shape,
#                                       initializer=tf.zeros_initializer,
#                                       trainable=False)
#         moving_variance = tf.get_variable('variance', params_shape,
#                                           initializer=tf.ones_initializer,
#                                           trainable=False)

#         def mean_var_with_update():
#             mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
#             with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
#                                           assign_moving_average(moving_variance, variance, decay)]):
#                 return tf.identity(mean), tf.identity(variance)
#         mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
#         x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
#         # if affine:
#         #     beta = tf.get_variable('beta', params_shape,
#         #                            initializer=tf.zeros_initializer)
#         #     gamma = tf.get_variable('gamma', params_shape,
#         #                             initializer=tf.ones_initializer)
#         #     x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
#         # else:
#         #     x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
#         return x


# def batch_norm2(x, beta, gamma, train_phase=True, eps=1e-5, decay=0.9, affine=True, name=None, scope_bn='zz'):
#     with tf.variable_scope(scope_bn):
#         beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
#         axises = list(np.arange(len(x.shape) - 1))
#         batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay)

#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)

#         mean, var = tf.cond(train_phase, mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
#     return normed


def batch_norm2(x, beta, gamma, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tf.shape(x)[-1:]
        print('xxxxxxshapexxxxxxxxxxxxx', np.shape(x))
        moving_mean = tf.get_variable('mean', shape=[x.shape[-1]],
                                      initializer=tf.zeros_initializer,
                                      trainable=True)
        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzz', np.shape(moving_mean))
        moving_variance = tf.get_variable('variance', shape=[x.shape[-1]],
                                          initializer=tf.ones_initializer,
                                          trainable=True)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, list(np.arange(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        print('kkkkkkkkkkkkkkkkkkkkkkkkkk', np.shape(mean))
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        # if affine:
        #     beta = tf.get_variable('beta', [x.shape[-1]],
        #                            initializer=tf.zeros_initializer)
        #     gamma = tf.get_variable('gamma', [x.shape[-1]],
        #                             initializer=tf.ones_initializer)
        #     x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        # else:
        #     x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x



# def batch_norm33(x, train_phase=True, eps=1e-5, decay=0.9, affine=True, name=None, scope_bn='zz'):
#     with tf.variable_scope(scope_bn):
#         beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
#         axises = list(np.arange(len(x.shape) - 1))
#         batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay)

#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)

#         mean, var = tf.cond(train_phase, mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
#     return normed


def batch_norm33(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tf.shape(x)[-1:]
        print('xxxxxxshapexxxxxxxxxxxxx', np.shape(x))
        moving_mean = tf.get_variable('mean', shape=[x.shape[-1]],
                                      initializer=tf.zeros_initializer,
                                      trainable=True)
        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzz', np.shape(moving_mean))
        moving_variance = tf.get_variable('variance', shape=[x.shape[-1]],
                                          initializer=tf.ones_initializer,
                                          trainable=True)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, list(np.arange(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        print('kkkkkkkkkkkkkkkkkkkkkkkkkk', np.shape(mean))
        # x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        if affine:
            beta = tf.get_variable('beta', [x.shape[-1]],
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', [x.shape[-1]],
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x

def batch_norm4(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2dxx'):
        params_shape = tf.shape(x)[-1:]
        print('xxxxxxshapexxxxxxxxxxxxx', np.shape(x))
        moving_mean = tf.get_variable('mean', shape=[x.shape[-1]],
                                      initializer=tf.zeros_initializer,
                                      trainable=True)
        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzz', np.shape(moving_mean))
        moving_variance = tf.get_variable('variance', shape=[x.shape[-1]],
                                          initializer=tf.ones_initializer,
                                          trainable=True)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, list(np.arange(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        print('kkkkkkkkkkkkkkkkkkkkkkkkkk', np.shape(mean))
        # x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        if affine:
            beta = tf.get_variable('beta', [x.shape[-1]],
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', [x.shape[-1]],
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x


def conv2d(input, n_cout, name, kernel_size=(1, 1),
           strides=(1, 1), padding='VALID', use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001)):
    return tf.layers.conv2d(input, n_cout, kernel_size=kernel_size, strides=strides,
                            padding=padding, name=name, use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)


def conv2dbn(input, n_cout, name, is_training, kernel_size=(1, 1),
           strides=(1, 1), padding='VALID', use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001)):
    out1 = tf.layers.conv2d(input, n_cout, kernel_size=kernel_size, strides=strides,
                            padding=padding, name=name, use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)
    print('my output', np.shape(out1))
    # return tf.layers.batch_normalization(out1, axis=0, training=is_training)
    return batch_norm33(out1, is_training, name='bn')

def conv2dbn4(input, n_cout, name, is_training, kernel_size=(1, 1),
           strides=(1, 1), padding='VALID', use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001)):
    out1 = tf.layers.conv2d(input, n_cout, kernel_size=kernel_size, strides=strides,
                            padding=padding, name=name, use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)
    print('my output', np.shape(out1))
    # return tf.layers.batch_normalization(out1, axis=0, training=is_training)
    return batch_norm4(out1, is_training, name='bnxx')


def conv2dbn3(input, n_cout, name, is_training, beta, gamma, kernel_size=(1, 1),
           strides=(1, 1), padding='VALID', use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001)):
    out1 = tf.layers.conv2d(input, n_cout, kernel_size=kernel_size, strides=strides,
                            padding=padding, name=name, use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)
    mean, variance = tf.nn.moments(out1, axes=[0, 1, 2], name='moments')

    fc = tf.contrib.layers.flatten(beta)
    # fc = tf.layers.conv2d(fc, 1, 1, activation=tf.nn.relu)
    fc = tf.nn.relu(fc)
    beta2 = tf.reshape(fc, beta.get_shape().as_list())
    gamma = beta2

    # return batch_norm2(out1, beta, gamma)
    return batch_norm2(out1, beta, gamma, is_training, name='bn')
    # return tf.nn.batch_normalization(out1, mean, variance, beta, gamma, variance_epsilon=1e-5)
    # return tf.layers.batch_normalization(out1, axis=0, training=is_training)


def conv2dbn5(input, n_cout, name, is_training, kernel_size=(1, 1),
           strides=(1, 1), padding='VALID', use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001)):
    out1 = tf.layers.conv2d(input, n_cout, kernel_size=kernel_size, strides=strides,
                            padding=padding, name=name, use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)
    mean, variance = tf.nn.moments(out1, axes=[0, 1, 2], name='moments')

    # return batch_norm2(out1, beta, gamma)
    return batch_norm5(out1, is_training, name='bnxx')
    # return tf.nn.batch_normalization(out1, mean, variance, beta, gamma, variance_epsilon=1e-5)
    # return tf.layers.batch_normalization(out1, axis=0, training=is_training)


def batch_norm5(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tf.shape(x)[-1:]
        print('xxxxxxshapexxxxxxxxxxxxx', np.shape(x))
        moving_mean = tf.get_variable('mean', shape=[x.shape[-1]],
                                      initializer=tf.zeros_initializer,
                                      trainable=True)
        print('zzzzzzzzzzzzzzzzzzzzzzzzzzzz', np.shape(moving_mean))
        moving_variance = tf.get_variable('variance', shape=[x.shape[-1]],
                                          initializer=tf.ones_initializer,
                                          trainable=True)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, list(np.arange(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        print('kkkkkkkkkkkkkkkkkkkkkkkkkk', np.shape(mean))
        # x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        if affine:
            beta = tf.get_variable('beta', [x.shape[-1]],
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', [x.shape[-1]],
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x

# def conv2dbn3(input, n_cout, name, is_training, beta, gamma, kernel_size=(1, 1),
#            strides=(1, 1), padding='VALID', use_bias=False,
#            kernel_initializer=tf.contrib.layers.xavier_initializer(),
#            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001)):
#     out1 = tf.layers.conv2d(input, n_cout, kernel_size=kernel_size, strides=strides,
#                             padding=padding, name=name, use_bias=use_bias,
#                             kernel_initializer=kernel_initializer,
#                             kernel_regularizer=kernel_regularizer)
#     mean, variance = tf.nn.moments(out1, axes=[0, 1, 2], name='moments')
    
#     fc = tf.contrib.layers.flatten(beta)
#     # fc = tf.layers.conv2d(fc, 1, 1, activation=tf.nn.relu)
#     fc = tf.nn.relu(fc)
#     beta2 = tf.reshape(fc, beta.get_shape().as_list())
#     gamma = beta2

#     return tf.nn.batch_normalization(out1, mean, variance, beta2, gamma, variance_epsilon=1e-5)


def batch_norm(input, is_training, name, bn_decay=None, use_bn=True, use_ibn=False):
    if use_bn:
        return tf.layers.batch_normalization(input, name=name, training=is_training, momentum=bn_decay)

    if use_ibn:
        return tf.contrib.layers.instance_norm(input, scope=name)

    return input


def group(xyz, points, k, dilation=1, use_xyz=False):
    _, idx = knn_point(k*dilation+1, xyz, xyz)
    idx = idx[:, :, 1::dilation]

    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, k, 3)
    grouped_xyz -= tf.expand_dims(xyz, 2)  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, k, channel)
        if use_xyz:
            grouped_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, k, 3+channel)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


def pool(xyz, points, k, npoint):
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
    _, idx = knn_point(k, xyz, new_xyz)
    new_points = tf.reduce_max(group_point(points, idx), axis=2)

    return new_xyz, new_points

#                 8,   128,    3
def pointcnn(xyz, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, activation=tf.nn.relu):
    with tf.variable_scope(scope):
        # grouped_points: knn points coordinates (normalized: minus centual points)
        _, grouped_points, _ = group(xyz, None, k)

        # print('n_blocks: ', n_blocks)
        # print('is_training: ', is_training)

        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                grouped_points = conv2d(grouped_points, n_cout, name='conv_xyz')
                if idx == n_blocks - 1:
                    return tf.reduce_max(grouped_points, axis=2)
                else:
                    grouped_points = batch_norm(grouped_points, is_training, 'bn_xyz',
                                                bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                    grouped_points = activation(grouped_points)


def res_gcn_up(xyz, points, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, indices=None, up_ratio=2):
    with tf.variable_scope(scope):
        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                shortcut = points

                # Center Features
                points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                points = tf.nn.relu(points)
                # Neighbor Features
                if idx == 0 and indices is None:
                    _, grouped_points, indices = group(xyz, points, k)
                else:
                    grouped_points = group_point(points, indices)
                # Center Conv
                center_points = tf.expand_dims(points, axis=2)
                points = conv2d(center_points, n_cout, name='conv_center')
                # Neighbor Conv
                grouped_points_nn = conv2d(grouped_points, n_cout, name='conv_neighbor')
                # CNN
                points = tf.reduce_mean(tf.concat([points, grouped_points_nn], axis=2), axis=2) + shortcut

                if idx == n_blocks - 1:
                    # Center Conv
                    points_xyz = conv2d(center_points, 3*up_ratio, name='conv_center_xyz')
                    # Neighbor Conv
                    grouped_points_xyz = conv2d(grouped_points, 3*up_ratio, name='conv_neighbor_xyz')
                    # CNN
                    new_xyz = tf.reduce_mean(tf.concat([points_xyz, grouped_points_xyz], axis=2), axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value, up_ratio, 3])
                    new_xyz = new_xyz + tf.expand_dims(xyz, axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value*up_ratio, 3])

                    return new_xyz, points


def res_gcn_up2(xyz, points, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, indices=None, up_ratio=2):
    with tf.variable_scope(scope):
        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                shortcut = points

                # Center Features
                points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                points = tf.nn.relu(points)
                # Neighbor Features
                if idx == 0 and indices is None:
                    _, grouped_points, indices = group(xyz, points, k)
                else:
                    grouped_points = group_point(points, indices)
                # Center Conv
                center_points = tf.expand_dims(points, axis=2)
                points = conv2dbn(center_points, n_cout, name='conv_center', is_training=is_training)
                # Neighbor Conv
                grouped_points_nn = conv2dbn(grouped_points, n_cout, name='conv_neighbor', is_training=is_training)
                # CNN
                points = tf.reduce_mean(tf.concat([points, grouped_points_nn], axis=2), axis=2) + shortcut

                if idx == n_blocks - 1:
                    # Center Conv
                    print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq', np.shape(center_points))
                    points_xyz = conv2dbn4(center_points, 3*up_ratio, name='conv_center_xyz', is_training=is_training)
                    # Neighbor Conv
                    grouped_points_xyz = conv2dbn4(grouped_points, 3*up_ratio, name='conv_neighbor_xyz', is_training=is_training)
                    # CNN
                    new_xyz = tf.reduce_mean(tf.concat([points_xyz, grouped_points_xyz], axis=2), axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value, up_ratio, 3])
                    new_xyz = new_xyz + tf.expand_dims(xyz, axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value*up_ratio, 3])

                    return new_xyz, points


def res_gcn_up3(xyz, points, labels_onehot, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, indices=None, up_ratio=2):
    with tf.variable_scope(scope):
        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                shortcut = points

                # Center Features
                points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                points = tf.nn.relu(points)
                # Neighbor Features
                if idx == 0 and indices is None:
                    _, grouped_points, indices = group(xyz, points, k)
                else:
                    grouped_points = group_point(points, indices)
                # Center Conv
                center_points = tf.expand_dims(points, axis=2)
                # if idx==0 or idx==1 or idx==2:
                points = conv2dbn3(center_points, n_cout, name='conv_center', is_training=is_training, beta=labels_onehot, gamma=labels_onehot)
                # else:
                    # points = conv2dbn(center_points, n_cout, name='conv_center', is_training=is_training)
                # Neighbor Conv
                grouped_points_nn = conv2dbn3(grouped_points, n_cout, name='conv_neighbor', is_training=is_training, beta=labels_onehot, gamma=labels_onehot)
                # grouped_points_nn = conv2dbn(grouped_points, n_cout, name='conv_neighbor', is_training=is_training)
                # CNN
                points = tf.reduce_mean(tf.concat([points, grouped_points_nn], axis=2), axis=2) + shortcut

                if idx == n_blocks - 1:
                    # Center Conv
                    points_xyz = conv2dbn5(center_points, 3*up_ratio, name='conv_center_xyz', is_training=is_training)
                    # Neighbor Conv
                    grouped_points_xyz = conv2dbn5(grouped_points, 3*up_ratio, name='conv_neighbor_xyz', is_training=is_training)
                    # CNN
                    new_xyz = tf.reduce_mean(tf.concat([points_xyz, grouped_points_xyz], axis=2), axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value, up_ratio, 3])
                    new_xyz = new_xyz + tf.expand_dims(xyz, axis=2)
                    new_xyz = tf.reshape(new_xyz, [-1, new_xyz.get_shape()[1].value*up_ratio, 3])

                    return new_xyz, points

def res_gcn_d(xyz, points, k, n_cout, n_blocks, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False, indices=None):
    with tf.variable_scope(scope):
        for idx in range(n_blocks):
            with tf.variable_scope('block_{}'.format(idx)):
                shortcut = points

                # Center Features
                points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
                points = tf.nn.leaky_relu(points)
                # Neighbor Features
                if idx == 0 and indices is None:
                    _, grouped_points, indices = group(xyz, points, k)
                else:
                    grouped_points = group_point(points, indices)
                # Center Conv
                center_points = tf.expand_dims(points, axis=2)
                points = conv2d(center_points, n_cout, name='conv_center')
                # Neighbor Conv
                grouped_points_nn = conv2d(grouped_points, n_cout, name='conv_neighbor')
                # CNN
                points = tf.reduce_mean(tf.concat([points, grouped_points_nn], axis=2), axis=2) + shortcut

    return points


def res_gcn_d_last(points, n_cout, is_training, scope, bn_decay=None, use_bn=False, use_ibn=False):
    with tf.variable_scope(scope):
        points = batch_norm(points, is_training, 'bn_center', bn_decay=bn_decay, use_bn=use_bn, use_ibn=use_ibn)
        points = tf.nn.leaky_relu(points)
        center_points = tf.expand_dims(points, axis=2)
        points = tf.squeeze(conv2d(center_points, n_cout, name='conv_center'), axis=2)

        return points
