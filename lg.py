import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import h5py
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import lgnet as lgnet

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_attacked', default='pointnet_cls', help='Attacked-model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--model_path', default='pointnet/model.ckpt', help='model checkpoint file path [default: pointnet/model.ckpt]')
parser.add_argument('--adv_path', default='LGGAN', help='output adversarial example path [default: LGGAN]')
parser.add_argument('--checkpoints_path', default='LGGAN', help='output checkpoints path [default: LGGAN]')
parser.add_argument('--log_path', default='LGGAN', help='output log file [default: LGGAN]')
parser.add_argument('--tau', type=float, default=1e2, help='balancing weight for loss function [default: 1e2]')
FLAGS = parser.parse_args()

LEARNING_RATE = 1e-3
ITERATION = 100

# create adversarial example path
ADV_PATH = FLAGS.adv_path
if not os.path.exists('results'): os.mkdir('results')
ADV_PATH = os.path.join('results', ADV_PATH)
if not os.path.exists(ADV_PATH): os.mkdir(ADV_PATH)
ADV_PATH = os.path.join(ADV_PATH, 'test')

# create LG-GAN checkpoint path
CHECKPOINTS_PATH = FLAGS.checkpoints_path
if not os.path.exists('checkpoints'): os.mkdir('checkpoints')
CHECKPOINTS_PATH = os.path.join('checkpoints', CHECKPOINTS_PATH)
if not os.path.exists(CHECKPOINTS_PATH): os.mkdir(CHECKPOINTS_PATH)

# create LG-GAN log file
LOG_PATH = FLAGS.log_path
if not os.path.exists('log'): os.mkdir('log')
LOG_FOUT = open(os.path.join('log', LOG_PATH+'.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = os.path.join('checkpoints', FLAGS.model_path)
GPU_INDEX = FLAGS.gpu
MODEL_ATTACKED = importlib.import_module(FLAGS.model_attacked)
TAU = FLAGS.tau

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


def log_string(out_str):

    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def generate_labels(labels):

    targets = np.zeros(np.size(labels))
    for i in range(len(labels)):
        rand_v = random.randint(0, NUM_CLASSES-1)
        while labels[i]==rand_v:
            rand_v = random.randint(0, NUM_CLASSES-1)
        targets[i] = rand_v
    targets = targets.astype(np.int32)

    return targets


def write_h5(data, data_orig, label, label_orig, num_batches):

    h5_filename = ADV_PATH+str(num_batches)+'.h5'
    h5f = h5py.File(h5_filename, 'w')
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('orig_data', data=data_orig)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('orig_label', data=label_orig)
    h5f.close()


def get_bn_decay(batch):

    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

    return bn_decay


def evaluate(num_votes=1):

    with tf.device('/gpu:'+str(0)):

        fout = open(os.path.join('log', 'pred_label.txt'), 'w')
        shape = (BATCH_SIZE, NUM_POINT, 3)
        is_training_pc = tf.placeholder(tf.bool, shape=())

        with tf.variable_scope('foo'):

            data_pl, labels_pl = MODEL_ATTACKED.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            labels_onehot = tf.one_hot(labels_pl, NUM_CLASSES, 1, 0)
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            is_training_pl_ae = tf.placeholder(tf.bool, shape=())

            generator_pc, _ = lgnet.get_gen_model_m(data_pl, labels_onehot, is_training_pl_ae, scope='generator', bradius=1.0,
                                                              reuse=None, use_normal=False, use_bn=False, use_ibn=False,
                                                              bn_decay=bn_decay, up_ratio=1)

        pred, end_points = MODEL_ATTACKED.get_model(generator_pc, is_training_pc)

        # loss functions
        pred_loss = MODEL_ATTACKED.get_loss(pred, labels_pl, end_points)
        generator_loss = tf.nn.l2_loss(generator_pc-data_pl)
        total_loss = generator_loss + TAU*pred_loss

        net_var = []
        gen_var = []
        for var in tf.global_variables():
            if not var.name.startswith('foo'):
                net_var.append(var)
            else:
                gen_var.append(var)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(net_var)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train = optimizer.minimize(total_loss, var_list=[gen_var])

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Attacked-model restored.")
    
    for my_iter in range(ITERATION):

        error_cnt = 0
        is_training = False
        total_correct_adv = 0
        total_seen = 0
        total_attack_adv = 0
        total_seen_class_adv = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_adv = [0 for _ in range(NUM_CLASSES)]

        # testing phase
        is_training_ae = False
        for test_fn in range(len(TEST_FILES)):

            log_string('----testing stage '+str(test_fn)+'----')
            
            test_data, test_label = provider.loadDataFile(TEST_FILES[test_fn])
            test_data = test_data[:,0:NUM_POINT,:]
            test_label = np.squeeze(test_label)

            test_file_size = test_data.shape[0]
            test_num_batches = test_file_size // BATCH_SIZE

            if(test_fn==0):
                iter_num = test_num_batches
        
            for test_batch_idx in range(test_num_batches):

                test_start_idx = test_batch_idx * BATCH_SIZE
                test_end_idx = (test_batch_idx+1) * BATCH_SIZE
                test_cur_batch_size = test_end_idx - test_start_idx

                test_batch_pred_sum = np.zeros((test_cur_batch_size, NUM_CLASSES))
                test_batch_loss_sum_adv = 0

                for test_vote_idx in range(num_votes):
                    test_rotated_data = test_data[test_start_idx:test_end_idx, :, :]

                    original_labels = test_label[test_start_idx:test_end_idx]
                    target_labels = generate_labels(original_labels)

                    feed_dict = {data_pl: test_rotated_data,
                             labels_pl: target_labels,
                             is_training_pc: is_training,
                             is_training_pl_ae: is_training_ae}
                    _, score, test_adv_data = sess.run([train, pred, generator_pc], feed_dict=feed_dict)

                pred_val_adv = np.argmax(score, axis=1)
                write_h5(test_adv_data, test_rotated_data, target_labels, original_labels, test_fn*iter_num+test_batch_idx)
            
                correct_adv = np.sum(pred_val_adv == original_labels)
                attack_adv = np.sum(pred_val_adv == target_labels)

                total_correct_adv += correct_adv
                total_attack_adv += attack_adv
                total_seen += test_cur_batch_size

                for i in range(test_start_idx, test_end_idx):

                    l = test_label[i]
                    total_seen_class_adv[l] += 1
                    total_correct_class_adv[l] += (pred_val_adv[i-test_start_idx] == l)

                    fout.write('%d, %d\n' % (pred_val_adv[i-test_start_idx], l))
                    fout.flush()

        log_string('eval adv accuracy: %f' % (total_correct_adv / float(total_seen)))
        log_string('eval adv attack success rate: %f' % (total_attack_adv / float(total_seen)))
        log_string('eval adv avg class acc: %f' % (np.mean(np.array(total_correct_class_adv)/np.array(total_seen_class_adv,dtype=np.float))))
    
        class_accuracies_adv = np.array(total_correct_class_adv)/np.array(total_seen_class_adv,dtype=np.float)

        for i, name in enumerate(SHAPE_NAMES):
            log_string('%10s:\t%0.3f' % (name, class_accuracies_adv[i]))



        # training phase
        is_training_ae = True
        for fn in range(len(TRAIN_FILES)):

            log_string('----training stage '+str(fn)+'----')

            current_data, current_label = provider.loadDataFile(TRAIN_FILES[fn])
            current_data = current_data[:,0:NUM_POINT,:]
            current_label = np.squeeze(current_label)
        
            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            if(fn==0):
                iter_num = num_batches
        
            for batch_idx in range(num_batches):

                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                cur_batch_size = end_idx - start_idx
            
                batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES))
                batch_loss_sum_adv = 0

                for vote_idx in range(num_votes):

                    rotated_data = current_data[start_idx:end_idx, :, :]
                    target_labels = generate_labels(current_label[start_idx:end_idx])

                    feed_dict = {data_pl: rotated_data,
                             labels_pl: target_labels,
                             is_training_pc: is_training,
                             is_training_pl_ae: is_training_ae}
                    _ = sess.run([train], feed_dict=feed_dict)
 
            if fn % 10 == 0:
                saver.save(sess, os.path.join(CHECKPOINTS_PATH, "model"+str(fn)+".cpkt"))

    fout.close()


if __name__=='__main__':

    with tf.Graph().as_default():
        evaluate(num_votes=1)

    LOG_FOUT.close()
