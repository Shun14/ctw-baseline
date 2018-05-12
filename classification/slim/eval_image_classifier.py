# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import numpy as np
import six
import tensorflow as tf
import time
from scipy import misc
import json
from chineselib import trainset
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from six.moves import cPickle
from tensorflow.python.training import saver as tf_saver
import settings

from pythonapi import common_tools
import sys
import eval

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  assert six.PY3

  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

#   with open(FLAGS.dataset_dir, 'rb') as f:
#     test_data = cPickle.load(f)
  
  pic_dir = '/home/zsz/ctw-baseline/classification/products/crop_char'
  pic_list = os.listdir(pic_dir)
  pic_list.sort()
  lines = len(pic_list)
  test_data = [[] for _ in range(lines)]
  
  j = 0
  for name in pic_list:
      image = np.array(misc.imread(os.path.join(pic_dir, name)))
      test_data[j].append(image)
      test_data[j].append(None)
      j += 1


  print(len(test_data))
  print(np.array(test_data).shape)

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = tf.train.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = trainset

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = trainset.get_tf_preprocess_image(
        is_training=False)

    assert FLAGS.eval_image_size is not None
    # assert FLAGS.eval_image_size == network_fn.default_image_size
    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    # eval_image_size = 32
    print('eval_image_size',eval_image_size)
    
    images_holder = [tf.placeholder(tf.uint8, shape=(None, None, 3)) for i in range(FLAGS.batch_size)]
    # images = map(lambda holder: image_preprocessing_fn(holder, eval_image_size, eval_image_size), images_holder)
    images = [image_preprocessing_fn(images_holder[i], eval_image_size, eval_image_size) for i in range(FLAGS.batch_size)]

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)
    eval_ops = logits

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        start_time = time.time()
        saver = tf_saver.Saver(variables_to_restore)
        saver.restore(session, checkpoint_path)
        results = []
        lo = 0
        while lo != len(test_data):
            hi = min(len(test_data), lo + FLAGS.batch_size)
            feed_data = test_data[lo:hi] + [(np.zeros((3, 3, 3), dtype=np.uint8), None)] * (lo + FLAGS.batch_size - hi)

            logits = session.run(eval_ops, feed_dict={images_holder[i]: feed_data[i][0] for i in range(FLAGS.batch_size)})
            results.append(logits[:hi - lo])
            lo = hi
            tf.logging.info('evaluated: %d / %d' % (lo, len(test_data)))
        end_time = time.time()

        logits = np.concatenate(results, axis=0)

        assert settings.NUM_CHAR_CATES + 1 == logits.shape[1]
        logits = logits[:, :settings.NUM_CHAR_CATES]
        explogits = np.exp(np.minimum(logits, 70))
        expsums = np.sum(explogits, axis=1) 
        expsums.shape = (logits.shape[0], 1)
        expsums = np.repeat(expsums, settings.NUM_CHAR_CATES, axis=1)
        probs = explogits / expsums
        argsorts = np.argsort(-logits, axis=1)
        
        lo = 0
        model_name = 'inception_v4'
        pred_file_name = os.path.join(settings.PRODUCTS_ROOT, 'self_predictions_{}.txt'.format(model_name))
        with open(settings.CATES) as f:
            cates = json.load(f)
        
#       j = 0
#       for name in os.listdir(pic_dir):
#           image = np.array(misc.imread(os.path.join(pic_dir, name)))
#           test_data[j].append(image)
#           test_data[j].append(None)
#           j += 1
        
        max_prediction = 5

        all_dict = [[] for _ in range(lines)]
        print(lines)
        for i in range(lines):
            pred = argsorts[lo][:max_prediction]
            prob = probs[lo][pred].tolist()
            final_prob = max(prob)
            print(prob)
            index = prob.index(final_prob)
            pred = list(map(lambda i: cates[i]['text'], pred.tolist()))
            print(pred)
            print(pic_list[i].split('_')[0])
            print(index)
            all_dict[int(pic_list[i].split('_')[0])].append((pic_list[i], str(pred[index].encode('utf-8').decode('utf-8'))))
                
                # 'gt_img:' + pic_list[i] +' ' +
                # 'predictions:' + str(pred[index].encode('utf-8').decode('utf-8'))
            i += 1
            lo += 1
        with open(pred_file_name, 'w') as fout:
            for l in all_dict:
                fout.write(str(l))
                fout.write('\n')
            
        assert lo == logits.shape[0]



        with open(FLAGS.eval_dir, 'wb') as f:
            cPickle.dump({
                'model_name': FLAGS.model_name,
                'checkpoint_path': checkpoint_path,
                'logits': np.concatenate(results, axis=0),
                'start_time': start_time,
                'end_time': end_time,
            }, f)

    tf.logging.info('Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                           time.gmtime()))


if __name__ == '__main__':
  tf.app.run()
