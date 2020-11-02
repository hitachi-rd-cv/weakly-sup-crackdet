# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - utils_deeplab.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import utils_deeplab
import tensorflow as tf

import os
import cv2
import json


_NUM_SHARDS = 4


def _convert_dataset(dataset_split, output_dir, image_folder, image_format,
                     semantic_segmentation_folder, label_format, list_folder):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  try:
      os.makedirs(output_dir)
  except:
    pass

  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = utils_deeplab.ImageReader('jpeg', channels=3)
  label_reader = utils_deeplab.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    print('-'*20)
    print(output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            image_folder, filenames[i] + '.' + image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            semantic_segmentation_folder,
            filenames[i] + '.' + label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = utils_deeplab.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def generateTFRecord(output_dir, image_folder, image_format,
                     semantic_segmentation_folder, label_format, list_folder):
  dataset_splits = tf.gfile.Glob(os.path.join(list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split, output_dir, image_folder, image_format,
                     semantic_segmentation_folder, label_format, list_folder)


def makeDir(dirname):
    # create a folder
    if type(dirname) == str:
        dirname = [dirname]

    for d in dirname:
        try:
            os.makedirs(d)
        except:
            pass


def convertInput(ori_logdir, out_logdir):
    in_imgdir = os.path.join(ori_logdir, 'original')
    gt_imgdir = os.path.join(ori_logdir, 'gt')
    data_split = os.path.join(ori_logdir, 'data_split.json')

    with open(data_split, 'r') as f:
        data_split_d = json.load(f)

    img_outdir = os.path.join(out_logdir, 'img')
    lbl_outdir = os.path.join(out_logdir, 'lbl')
    lst_outdir = os.path.join(out_logdir, 'lst')
    testimg_outdir = os.path.join(out_logdir, 'test_img')
    testlbl_outdir = os.path.join(out_logdir, 'test_lbl')
    makeDir([out_logdir, img_outdir, lbl_outdir, lst_outdir,
             testimg_outdir, testlbl_outdir])

    dir_map = [(in_imgdir, img_outdir, data_split_d['train']['input'], False),
               (gt_imgdir, lbl_outdir, data_split_d['train']['gt'], True),
               (in_imgdir, testimg_outdir, data_split_d['test']['input'], False),
               (gt_imgdir, testlbl_outdir, data_split_d['test']['gt'], True)]

    for fromdir, todir, namelist, scale in dir_map:
        for imgname in namelist:
            from_name = os.path.join(fromdir, imgname)

            to_fname = imgname.replace('_gt', '')
            to_fname = os.path.splitext(to_fname)[0] + '.png'
            to_name = os.path.join(todir, to_fname)

            img = cv2.imread(from_name) / 255 if scale else cv2.imread(from_name)
            cv2.imwrite(to_name, img)
            # shutil.copy(from_name, to_name)

    imgname = [os.path.splitext(fname)[0] for fname
               in data_split_d['train']['input']]
    with open(os.path.join(lst_outdir, 'train.txt'), 'w+') as f:
        f.write('\n'.join(imgname))


def main(ori_logdir, img_out_logdir, tf_out_logdir):
    convertInput(ori_logdir, img_out_logdir)

    image_folder = os.path.join(img_out_logdir, 'img')
    semantic_segmentation_folder = os.path.join(img_out_logdir, 'lbl')
    list_folder = os.path.join(img_out_logdir, 'lst')
    generateTFRecord(
        output_dir=tf_out_logdir,
        image_folder=image_folder,
        semantic_segmentation_folder=semantic_segmentation_folder,
        list_folder=list_folder,
        label_format='png', image_format='png')
