# -*- coding: utf-8 -*-

import os
import cv2
import time

import numpy as np
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
DEFAULT_INPUT_MODEL = 'model'
DEFAULT_INPUT_GRAPHDEF = 'graph.pb'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('graph_dir_str', None, 'Inference Graph path')
flags.DEFINE_string('img_dir', None, 'Input image path')
flags.DEFINE_string('out_dir', None, 'Output image path')


class DL_Model(object):

    def __init__(self, graph_dir, input_node_name, out_tr_names):
        """
        Creates a DL model from a frozen graph file

        Args:
                :graph_dir (str):       frozen graph filename
                :input_node_name (str): input tensor node name
                :output_tr_name (str):  output tensor node name
        """
        config_proto = tf.ConfigProto(log_device_placement=False)
        self.sess = tf.Session(config=config_proto)

        # graph_defファイルを読み込んでデフォルトグラフにします。
        with tf.gfile.FastGFile(graph_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        graph = tf.get_default_graph()
        # 入力のtf.placeholderを取得します
        self.images_placeholder = graph.get_tensor_by_name(
            '%s:0' % input_node_name)

        # 推論オペレーションを取得します
        self.out_trs, self.out_tr_name = [], out_tr_names
        if type(out_tr_names) != list:
            out_tr_names = [out_tr_names]
        for out_tr_name in out_tr_names:
            out_tr = graph.get_tensor_by_name('%s:0' % out_tr_name)
            self.out_trs.append(out_tr)
        self.inf_type = 'normal'

    def inference(self, src_img, out_tr_indx=0):
        # 画像を読み込んでサイズ1のミニバッチの形式にします。
        src_img = np.expand_dims(src_img, axis=0).astype(np.float32)

        # 入力画像をplaceholderに仕込んで推論オペレーションを実行します。
        # feed_dict = {self.images_placeholder: src_img,
        #               self.pred_thresh: [0.5]}
        feed_dict = {self.images_placeholder: src_img}
        start = time.time()
        out_img = self.sess.run(self.out_trs[out_tr_indx],
                                feed_dict=feed_dict)
        duration = time.time() - start

        return out_img[0, :, :, 1], duration


def convertFnameDomains(from_domain, to_domain, fname):
    """
    Converts filenames in different domains
    (ex. ground truth imagename to original imagename)

    Available domain names:
    "original", "original2", "ground_truth", "annotated", "unknown", "output",
    "output_heatmap"

    Usage Example:
    convertFnameDomains(original, ground_truth, a.jpg)
    -> returns "a_gt.png"
    """
    domain_dict = {'jpg': '.jpg', 'jpeg': '.jpeg',
                   'ground_truth': '_gt.png', 'png': '.png',
                   'original': '.png'}

    from_keyword = (domain_dict[from_domain]
                    if from_domain in domain_dict
                    else '_%s.png' % from_domain)

    to_keyword = (domain_dict[to_domain]
                  if to_domain in domain_dict
                  else '_%s.png' % to_domain)

    if from_domain == 'original':
        fname = os.path.splitext(fname)[0] + '.png'
    fname = fname.replace(from_keyword, to_keyword)

    return fname


def create_argument_parser():
    # 学習モデルのグラフを指定します
    # 入出力がテンソルの名前も含めて同一仕様であれば差し替えもできます
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-graphdef', type=str,
                        default=DEFAULT_INPUT_GRAPHDEF)
    return parser


def add_application_arguments(parser):
    # 変換する画像の指定
    parser.add_argument('imagefiles', nargs='+', type=str)
    return parser


def main(unused_argv):
    model = DL_Model(FLAGS.graph_dir_str, 'ImageTensor', 'SemanticPredictions')
    fnames = sorted(os.listdir(FLAGS.img_dir))
    for fname in fnames:
        print(fname)
        img = cv2.imread(os.path.join(FLAGS.img_dir, fname))
        out_img, duration = model.inference(img)
        cv2.imwrite(os.path.join(FLAGS.out_dir,
                    convertFnameDomains('original', 'output', fname)),
                    out_img * 255)


if __name__ == '__main__':
    flags.mark_flag_as_required('graph_dir_str')
    flags.mark_flag_as_required('img_dir')
    flags.mark_flag_as_required('out_dir')
    tf.app.run()
