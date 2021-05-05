# coding: utf-8

# Copyright (c) 2021-2060 Curtis G. Northcutt
# This file is part of cgnorthcutt/label-errors.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cgnorthcutt/label-errors is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of
# cgnorthcutt/label-errors.

"""
Preprocess the tfrecord AudioSet feature embeddings into numpy data files.

Resources used:
1. https://github.com/tensorflow/models/tree/master/research/audioset
2. https://research.google.com/audioset/download.html
3. https://github.com/audioset/ontology
"""


import argparse
import os
import numpy as np
import tensorflow as tf  # version 1.15.4
import multiprocessing
import tqdm
import pickle
from keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--audioset-dir', metavar='AUDIOSET_DIR',
    help='Specify path to ../audioset/audioset_v1_embeddings/',
)


def read_data(path, include_times=False):
    result = [[], [], []]
    if include_times:
        result += [[], []]
    for example in tf.python_io.tf_record_iterator(path):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['video_id'].bytes_list.value[
            0].decode(encoding='UTF-8')
        label = tf_example.features.feature['labels'].int64_list.value
        if include_times:
            result[3].append(tf_example.features.feature[
                                 'start_time_seconds'].float_list.value)
            result[4].append(tf_example.features.feature[
                                 'end_time_seconds'].float_list.value)
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        tf_feature = tf_seq_example.feature_lists.feature_list[
            'audio_embedding'].feature
        n_frames = len(tf_feature)
        audio_frames = []
        # Iterate through frames.
        for i in range(n_frames):
            hexembed = tf_feature[i].bytes_list.value[0].hex()
            arrayembed = [int(hexembed[i:i + 2], 16) for i in
                          range(0, len(hexembed), 2)]
            audio_frames.append(arrayembed)
        result[0].append(vid_id)
        result[1].append(list(label))
        result[2].append(np.stack(audio_frames).astype(np.uint8))
    return result


def pad(feature_matrix, maxlen=10):
    return pad_sequences(feature_matrix.T, maxlen=maxlen).T.astype(np.uint8)


def preprocess_data(path, prefix='bal_train'):
    fns = [path + fn for fn in os.listdir(path)]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = list(tqdm.tqdm(p.imap(read_data, fns), total=len(fns)))

    print('\nAll files read in. Now post-processing.')
    video_ids = [v for r in results for v in r[0]]
    labels = [l for r in results for l in r[1]]
    features = [f for r in results for f in r[2]]
    del results  # Free memory
    # Make all inputs exactly the same shape.
    print("Padding with 0 to make all features shape (10,128) of type uint8.")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        features = list(tqdm.tqdm(p.imap(pad, features), total=len(features)))

    print('Saving pickled results.')
    with open(prefix + '_features.p', 'wb') as wf:
        pickle.dump(features, wf, pickle.HIGHEST_PROTOCOL)
    with open(prefix + '_video_ids.p', 'wb') as wf:
        pickle.dump(video_ids, wf, pickle.HIGHEST_PROTOCOL)
    with open(prefix + '_labels.p', 'wb') as wf:
        pickle.dump(labels, wf, pickle.HIGHEST_PROTOCOL)

    print('Preprocessing complete.')


def main(audioset_dir):
    for kind in ["eval", "bal_train", "unbal_train"]:
        preprocess_data(audioset_dir + kind + "/", prefix=kind)


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    if arg_parser.audioset_dir is None:
        parser.error("Specify the path to the audioset embeddings "
                     "directory.\nFor example, if the data is stored in "
                     "'/datasets/audioset/audioset_v1_embeddings/' "
                     "you should call this script like this:\npython "
                     "audioset_preprocessing.py --audioset-dir "
                     "'/datasets/audioset/audioset_v1_embeddings/''")
    main(audioset_dir=arg_parser.audioset_dir)
