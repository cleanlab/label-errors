
# Copyright (c) 2021-2022 Cleanlab Inc.
# This file is part of cleanlab/label-errors.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab/label-errors is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of
# cleanlab/label-errors.

"""
Provides a working example to index the examples in AudioSet so you can use corrected test set and manipulate the test dataset based on which examples you find are label errors.

You can find the label errors using https://github.com/cleanlab/cleanlab (all you need is the predicted probabilites, pyx, and the noisy labels, s).

Download the Numpy AudioSet Dataset here: https://github.com/cleanlab/label-errors/releases/tag/numpy-audioset-dataset
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

#!!! CHANGE THIS TO YOUR AUDIOSET MAIN DIRECTORY
audioset_main_dir = "/datasets/datasets/audioset/"

def row2url(d):
    '''Converts a dict-like object to a youtube URL.'''
    if type(d) == pd.DataFrame:
        return "http://youtu.be/{vid}?start={s}&end={e}".format(
            vid = d['# YTID'].iloc[0],
            s = int(d['start_seconds'].iloc[0]),
            e = int(d['end_seconds'].iloc[0]),
        )
    else:
        return "http://youtu.be/{vid}?start={s}&end={e}".format(
            vid = d['# YTID'],
            s = int(d['start_seconds']),
            e = int(d['end_seconds']),
        )
# Information about the given (potentially noisy) test labels.
test_label_info = pd.read_csv(
    audioset_main_dir + "audioset_v1_embeddings/eval_segments.csv", 
    header=2, delimiter=", ", engine='python', )
# Read in the labels that are now easily accessible from the pickle files.
labels = np.load(audioset_main_dir + "preprocessed/eval_labels.p", allow_pickle=True)
test_video_ids = np.load(audioset_main_dir + "preprocessed/eval_video_ids.p", allow_pickle=True)
labels_one_hot = MultiLabelBinarizer().fit_transform(labels)
# Get human-readable class name mapping
# label_df = pd.read_csv("/media/ssd/datasets/datasets/audioset/class_labels_indices.csv")
label_df = pd.read_csv(audioset_main_dir + "class_labels_indices.csv")
label2mid = list(label_df["mid"].values)
label2name = list(label_df["display_name"].values)
num_unique_labels = len(set([zz for z in labels for zz in z]))
# Convert list of labels for each test example to human-readable class names
# lol = list of labels, because the AudioSet test set is multi-label
y_test_lol = [[label2name[z] \
                for z in np.arange(num_unique_labels)[p.astype(bool)]] \
                for p in labels_one_hot]
# Take a look at the first few label error indices/predictions we provide
label_errors_idx = np.array([11536,  2744,  3324])
predicted_labels = dict(zip(label_errors_idx, [['Wind instrument, woodwind instrument', 'Bagpipes'], ['Singing', 'Music', 'Folk music', 'Middle Eastern music'], ['Music']]))
for idx in label_errors_idx:
    row = test_label_info[test_label_info["# YTID"] == test_video_ids[0]]
    print('\nIndex of test/eval example:', idx)
    print('YouTube URL:', row2url(row))
    print('Given Labels:', y_test_lol[idx])
    print('Pred/Guessed Labels:', predicted_labels[idx])
