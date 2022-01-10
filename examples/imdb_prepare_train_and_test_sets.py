
# coding: utf-8

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
Prepare the IMDB dataset into two python dicts (text and labels),
each with keys keys ['train', 'test'].

Download the IMDB dataset from: https://ai.stanford.edu/~amaas/data/sentiment/

Use cleanlab to find label errors in the dataset: https://github.com/cleanlab/cleanlab 
"""

import os
import numpy as np

# !!!CHANGE THIS TO THE LOCATION WHERE YOU EXTRACTED THE IMDB DATASET
data_dir = "/datasets/datasets/aclImdb/"

# This stores the data as dict with keys ['train', 'test']
text = {}
# This stores the labels as a dict with keys ['train', 'test']
labels = {}
for dataset in ['train', 'test']:
    text[dataset] = []
    dataset_dir = data_dir + dataset + '/'
    for i, fn in enumerate(os.listdir(dataset_dir + "neg/")):
        with open(dataset_dir + "neg/" + fn, 'r') as rf:
            text[dataset].append(rf.read())
    labels[dataset] = np.zeros(i + 1)
    for i, fn in enumerate(os.listdir(dataset_dir + "pos/")):
        with open(dataset_dir + "pos/" + fn, 'r') as rf:
            text[dataset].append(rf.read())
    labels[dataset] = np.concatenate([labels[dataset], np.ones(i + 1)]).astype(int)
