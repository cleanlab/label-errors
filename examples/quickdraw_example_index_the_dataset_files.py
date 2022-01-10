
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
A working code example to index the Quickdraw data given a global index from labelerrors.com or from the corrected test sets in this repo.

Download the QuickDraw dataset here: https://github.com/cleanlab/label-errors/releases/tag/quickdraw-pyx-v1

Use cleanlab to find label errors in the dataset: https://github.com/cleanlab/cleanlab

Run this code in a Google Colab, or Jupyter Notebook, if you want to visualize the errors via matplotlib.
"""

import os
import numpy as np

# !!!CHANGE THIS TO YOUR DIRECTORY WHERE YOU DOWNLOADED THE NUMPY BITMAPS
QUICKDRAW_NUMPY_BITMAP_DIR = '/datasets/datasets/quickdraw/numpy_bitmap/'

# !!!CHANGE THESE TO WHERE YOU CLONE https://github.com/cleanlab/label-errors
# Load predictions and indices of label errors
pred = np.load('/datasets/cgn/pyx/quickdraw/pred__epochs_20.npy')
le_idx = np.load('/datasets/cgn/pyx/quickdraw/label_errors_idx__epochs_20.npy')

display_predicted_label = False  # Set to true to print the predicted label.

def fetch_class_counts(numpy_bitmap_dir):
    # Load class counts for QuickDraw dataset.
    class_counts = []
    for i, f in enumerate(sorted(os.listdir(numpy_bitmap_dir))):
        loc = os.path.join(numpy_bitmap_dir, f)
        with open(loc, 'rb') as rf:
            line = rf.readline()
            cnt = int(line.split(b'(')[1].split(b',')[0])
            class_counts.append(cnt)
    print('Total number of examples in QuickDraw npy files: {:,}'.format(
        sum(class_counts)))
    assert sum(class_counts) == 50426266
    return class_counts

# Get the number of examples in each class/file based on the numpy bitmap files.
class_counts = fetch_class_counts(QUICKDRAW_NUMPY_BITMAP_DIR)
# We'll use the cumulative sum of the class counts to map the 
#    global index to index in each file.

counts_cumsum = np.cumsum(class_counts)

# Get the list of all class names sorted corresponding to their numerical label
# make sure you sort the filenames using sorted!
label2name = [z[:-4] for z in sorted(os.listdir(QUICKDRAW_NUMPY_BITMAP_DIR))]


# Let's look at an example from the label errors site.
# https://labelerrors.com/static/quickdraw/44601012.png


# !!!CHANGE THIS TO THE ID OF ANY QUICKDRAW ERROR ON https://labelerrors.com
# You can find the id by right-clicking the image, and copying the image url
idx = 44601012
# The true class of this image is 'angel', i.e., class 7
# The given class of this image is 'triangle', i.e., class 324
if idx >= counts_cumsum[-1]:
    raise ValueError('index {} must be smaller than size of dataset {}.'.format(
        idx, counts_cumsum[-1]))

# !!!The next 5 lines of code are IMPORTANT.
# Here's how you map the global index (idx) to the local index within each file.
given_label = np.argmax(counts_cumsum > idx)
if given_label > 0:
    # local index = global index - the cumulative items in the previous classes
    local_idx = idx - counts_cumsum[given_label - 1]
else:
    # Its class 0, in the first npy file, so the local index == global index
    local_idx = idx

# Check the given label matches the corresponding class name
print('\nQuickdraw Given label: {} (label id: {})'.format(
    label2name[given_label], given_label))
if display_predicted_label:
    print('Pred label: {} (label id: {})'.format(
        label2name[pred[idx]], pred[idx]))

# Visualize the example
from matplotlib import pyplot as plt
plt.imshow(
    256 - np.load(QUICKDRAW_NUMPY_BITMAP_DIR + '{}.npy'.format(
        label2name[given_label]),
    )[local_idx].reshape(28, 28),
    interpolation='nearest',
    cmap='gray',
)
plt.show()
print('^ should match https://labelerrors.com/static/quickdraw/44601012.png')
