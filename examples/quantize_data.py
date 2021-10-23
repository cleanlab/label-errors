
# Copyright (c) 2021-2060 Curtis G. Northcutt
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
Script used to quantize (reduce file size of) the data stored at:
* label-errors/cross_validated_predicted_labels/
* label-errors/cross_validated_predicted_probabilities/
* label-errors/original_test_labels/
"""


import os
import numpy as np


path = "/home/cgn/cgn/label-errors/cross_validated_predicted_labels/"
# quantize the stored predictions so they are smaller using uint16
for f in os.listdir(path):
    if '.npy' != f[-4:]:
        continue
    print(f)
    pred = np.load(path + f, allow_pickle=True)
    if type(pred[0]) == np.ndarray:
        # Handle multi-label case differently
        pred_quantized = np.asarray([r.astype(np.uint16) for r in pred])
        assert all([all(pred[i] == r) for i, r in enumerate(pred_quantized)])
        assert pred_quantized[0].dtype == np.uint16
        np.save(path + f, pred_quantized)
    else:
        pred_quantized = pred.astype(np.uint16)
        assert all(pred == pred_quantized)
        np.save(path + f, pred_quantized)


path = "/home/cgn/cgn/label-errors/cross_validated_predicted_probabilities/"
# quantize the predicted probabilities so they are smaller using uint16
for f in os.listdir(path):
    if 'pyx.npy' != f[-7:] or f == 'quickdraw_pyx.npy':
        continue
    print(f)
    pyx = np.load(path + f)
    pyx_quantized = pyx.astype(np.float16)
    np.save(path + f, pyx_quantized)


path = "/home/cgn/cgn/label-errors/original_test_labels/"
# quantize the stored labels so they are smaller using uint16
for f in os.listdir(path):
    if '.npy' != f[-4:]:
        continue
    print(f)
    s = np.load(path + f, allow_pickle=True)
    if type(s[0]) == np.ndarray:
        # Handle multi-label case differently
        s_quantized = np.asarray([r.astype(np.uint16) for r in s])
        assert all([all(s[i] == r) for i, r in enumerate(s_quantized)])
        assert s_quantized[0].dtype == np.uint16
        np.save(path + f, s_quantized)
    else:
        s_quantized = s.astype(np.uint16)
        assert all(s == s_quantized)
        np.save(path + f, s_quantized)



