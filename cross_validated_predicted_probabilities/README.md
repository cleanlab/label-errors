# Cross Validated Predicted Probabilties for each of the ten datasets

Each file is an `numpy.array(dtype=np.float16)` of shape `(num_test_examples x num_classes)`. All predicted probabilities are quantize to `np.float16` to reduce file size.

Predicted probabilities are computed out of sample using cross validation. In cases, where the dataset has a seperate train and test set, we first pre-trained on the train set, then fine-tuned, using cross-validation, to obtain the out-of-sample predicted probabilities on the test set.

## Quickdraw predicted probabilities
`quickdraw_pyx.npy` is not included here because it is enormous (33GB) (the dataset has over 50 million examples). We provide `quickdraw_pyx.npy` as its own release here: https://github.com/cgnorthcutt/label-errors/releases/tag/quickdraw-pyx-v1
