# Original Test Set Labels of the Ten Datasets

Each file is an `np.array(dtype=np.uint16)` of length `num_test_examples`. In the case of AudioSet (a multi-label dataset), it is an `np.array` of `np.arrays<np.uint16>`. All labels are quantized to type `np.uint16` to reduce file size.

The order of examples in each `original_labels.npy` file corresponds exactly to the order of the corresponding dataset files in:
* [label-errors/cross_validated_predicted_labels/](https://github.com/cleanlab/label-errors/tree/main/cross_validated_predicted_labels)
* [label-errors/cross_validated_predicted_probabilities/](https://github.com/cleanlab/label-errors/tree/main/cross_validated_predicted_probabilities)


