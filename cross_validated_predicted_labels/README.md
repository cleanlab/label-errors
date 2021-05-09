# Cross Validated Predictions for each of the ten datasets

Each file is an `np.array(dtype=np.uint16)` of length `num_test_examples`. In the case of AudioSet (a multi-label dataset), it is an array of lists.

The predictions for each dataset (except AudioSet) are computed like this:
```python
for f in os.listdir('path/to/label-errors/cross_validated_predicted_probabilities/):
    if 'pyx.npy' != f[-7:]:
        continue
    pyx = np.load(path + f)  # Load the predicted probabilities
    pred = pyx.argmax(axis=1)  # Take the argmax prediction per example across classes
    pred = pred.astype(dtype=np.uint16)  # Quantization to reduce file size
    np.save(path+f.replace('pyx', 'pyx_argmax_predicted_labels'), pred)  # Save result
```

These predictions are not intended to be state-of-the-art, or even reasonablly close to state-of-the-art. 
Instead, they are baselines so we can get an idea of what our models guess for each label error we find. 
We use [cleanlab](https://github.com/cgnorthcutt/cleanlab) to automatically *find* the label errors, but to *fix* the labels, 
we used a mechanical turk validation experiment. Details in [our paper](https://arxiv.org/abs/2103.14749).

If you want really accurate predictions, I recommend first cleaning the train set using [cleanlab](https://github.com/cgnorthcutt/cleanlab), then
pre-train on the cleaned train set using a state-of-the-art model, and then fine-tune on the test set using cross-validation to predict out-of-sample, 
using as many folds as you can afford (in terms of time/computation). 

## AudioSet (special case because its multi-label)

The AudioSet predictions are relased as an `np.array<np.array(dtype=np.uint16)>`(a numpy array of numpy arrays) because AudioSet is multi-label.

Because AudioSet is multi-label, we use the following procedure to find the predictions based on whether the softmax
output, for each class, for each example, exceeds a threshold for that class. We select the thresholds that maximize
the f1 accuracy on the original labels. Change `visualize = False` to `visualize = True` if you are working in an
environment with plotting capabilities.

```python
# Code assumes you have labels (formatted as a one hot encoded matrix) and pyx (predicted probabilities)

import numpy as np

visualize = False

# Determine threshold to use for estimating predictions (and accuracy metrics.)
thresholds = np.arange(0.1, 0.6, .05)
f1s = [f1_score(labels_one_hot, (pyx > threshold).astype(np.uint8), average='micro') for threshold in thresholds]
threshold = thresholds[np.argmax(f1s)]

if visualize:
    from matplotlib import pyplot as plt
    plt.figure(figsize = (10, 5))
    plt.plot(thresholds, f1s)
    plt.vlines([threshold], ymin = 0, ymax = 1.)
    sns.despine()
    plt.show()
    print("The threshold that maximizes F1 score on the ROC-precision-recall curve: {:.2f}".format(threshold))

pred = (pyx > threshold).astype(np.uint8)
# Some examples may have no prediction that exceeds the threshold, choose argmax in this case.
for i in range(len(pred)):
    if sum(pred[i]) == 0:        
        pred[i][np.argmax(pyx[i])] = 1
print('Hamming loss of pyx:', np.round(hamming_loss(labels_one_hot, pred), 4))
precision = np.mean([(pyx[i].argmax() in labels[i]) for i in range(len(labels))])
print('Precision (argmax label is in the ground truth labels):', precision.round(2))
```
