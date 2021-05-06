# Cross Validated Predictions for each of the ten datasets

Each file is a numpy array of length `num_test_examples`. In the case of AudioSet (a multi-label dataset), it is an array of lists.

The predictions for each dataset are computed like this:
```python
for f in os.listdir('path/to/label-errors/cross_validated_predicted_probabilities/):
    if 'pyx.npy' != f[-7:]:
        continue
    pyx = np.load(path + f)  # Load the predicted probabilities
    pred = pyx.argmax(axis=1)  # Take the argmax prediction per example across classes
    np.save(path+f.replace('pyx', 'pyx_argmax_predicted_labels'), pred)  # Save result
```

These predictions are not intended to be state-of-the-art, or even reasonablly close to state-of-the-art. 
Instead, they are baselines so we can get an idea of what our models guess for each label error we find. 
We use [cleanlab](https://github.com/cgnorthcutt/cleanlab) to automatically *find* the label errors, but to *fix* the labels, 
we used a mechanical turk validation experiment. Details in [our paper](https://arxiv.org/abs/2103.14749).

If you want really accurate predictions, I recommend first cleaning the train set using [cleanlab](https://github.com/cgnorthcutt/cleanlab), then
pre-train on the cleaned train set using a state-of-the-art model, and then fine-tune on the test set using cross-validation to predict out-of-sample, 
using as many folds as you can afford (in terms of time/computation). 
