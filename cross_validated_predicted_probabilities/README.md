# Cross Validated Predicted Probabilties for each of the ten datasets

Each file is an `numpy.array(dtype=np.float16)` of shape `(num_test_examples x num_classes)`. We DO **NOT** quantize predicted probabilities to`np.float16` to reduce file size. Quantization won't work with confident learning / cleanlab because label error ordering depends on the **rank** of the predicted probabilities. If you have a lot of examples with a probability near 1, then you will lose the ranking over those examples if you quantize. Thus, we have to upload the full `float64` file size.

Predicted probabilities are computed out of sample using cross validation. In cases, where the dataset has a seperate train and test set, we first pre-trained on the train set, then fine-tuned, using cross-validation, to obtain the out-of-sample predicted probabilities on the test set.

## ImageNet and Amazon Reviews Predicted Probabilities

The `pyx.npy` predicted probabilities file for these two datasets exceeds the 100MB GitHub limit for a file. To get around this, we uploaded these pyx files in parts.

To combine the parts for ImageNet, you can run:

```python
import numpy as np
n_parts = 4
fn = 'imagenet_val_set_pyx.part{}_of_{}.npy'
parts = [np.load(fn.format(i + 1, n_parts)) for i in range(n_parts)]
# Combine the parts using np.vstack like this 
imagenet_pyx = np.vstack(parts)
```

`amazon_pyx` works similarly.

## Quickdraw Predicted Probabilities
`quickdraw_pyx.npy` is not included here because it is enormous (33GB) (the dataset has over 50 million examples). We provide `quickdraw_pyx.npy` as its own release here: https://github.com/cgnorthcutt/label-errors/releases/tag/quickdraw-pyx-v1 

Although it affects confident learning's ability to rank, we quantize the `quickdraw_pyx.npy` to `np.float16` to make the size more manageable.
