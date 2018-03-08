## Utilities for Anomaly/Novelty and Adversarial Detection Thesis Project
Created for thesis project *Input Verification for Neural Networks*. The script downloads MNIST and Omniglot images and returns training, validation and test data with labels. 

#### Pref
The MNIST test set contains 10k images, wheras the Omniglot set contains 32460. The number of Omniglot images in the test set has been constrained to the same amount as the MNIST test set, set by `test_size`.

## Guide
To load the datasets, one calls

```python
train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            create_dataset(test_size=10000, omniglot_bool=True, name_data_set='data.h5', force=False, per_train=0.9, create_file=True, r_seed=None)
```

where the several parameters are explained below.

### Parameters
* *test_size:* 		Number of Omniglot images to be used for test set. Bounded to be max 50% of test set.
* *omniglot_bool:* 	Boolean deciding if Omniglot should be used or not.
* *name_data_set:* 	Name of saved file.
* *force:* 			Boolean which decides if existing file should be deleted and a new one created.
* *per_train:* 		Percent of train_data that should be used for training. Remaining is used as validation set.
* *create_file:* 		Boolean deciding if a file should be created or not.
* *r_seed:* 			Random seed if reproducable sets is wanted.

### Returns
* *train_data:* [per_train*55000, 28, 28, 1]
* *train_labels:* [per_train*55000, 1]
* *val_data:* [(1-per_train)*55000, 28, 28, 1]
* *test_data:* [2*test_size, 28, 28, 1]
