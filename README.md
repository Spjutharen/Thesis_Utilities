## Utilities for Anomaly/Novelty and Adversarial Detection Thesis Project
Created for thesis project 'Input Verification for Neural Networks'. The script 'utilities.py' downloads MNIST and Omniglot images and returns training, validation and test data with labels. 

## Guide
To download the datasets, one calls

```python
train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            create_dataset(test_size, omniglot_bool, name_data_set, per_train, create_file, r_seed)
```

where the several parameters are explained below.

* test_size: 		Number of Omniglot images to be used for test set. Bounded to be max 50% of test set.
* omniglot_bool: 	Boolean deciding if Omniglot should be used or not.
* name_data_set: 	Name of saved file.
* force: 			Boolean which decides if existing file should be deleted and a new one created.
* per_train: 		Percent of train_data that should be used for training. Remaining is used as validation set.
* create_file: 		Boolean deciding if a file should be created or not.
* r_seed: 			Random seed if reproducable sets is wanted.
