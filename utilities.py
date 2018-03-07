import os
import urllib.request
import zipfile
import numpy as np
from scipy.misc import imread, imresize
import h5py
from tensorflow.examples.tutorials.mnist import input_data
from shutil import rmtree


def download_and_load_mnist(test_size = 10000):
    """
    Load MNIST. Taken from https://github.com/tensorflow
    :return: numpy array
    """

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    idx = np.random.permutation(len(eval_data))
    eval_data = eval_data[idx]
    eval_labels = eval_labels[idx]

    return train_data, train_labels, eval_data, eval_labels


def download_omniglot(target_dir):
    """
    Download Omniglot alphabets.
    :parameter:
    target_dir: target directory of download.

    :return:
    """

    os.makedirs(target_dir, exist_ok=True)
    origin_eval = (
        "https://github.com/brendenlake/omniglot/"
        "raw/master/python/images_evaluation.zip"
    )
    origin_back = (
        "https://github.com/brendenlake/omniglot/"
        "raw/master/python/images_background.zip"
    )
    if not os.path.isdir(target_dir + 'images_evaluation'):
        print("Downloading omniglot part 1(2) from github/brendenlake/omniglot")
        urllib.request.urlretrieve(origin_eval, target_dir + 'images_evaluation.zip')
        with zipfile.ZipFile(target_dir + "images_evaluation.zip", "r") as zRef:
            zRef.extractall(target_dir)
    if not os.path.isdir(target_dir + 'images_background'):
        print("Downloading omniglot part 2(2) from github/brendenlake/omniglot")
        urllib.request.urlretrieve(origin_back, target_dir + 'images_background.zip')
        with zipfile.ZipFile(target_dir + "images_background.zip", "r") as zRef:
            zRef.extractall(target_dir)


def load_omniglot(num_omniglot):
    """
    The dataset contains 50 alphabets in total: 30 as training and 20 as evaluation. These are merged into one.

    :param num_omniglot ; the number of omniglot images to be saved for testing.
    :return: the function creates an hdf5 file containing num_omniglot number of randomly sampled omniglot letters
             plus the corresponding label=10.
    """
    folder = "OMNIGLOT_data/"

    # Download the alphabets if they don't already exist
    if not os.path.isdir(folder):
        download_omniglot(folder)

    # Gather image names
    images = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".png"):
                images.append(os.path.join(root, name))
    print("Successfully extracted omniglot.")

    def load_image(im):
        """
        Reads an image and changes it to MNIST size (28,28,1).
        Normalizes the image before returning it.
        """
        image = imread(im, flatten=True)
        image = imresize(image, (28, 28, 1))
        image = image.reshape(-1)
        image = np.multiply(image, (255.0 / image.max()), casting='unsafe')
        return image

    # Choose num_omniglot number of random images.
    print("Total number of Omniglot images': {}".format(len(images)))
    np.random.shuffle(images)
    im_omni = []
    for im in images[:num_omniglot]:
        im_omni.append(load_image(im))
    im_omni = np.asarray(im_omni)
    print("Number of chosen Omniglot images: {}".format(len(im_omni)))

    # Create one-hot label set
    values = np.array([10] * num_omniglot)
    labels_omni = np.eye(11)[values]

    return im_omni, labels_omni


def create_dataset(test_size=10000, omniglot_bool=True, name_data_set='data.h5'):
    """
    Download MNIST and OMNIGLOT data and concatenate OMNIGLOT images to MNIST evaluation images.

    :return:
    hdf5 file containing data sets
    """
    if omniglot_bool and (test_size > 10000 or test_size < 1):
        raise ValueError("Size of test set must be between 0 and 10000 when using Omniglot.")

    # Gather MNIST data
    train_data, train_labels, eval_data, eval_labels = download_and_load_mnist()

    # Setup datasets and labels depending on OMNIGLOT dataset
    if omniglot_bool:
        # Gather OMNIGLOT data
        im_omni, labels_omni = load_omniglot(test_size)

        # Bit of fixing with arrays.
        extend_labels = np.array([[0] * eval_labels.shape[0]])
        eval_labels = np.append(eval_labels, extend_labels.T, axis=1)
        eval_labels = np.concatenate((eval_labels, labels_omni), axis=0)
        eval_labels = np.int32(eval_labels)

        eval_data = np.concatenate((eval_data, im_omni), axis=0)

    # Save dataset to hdf5 filetype.
    f = h5py.File(name_data_set, 'w')
    f.create_dataset("train_data", data=train_data)
    f.create_dataset("train_labels", data=train_labels)
    f.create_dataset("eval_data", data=eval_data)
    f.create_dataset("eval_labels", data=eval_labels)
    f.close()

    rmtree("MNIST_data/")
    rmtree("OMNIGLOT_data/")
