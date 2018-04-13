import os
import urllib.request
import zipfile
import numpy as np
from numpy import random
from scipy.misc import imread, imresize
import h5py
from tensorflow.examples.tutorials.mnist import input_data
from shutil import rmtree


def download_and_load_mnist(test_size=10000, val_size=5000, r_seed=None):
    """
    Downloads and shuffles MNIST images.

    :param test_size: chosen size of test set.
    :param val_size: chosen size of validation set.
    :param r_seed: random seed if reproducable sets is wanted.
    :return: training data with labels and test data with labels.
    """

    # Download MNIST from Tensorflow and assign to variables.
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, seed=r_seed, validation_size=0)

    train_val_data = mnist.train.images
    train_val_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mass = np.concatenate((train_val_data, test_data), axis=0)
    mass_labels = np.concatenate((train_val_labels, test_labels), axis=0)

    # Divide into training, validation and test sets.
    test_data = mass[:test_size]
    test_labels = mass_labels[:test_size]
    val_data = mass[test_size:test_size+val_size]
    val_labels = mass_labels[test_size:test_size+val_size]
    train_data = mass[test_size+val_size:]
    train_labels = mass_labels[test_size+val_size:]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def download_omniglot(target_dir = 'OMNIGLOT_data/'):
    """
    Download Omniglot alphabets from https://github.com/brendenlake/omniglot/.

    :param target_dir: target directory of download.
    :return: Nothing. The function only saves the files to target directory.
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
        print("Downloading omniglot part 1(2).")
        urllib.request.urlretrieve(origin_eval, target_dir + 'images_evaluation.zip')
        with zipfile.ZipFile(target_dir + "images_evaluation.zip", "r") as zRef:
            zRef.extractall(target_dir)
    if not os.path.isdir(target_dir + 'images_background'):
        print("Downloading omniglot part 2(2).")
        urllib.request.urlretrieve(origin_back, target_dir + 'images_background.zip')
        with zipfile.ZipFile(target_dir + "images_background.zip", "r") as zRef:
            zRef.extractall(target_dir)

    # Remove Omniglot images too similar to Mnist
    remove_letters_1 = {"Anglo-Saxon_Futhorc": [11, 21],
                      "Arcadian": [1, 12],
                      "Armenian": [6, 10, 12, 14, 19, 21, 23, 27, 33, 34, 37, 40],
                      "Asomtavruli_(Georgian)": [5, 6, 10, 16, 18, 23, 24, 25, 29, 35, 40],
                      "Balinese": [5],
                      "Bengali": [10],
                      "Blackfoot_(Canadian_Aboriginal_Syllabics)": [7, 8, 9, 10, 11, 12, 13, 14],
                      "Burmese_(Myanmar)": [6, 11, 13, 19, 20, 30],
                      "Cyrillic": [2, 9, 16, 24, 25, 28, 30],
                      "Early_Aramaic": [2, 3, 4, 6, 10, 12, 14, 16, 17, 20],
                      "Futurama": [14],
                      "Grantha": [5, 24, 36],
                      "Greek": [9, 15],
                      "Gujarati": [19, 25, 26, 27, 39, 41],
                      "Hebrew": [4, 6, 10, 12, 15, 20],
                      "Japanese_(hiragana)": [39],
                      "Korean": [1, 5, 10, 12, 21, 22, 27, 33, 37],
                      "Latin": [2, 7, 9, 12, 15, 17, 19, 26],
                      "Malay_(Jawi_-_Arabic)": [1, 29, 36, 37, 38],
                      "Mkhedruli_(Georgian)": [11, 14, 17, 22, 30, 37, 40],
                      "N_Ko": [1, 2, 9, 11, 12, 14, 21, 22, 26, 27, 33],
                      "Ojibwe_(Canadian_Aboriginal_Syllabics)": [4, 5, 6, 10, 12],
                      "Sanskrit": [21, 26],
                      "Syriac_(Estrangelo)": [7, 12, 19, 21],
                      "Tagalog": [3],
                      "Tifinagh": [1, 32, 34, 37, 42, ]}

    remove_letters_2 = {"Angelic": [14, 19],
                        "Atemayar_Qelisayer": [1, 5, 9, 16],
                        "Aurek-Besh": [4, 9, 15, 18],
                        "Avesta": [2, 4, 9, 11, 13, 15, 18, 21, 26],
                        "Ge_ez": [16, 20],
                        "Glagolitic": [6, 20, 21, 45],
                        "Gurmukhi": [18, 26, 30, 35],
                        "Malayalam": [15, 25],
                        "Manipuri": [22, 26, 27, 34],
                        "Mongolian": [6, 8, 9, 12, 13, 14],
                        "Old_Church_Slavonic_(Cyrillic)": [2, 9, 12, 17, 23, 27, 28],
                        "Oriya": [24],
                        "Syriac_(Serto)": [6],
                        "Tengwar": [1, 13],
                        "Tibetan": [15],
                        "ULOG": [9]}

    # Check if folders been tampered with. Else remove.
    if os.path.isfile(target_dir + 'images_background/Anglo-Saxon_Futhorc/character11'):
        for alphabet in remove_letters_1:
            for letter in remove_letters_1[alphabet]:
                if letter < 10:
                    letter = '0' + str(letter)
                    rmtree(target_dir + 'images_background/' + alphabet + '/character' + letter)
                else:
                    rmtree(target_dir + 'images_background/' + alphabet + '/character' + str(letter))
        for alphabet in remove_letters_2:
            for letter in remove_letters_1[alphabet]:
                if letter < 10:
                    letter = '0' + str(letter)
                    rmtree(target_dir + 'images_evaluation/' + alphabet + '/character' + letter)
                else:
                    rmtree(target_dir + 'images_evaluation/' + alphabet + '/character' + str(letter))


def load_omniglot(num_omniglot, r_seed=None):
    """
    Downloads and loads shuffled Omniglot images.

    :param num_omniglot: number of images to be returned. Same as size of MNIST test set.
    :param r_seed: random seed if reproducable sets is wanted.
    :return: Omniglot images with one-hot encoded labels. All Omniglot label sets are all-zeroed.
    """
    folder = "OMNIGLOT_data/"

    # Download the alphabets if they don't already exist.
    if not os.path.isdir(folder):
        download_omniglot(folder)

    # Gather image names
    images = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".png"):
                images.append(os.path.join(root, name))
    print("Successfully extracted {} Omniglot images.".format(len(images)))

    def load_image(in_image):
        """
        Reads an image and changes it to MNIST size (28,28,1).
        Normalizes the image before returning it.
        """
        image = imread(in_image, flatten=False)
        image = imresize(image, (28, 28), interp='bicubic')
        image = (np.abs(image - 255.)/255.)
        image = image[:,:,np.newaxis]
        return image

    # Choose num_omniglot number of random images.
    if r_seed:
        random.seed(r_seed)
    np.random.shuffle(images)
    im_omni = np.empty(shape=(num_omniglot, 28, 28, 1))
    for i in range(0, num_omniglot):
        im_omni[i] = load_image(images[i])
    im_omni = np.asarray(im_omni, dtype=np.float32)

    # Create one-hot encoded label set. Basically zero-matrix.
    labels_omni = np.zeros((num_omniglot, 11), dtype=np.int32)
    labels_omni[:, 10] = 1

    return im_omni, labels_omni


def create_dataset(test_size=10000, val_size=5000, omniglot_bool=True, name_data_set='data.h5',
                   create_file=True, r_seed=None):
    """
    Creates a shuffled dataset consisting of MNIST and Omniglot (if chosen) images. Saves to a file if chosen.

    :param test_size: Number of images taken from MNIST and Omniglot each, for the test data.
    :param val_size: Number of images for the MNIST validation set.
    :param omniglot_bool: Boolean deciding if Omniglot should be used or not.
    :param name_data_set: Name of saved file.
    :param create_file: Boolean deciding if a file should be created or not.
    :param r_seed: Random seed if reproducable sets is wanted.
    :return: Variables containing training, validation and test data with labels.
    """

    # Gather MNIST data.
    train_data, train_labels, val_data, val_labels, test_data, test_labels = \
        download_and_load_mnist(test_size, val_size, r_seed)

    # Setup datasets and labels depending on OMNIGLOT dataset
    if omniglot_bool:
        # Gather OMNIGLOT data
        im_omni, labels_omni = load_omniglot(test_size, r_seed)

        # Extending data and label sets with Omniglot.
        extend = np.array([np.zeros(test_size)], dtype=np.int32)
        test_labels = np.concatenate((test_labels, extend.T), axis=1)
        test_labels = np.concatenate((test_labels, labels_omni), axis=0)
        test_data = np.concatenate((test_data, im_omni), axis=0)

        # Remove Omniglot files
        rmtree("OMNIGLOT_data/")

    # Save dataset to hdf5 filetype if wanted.
    if create_file:
        f = h5py.File(name_data_set, 'w')
        f.create_dataset("train_data", data=train_data)
        f.create_dataset("train_labels", data=train_labels)
        f.create_dataset("val_data", data=val_data)
        f.create_dataset("val_labels", data=val_labels)
        f.create_dataset("test_data", data=test_data)
        f.create_dataset("test_labels", data=test_labels)
        f.close()

    # Remove MNIST files.
    rmtree("MNIST_data/")

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def load_datasets(test_size=10000, val_size=5000, omniglot_bool=True, name_data_set='data.h5',
                  force=False, create_file=True, r_seed=None):
    """
    Loads training, validation and test sets with labels.

    :param test_size: Number of Omniglot images to be used for test set. Bounded to be max 50% of test set.
    :param val_size: Number of images for the MNIST validation set.
    :param omniglot_bool: Boolean deciding if Omniglot should be used or not.
    :param name_data_set: Name of saved file.
    :param force: Boolean which decides if existing file should be deleted and a new one created.
    :param create_file: Boolean deciding if a file should be created or not.
    :param r_seed: Random seed if reproducable sets is wanted.
    :return: Variables containing training, validation and test data with labels.
    """
    if os.path.isfile(name_data_set) and force:
        # Remove file and create new.
        print("Removing existing file '{}' and creates + loads new.".format(name_data_set))
        os.remove(name_data_set)
        train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            create_dataset(test_size, val_size, omniglot_bool, name_data_set, create_file, r_seed)
    elif os.path.isfile(name_data_set) and not force:
        # Load file.
        print("Loading existing file '{}'.".format(name_data_set))
        f = h5py.File(name_data_set, 'r')
        train_data = f['train_data'][:]
        train_labels = f['train_labels'][:]
        val_data = f['val_data'][:]
        val_labels = f['val_labels'][:]
        test_data = f['test_data'][:]
        test_labels = f['test_labels'][:]
        if test_labels.shape[1] == 10:
            print('{} :OBS: Loaded file not containing Omniglot images :OBS: {}'.format(('='*10), ('='*10)))
        else:
            print('{} :OBS: Loaded file contains {} Omniglot images :OBS: {}'.format(('='*10),
                                                                                     len(test_labels)/2, ('='*10)))
        f.close()
    else:
        print("Creating and loading new file '{}'.".format(name_data_set))
        train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            create_dataset(test_size, val_size, omniglot_bool, name_data_set, create_file, r_seed)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def load_confusion_matrix(predictions, labels):
    """
    Builds confusion matrices.

    :param predictions: Predicted labels for test set.
    :param labels: True labels for test set.
    :return: One full confusion matrix and one showing if MNIST or Omniglot.
    """
    predictions = np.asarray(predictions, dtype=np.int)

    # Reverse one-hot encoding of y_test labels
    _, labels = np.where(labels == 1)

    # Full matrix
    matrix1 = np.zeros((11, 11))
    for a, p in zip(labels, predictions):
        matrix1[a][p] = np.add(np.int(1), matrix1[a][p])

    # MNIST-or-not matrix.
    matrix2 = np.zeros((2, 2))
    matrix2[0, 0] = np.sum(matrix1[:10, :10], dtype=np.int)
    matrix2[0, 1] = np.sum(matrix1[:10, 10], dtype=np.int)
    matrix2[1, 0] = np.sum(matrix1[10, :10], dtype=np.int)
    matrix2[1, 1] = np.int(matrix1[10, 10])

    return matrix2
