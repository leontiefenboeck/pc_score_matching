import numpy as np
import os
import tempfile
import urllib.request
import utils
import shutil
import gzip
import subprocess
import csv
import scipy.io as sp


def maybe_download(directory, url_base, filename):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        return False

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base + filename
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading {} to {}'.format(url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return True


def maybe_download_mnist(data_dir):
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        # if not maybe_download('data/mnist', 'https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=', file):
        #     continue
        print('unzip data/mnist/{}'.format(file))
        filepath = os.path.join(data_dir, file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_mnist():
    """Load MNIST"""

    data_dir = os.path.join('data', 'mnist')

    # maybe_download_mnist(data_dir)

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def maybe_download_fashion_mnist(data_dir):
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        # if not maybe_download('../data/fashion-mnist', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file):
        #     continue
        print('unzip ../data/fashion-mnist/{}'.format(file))
        filepath = os.path.join(data_dir, file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_fashion_mnist():
    """Load fashion-MNIST"""

    data_dir = os.path.join('data', 'fashion-mnist')

    # maybe_download_fashion_mnist(data_dir)

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def maybe_download_svhn():
    svhn_files = ['train_32x32.mat', 'test_32x32.mat', "extra_32x32.mat"]
    for file in svhn_files:
        maybe_download('../data/svhn', 'http://ufldl.stanford.edu/housenumbers/', file)


def load_svhn(dtype=np.uint8):
    """
    Load the SVHN dataset.
    """

    maybe_download_svhn()

    data_dir = '../data/svhn'

    data_train = sp.loadmat(os.path.join(data_dir, "train_32x32.mat"))
    data_test = sp.loadmat(os.path.join(data_dir, "test_32x32.mat"))
    data_extra = sp.loadmat(os.path.join(data_dir, "extra_32x32.mat"))

    train_x = data_train["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    train_labels = data_train["y"].reshape(-1)

    test_x = data_test["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    test_labels = data_test["y"].reshape(-1)

    extra_x = data_extra["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    extra_labels = data_extra["y"].reshape(-1)

    return train_x, train_labels, test_x, test_labels, extra_x, extra_labels


if __name__ == '__main__':
    print('Downloading dataset -- this might take a while')

    print()
    print('MNIST')
    maybe_download_mnist()

    print()
    print('fashion MNIST')
    maybe_download_fashion_mnist()

    print()
    print('SVHN')
    maybe_download_svhn()
