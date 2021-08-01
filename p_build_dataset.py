import numpy as np
import os
import sys
import stat
import tarfile
import PIL
import collections
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import random
import shutil
import p_config as config
from six.moves import urllib
import pathlib
from uuid import uuid4

FLOWERS_ORIG_DIR = config.FLOWERS_ORIG_DIR

#RANDOM_SEED = 2021           

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

download_file = pathlib.Path.cwd() / 'flower_photos/flower_photos'
mk_folder = pathlib.Path.cwd() / 'flower_photos'
train_folder = pathlib.Path.cwd() / 'flower_photos' / 'training'
val_folder = pathlib.Path.cwd() / 'flower_photos' / 'validation'
eval_folder = pathlib.Path.cwd() / 'flower_photos' / 'evaluation'
destination_folder = train_folder

def extract(tar_url, extract_path='.'):
     print(tar_url)
     tar = tarfile.open(tar_url, 'r')
     for item in tar:
          tar.extract(item, extract_path)
          if item.name.find('.tgz') != -1:
               extract(item.name, "./" + item.name[:item.name.rfind('/')])

def download_images():
     '''If the images aren't downloaded, save them to FLOWERS_DIR'''
     if not os.path.exists(FLOWERS_ORIG_DIR):
          os.mkdir(FLOWERS_ORIG_DIR)
          DOWNLOAD_URL = dataset_url
          print(f'Downloading flower images from {dataset_url}...')
          urllib.request.urlretrieve(DOWNLOAD_URL, FLOWERS_ORIG_DIR / 'flower_photos.tgz')
          try:
               extract(FLOWERS_ORIG_DIR / 'flower_photos.tgz')
               print('Done extracting.')
               print(f'Photos are located in {FLOWERS_ORIG_DIR}')
          except:
               print("Unexpected error:", sys.exc_info()[0])
               raise
        
def make_train_valid_test_sets():
     '''Split the data into training, validation, and test sets and get label classes'''
     is_root = True
     train_examples, valid_examples, test_examples = [], [], []
     shuffler = random.Random(config.RANDOM_SEED)
     random.seed(config.RANDOM_SEED)
     for (dirname, subdirs, filenames) in tf.io.gfile.walk(FLOWERS_ORIG_DIR):
          subdirs = sorted(subdirs)
          classes = collections.OrderedDict(enumerate(subdirs))
          label_to_class = dict([(x, i) for i, x in enumerate(subdirs)])
          if not os.path.exists(train_folder):
               os.mkdir(train_folder)
          if not os.path.exists(val_folder):
               os.mkdir(val_folder)
          if not os.path.exists(eval_folder):
               os.mkdir(eval_folder)
          if is_root:
               for class_item in subdirs:
                    # move all the items to the training folder
                    source_folder = pathlib.Path(pathlib.Path.cwd() / 'flower_photos' / class_item)
                    destination_folder = train_folder
                    dest = shutil.move(str(source_folder), str(destination_folder))
                    # create category folders in eval.
                    path = pathlib.Path(pathlib.Path.cwd() / 'flower_photos' / 'evaluation' / class_item)
                    os.mkdir(path)
                    # create category folders in val.
                    path = pathlib.Path(pathlib.Path.cwd() / 'flower_photos' / 'validation' / class_item)
                    os.mkdir(path)
               
     for (dirname, subdirs, filenames) in tf.io.gfile.walk(train_folder):
          subdirs = sorted(subdirs)
          for class_item in subdirs:
               source_folder = pathlib.Path(train_folder / class_item)
               destination_folder = pathlib.Path(val_folder / class_item)
               files = os.listdir(source_folder)
               no_of_files = len(files) // 5
               
               # move to validation
               for file_name in random.sample(files, no_of_files):
                    shutil.move(os.path.join(source_folder, file_name), destination_folder)

               # move to testing
               destination_folder = pathlib.Path(eval_folder / class_item)
               files = os.listdir(source_folder)
               for file_name in random.sample(files, no_of_files):
                    shutil.move(os.path.join(source_folder, file_name), destination_folder)
        
download_images()
make_train_valid_test_sets()
exec(open("./p_extract_features.py").read())
exec(open("./p_train.py").read())