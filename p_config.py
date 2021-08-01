# import the necessary packages
import os
import pathlib

# initialize the path to the *original* input directory of images
#ORIG_INPUT_DATASET = "Food-5K"

# initialize the base path to the *new* directory that will contain 
# our images after computing the training and testing split
BASE_PATH = "flower_photos"
RANDOM_SEED = 2021
EPOCHS = 5
FLOWERS_ORIG_DIR = pathlib.Path.cwd() / 'flower_photos'

# define the names of the training, testing, and validation
# directores
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# init the list of class label names
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# set the batch size
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"

# The '.h5' extension indicates that the model should be saved to HDF5.
MODEL_PATH = os.path.sep.join(["output", "model.h5"])




