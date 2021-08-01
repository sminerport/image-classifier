# Overview

To run the program simply run the `p_build_dataset.py` script, which in turn runs the `p_extract_features.py` and `p_train.py` scripts.

Alternatively, once the `p_build_dataset.py` script has been run, the `p_train.py` file can be run by itself to re-train the model and output different images if the RANDOM_SEED value is updated in the `p_config.py` file.

---

## Program scripts:
* **p_config.py:** sets configuration variables
* **p_build_dataset.py:** downloads the dataset and creates the training, validation, and evaluation splits
* **p_extract_features.py:** extracts features to CSV files
* **p_train.py:** trains the simple feedforward NN, evaluates the model, saves model ouput for future re-use, and outputs a random set of 25 images to assess the model (to output a different batch of images, updated the RANDOM_SEED in the `p_config.py` file
