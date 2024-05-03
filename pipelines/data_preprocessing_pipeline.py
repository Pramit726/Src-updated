import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'components')))

from data_preprocessing import load_datasets, preprocess_datasets

# Define global variables
img_height = 224
img_width = 224
batch_size = 32

# Call data preprocessing functions
val_ds, test_ds, train_ds = load_datasets(img_height, img_width, batch_size)


# Visualize images before preprocessing
class_names = val_ds.class_names

# Preprocess datasets
val_ds_preprocessed, test_ds_preprocessed, train_ds_preprocessed = preprocess_datasets(val_ds, test_ds, train_ds)
