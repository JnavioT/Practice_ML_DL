import os

this_file_path = os.path.abspath(__file__)
BASE_DIR =  os.path.dirname(this_file_path)
BASE_DIR2 =  os.path.dirname(BASE_DIR)

TRAINING_FILE = os.path.join(BASE_DIR2, "input","cat_train_folds.csv")
TRAINING_FILE_ADULT = os.path.join(BASE_DIR2, "input","adult_folds.csv")

MODELS_ADDRESS = os.path.join(BASE_DIR2, "models")
