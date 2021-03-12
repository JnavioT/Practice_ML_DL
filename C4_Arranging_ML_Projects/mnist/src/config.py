import os

this_file_path = os.path.abspath(__file__)
BASE_DIR =  os.path.dirname(this_file_path)
BASE_DIR2 =  os.path.dirname(BASE_DIR)
DATA_ADD = os.path.join(BASE_DIR2, "input","mnist_train_folds.csv")
MODELS_ADD = os.path.join(BASE_DIR2, "models")