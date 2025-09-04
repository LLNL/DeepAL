import re

from os import getcwd
from os.path import exists, join, basename

import glob

from torch import load, save


def load_model(checkpoint_dir, model_filename, model):
    if exists(join(checkpoint_dir, model_filename)):
        # print ("Reloading model state dict")
        state_dict = load(join(checkpoint_dir, model_filename), map_location=model.device)
        model.load_state_dict(state_dict)
    return model


def save_model(checkpoint_dir, model_filename, model,
               rng_filename=None, rng=None):
    save(model.state_dict(), join(checkpoint_dir, model_filename))
    if (rng_filename is not None) and (rng is not None):
        save(rng, join(checkpoint_dir, rng_filename))


def find_latest_checkpoint(directory_path=getcwd()):
    # Using glob to filter files directly by pattern
    files = [basename(f) for f in glob.glob(f"{directory_path}/model_step_*.pt")]

    # Get the file with the maximum number in its name
    latest_file = max(files, key=lambda x: int(re.search(r'model_step_(\d+)\.pt', x).group(1)), default=None)
    # Create a list of tuples (number, file) for valid files
    valid_files = [(int(re.search(r'model_step_(\d+)\.pt', file).group(1)), file) 
                   for file in files if re.search(r'model_step_(\d+)\.pt', file)]
    # Find the file with the maximum number
    latest_step, latest_file = max(valid_files, key=lambda x: x[0], default=(-1,"model_init.pt"))
    return latest_step, latest_file
