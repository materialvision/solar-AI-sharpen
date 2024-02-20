import os
import numpy as np
from PIL import Image
from options.test_options import TestOptions
from models import create_model
from data import create_dataset
from util import util  # Assuming util has the necessary function to convert tensor to image

def save_image(image_tensor, image_path):
    """Save a single image to the disk."""
    image_numpy = util.tensor2im(image_tensor)  # Convert tensor to numpy array
    image_pil = Image.fromarray(image_numpy)  # Convert numpy array to PIL Image
    image_pil.save(image_path)  # Save image

# Simplified version of the test script
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks

    for i, data in enumerate(dataset):
        #if i >= 10:  # only process one image
        #    break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image path

        for label, image_tensor in visuals.items():
            image_filename = os.path.splitext(os.path.basename(img_path[0]))[0] + '_' + label + '.png'
            save_path = os.path.join(opt.results_dir, image_filename)
            save_image(image_tensor, save_path)  # save the result image
            print(save_path)
