import os
import numpy as np
from PIL import Image
from options.infer_options import InferOptions
from models import create_model
from data import create_dataset
from util import util  # Assuming util has the necessary function to convert tensor to image
import suntrast
import overlay


def save_image(image_tensor, image_path):
    """Save a single image to the disk."""
    image_numpy = util.tensor2im(image_tensor)  # Convert tensor to numpy array
    image_pil = Image.fromarray(image_numpy)  # Convert numpy array to PIL Image
    h, w, _ = image_numpy.shape

    if opt.aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * opt.aspect_ratio)), Image.BICUBIC)
    if opt.aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / opt.aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)  # Save image

def convert2image(image_tensor):
    image_numpy = util.tensor2im(image_tensor)  # Convert tensor to numpy array
    image_pil = Image.fromarray(image_numpy)  # Convert numpy array to PIL Image
    h, w, _ = image_numpy.shape

    if opt.aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * opt.aspect_ratio)), Image.BICUBIC)
    if opt.aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / opt.aspect_ratio), w), Image.BICUBIC)
    return image_pil

# Simplified version of the test script
if __name__ == '__main__':
    opt = InferOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    if (opt.suntrast):
        #load images from dataroot and transform with suntrast functions then save results in new folder and make that dataroot
        suntrast.process_image_folder(opt.dataroot,opt.suntrast_dir)
    
    opt.dataroot = opt.suntrast_dir
    print('root: '+opt.dataroot)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image path

        # Assuming `visuals` is your dictionary
        real_image_tensor = visuals.get("real")  # Gets the value associated with 'real', if it exists
        fake_image_tensor = visuals.get("fake")  # Gets the value associated with 'fake', if it exists

        overlay_result = overlay.overlay_image(convert2image(real_image_tensor),convert2image(fake_image_tensor))
        #image_filename = os.path.splitext(os.path.basename(img_path[0]))[0] + '_average.png'
        #save_path = os.path.join(opt.results_dir, image_filename)
        #average_result.save(save_path)  # Save image
        #print(save_path)
        #image_filename = os.path.splitext(os.path.basename(img_path[0]))[0] + '_soft_light.png'
        #save_path = os.path.join(opt.results_dir, image_filename)
        #soft_light_result.save(save_path)  # Save image
        #print(save_path)
        image_filename = os.path.splitext(os.path.basename(img_path[0]))[0] + '_overlay.png'
        save_path = os.path.join(opt.results_dir, image_filename)
        overlay_result.save(save_path)  # Save image
        print(save_path)
        
        #for label, image_tensor in visuals.items():
        image_filename = os.path.splitext(os.path.basename(img_path[0]))[0] + '_AI.png'
        save_path = os.path.join(opt.results_dir, image_filename)
        save_image(fake_image_tensor, save_path)  # save the result image
        print(save_path)
