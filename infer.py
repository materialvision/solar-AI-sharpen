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

def ensure_directory_exists(directory):
    """Create the directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Simplified version of the test script
if __name__ == '__main__':
    opt = InferOptions().parse()  # get test options

    # Assuming opt.dataroot is mandatory and provided
    base_dir = opt.dataroot
    print (base_dir)
    print (opt.suntrast_dir)
    if not opt.suntrast_dir:
        opt.suntrast_dir = os.path.join(base_dir, "suntrast")
    ensure_directory_exists(opt.suntrast_dir)  # Check and create suntrast_dir

    if not opt.results_dir:
        opt.results_dir = os.path.join(base_dir, "ai")
    ensure_directory_exists(opt.results_dir)  # Check and create results_dir

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
        model.test()  # run inference the first time
        visuals = model.get_current_visuals()  # get image results after first inference
        img_path = model.get_image_paths()  # get image path

        # Save the result from the first inference
        fake_image_tensor = visuals.get("fake")  # Assuming 'fake' is the result from the first inference
        image_filename = os.path.splitext(os.path.basename(img_path[0]))[0] + '_AI.png'
        save_path = os.path.join(opt.results_dir, image_filename)
        save_image(fake_image_tensor, save_path)  # Save the result image
        print(save_path)

        # Assuming `visuals` is your dictionary
        real_image_tensor = visuals.get("real")  # Gets the value associated with 'real', if it exists

        overlay_result = overlay.overlay_image(convert2image(real_image_tensor),convert2image(fake_image_tensor))
        image_filename = os.path.splitext(os.path.basename(img_path[0]))[0] + '_overlay.png'
        save_path = os.path.join(opt.results_dir, image_filename)
        overlay_result.save(save_path)  # Save image
        print(save_path)

        if (opt.double):
            # Convert the result of the first inference to the expected input format for the second run
            # This might require specific adjustments depending on how 'set_input' expects the data
            new_input_data = {'A': fake_image_tensor, 'A_paths': img_path}  # Adjust this dictionary as per your input format
            model.set_input(new_input_data)  # Set the output of the first inference as input for the second
            model.test()  # run inference again
            visuals_double = model.get_current_visuals()  # get image results after second inference

            # Save the result from the second inference
            fake_image_tensor_double = visuals_double.get("fake")  # Assuming 'fake' is the result from the second inference
            image_filename_double = os.path.splitext(os.path.basename(img_path[0]))[0] + '_AI_double.png'
            save_path_double = os.path.join(opt.results_dir, image_filename_double)
            save_image(fake_image_tensor_double, save_path_double)  # Save the result image with "_double" appended
            print(save_path_double)

            # Assuming `visuals` is your dictionary
            real_image_tensor_double = visuals_double.get("real")  # Gets the value associated with 'real', if it exists

            overlay_result = overlay.overlay_image(convert2image(real_image_tensor_double),convert2image(fake_image_tensor_double))
            image_filename_double = os.path.splitext(os.path.basename(img_path[0]))[0] + '_overlay_double.png'
            save_path_double = os.path.join(opt.results_dir, image_filename_double)
            overlay_result.save(save_path_double)  # Save image
            print(save_path_double)
