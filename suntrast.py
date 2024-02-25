#Thank you Greg Alt for the basic method for this! Check out his SolarFinish script https://github.com/GregAlt/SolarFinish.git

import cv2 as cv
import numpy as np
import scipy as sp
import math
import os

# Turns a centered solar disk image from a disk to a rectangle,
# with rows being angle and columns being distance from center
def polar_warp(img):
    return cv.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), img.shape[0], cv.WARP_FILL_OUTLIERS)


# Turns polar warped image from rectangle back to unwarped solar disk
def polar_unwarp(img, shape):
    # INTER_AREA works best to remove artifacts
    # INTER_CUBIC works well except for a horizontal line artifact at angle = 0
    # INTER_LINEAR, the default has a very noticeable vertical banding artifact across the top, and similarly around the limb
    unwarped = cv.linearPolar(img, (shape[0] / 2, shape[1] / 2), shape[0],
                              cv.WARP_FILL_OUTLIERS | cv.WARP_INVERSE_MAP | cv.INTER_AREA)
    return unwarped


# Calc images with kernel mean and stddev per pixel, with n being fraction of
# circle for kernel around each pixel. Example, 6 means kernels are 60 degree
# arcs. Assumes polar image so that curved kernels are treated as rectangles.
def get_mean_and_std_dev_image(polar_image, n):
    h = polar_image.shape[0]  # image is square, so h=w
    k = (h // (n * 2)) * 2 + 1  # find kernel size from fraction of circle, ensure odd
    left_half = polar_image[:, 0:h // 2]  # left half is circle of radius h//2
    (mean, stddev) = mean_and_std_dev_filter_2d_with_wraparound(left_half, (k, 1))

    # don't use mean filter for corners, just copy that data directly to minimize artifacts
    right_half = polar_image[:, h // 2:]  # right half is corners and beyond
    mean_image = cv.hconcat([mean, right_half])

    # don't use stddev filter for corners, just repeat last column to minimize artifacts
    std_dev_image = np.hstack((stddev, np.tile(stddev[:, [-1]], h - h // 2)))
    return mean_image, std_dev_image


# Pad the image on top and bottom to allow filtering with simulated wraparound
def pad_for_wrap_around(inp, pad):
    return cv.vconcat([inp[inp.shape[0] - pad:, :], inp, inp[:pad, :]])


# Remove padding from top and bottom
def remove_wrap_around_pad(input_padded, pad):
    return input_padded[pad:input_padded.shape[0] - pad, :]


# Low-level func to find mean and std dev images for polar warped image,
# ensuring results are as if kernels wrap around at top and bottom
def mean_and_std_dev_filter_2d_with_wraparound(inp, kernel_size):
    # pad input image with half of kernel to simulate wraparound
    image_pad = pad_for_wrap_around(inp, kernel_size[0] // 2)

    # filter the padded image
    mean_pad = sp.ndimage.uniform_filter(image_pad, kernel_size, mode='reflect')
    mean_of_squared_pad = sp.ndimage.uniform_filter(image_pad * image_pad, kernel_size, mode='reflect')

    # sqrt(mean_of_squared - mean*mean) is mathematically equivalent to std dev:
    #   https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
    std_dev_pad = np.sqrt((mean_of_squared_pad - mean_pad * mean_pad).clip(min=0))

    mean = remove_wrap_around_pad(mean_pad, kernel_size[0] // 2)
    stddev = remove_wrap_around_pad(std_dev_pad, kernel_size[0] // 2)
    return mean, stddev


# CNRGF split into two parts, first part does expensive convolutions
def cnrgf_enhance_part1(img, n, show_intermediate_1, fn):
    # find mean and standard deviation image from polar-warped image, then un-warp
    polar_image = polar_warp(img)
    mean_image, std_devs = get_mean_and_std_dev_image(polar_image, n)
    unwarped_mean = polar_unwarp(mean_image, img.shape)
    unwarped_std_dev = polar_unwarp(std_devs, img.shape)

    if show_intermediate_1 is not None:
        show_intermediate_1(polar_image, mean_image, unwarped_mean, fn)
    return unwarped_mean, unwarped_std_dev


# convert img with cropping/padding to be of size shape, and midpoint at center
def extend_to_match(img, shape, center):
    crop_y_start = img.shape[0] // 2 - center[1]
    crop_y_end = shape[0] - center[1] + img.shape[0] // 2
    crop_x_start = img.shape[1] // 2 - center[0]
    crop_x_end = shape[1] - center[0] + img.shape[1] // 2
    cropped = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    before_y = min(0, img.shape[0] // 2 - center[1])
    after_y = shape[0] - cropped.shape[0] - before_y
    before_x = min(0, img.shape[1] // 2 - center[0])
    after_x = shape[1] - cropped.shape[1] - before_x

    result = np.pad(cropped, ((before_y, after_y), (before_x, after_x)), mode='constant')
    return result


# CNRGF split into two parts, second part is cheaper and has tunable parameters
# using scaleStdDev as a function that has tunable parameters baked into it.
def cnrgf_enhance_part2(img, center, mean_and_stddev, scale_std_dev, show_intermediate_2, fn):
    # adjust range of standard deviation image to get preferred range of contrast enhancement
    unwarped_mean, unwarped_std_dev = mean_and_stddev
    norm_std_dev = scale_std_dev(unwarped_std_dev)

    # subtract mean, divide by standard deviation, and add back mean
    enhance_factor = np.reciprocal(norm_std_dev)

    # img might not be centered and square. So unwarped_mean and enhance_factor need to be
    # first extended to be the same size and solar center as img
    um2 = extend_to_match(unwarped_mean, img.shape, center)
    ef2 = extend_to_match(enhance_factor, img.shape, center)
    diff = img - um2
    enhanced = diff * ef2 + um2

    if show_intermediate_2 is not None:
        show_intermediate_2(diff, norm_std_dev, enhance_factor, fn)
    return enhanced

# expand given a distance
def center_and_expand_to_dist(center, src, max_dist):
    to_left, to_right = (center[0], src.shape[1] - center[0])
    to_top, to_bottom = (center[1], src.shape[0] - center[1])
    new_center = (max_dist, max_dist)
    out_img = np.pad(src, ((max_dist - to_top, max_dist - to_bottom), (max_dist - to_left, max_dist - to_right)),
                     mode='edge')
    return new_center, out_img


# calculate the distance needed for expanding
def center_and_expand_get_dist(center, shape):
    to_left, to_right = (center[0], shape[1] - center[0])
    to_top, to_bottom = (center[1], shape[0] - center[1])
    to_ul = math.sqrt(to_top * to_top + to_left * to_left)
    to_ur = math.sqrt(to_top * to_top + to_right * to_right)
    to_bl = math.sqrt(to_bottom * to_bottom + to_left * to_left)
    to_br = math.sqrt(to_bottom * to_bottom + to_right * to_right)
    return int(max(to_ul, to_ur, to_bl, to_br)) + 1


# Create an expanded image centered on the sun. Ensure that a bounding circle
# centered on the sun and enclosing the original image's four corners is fully
# enclosed in the resulting image. For added pixels, pad by copying the existing
# edge pixels. This means that processing of the polar-warped image has reasonable
# values out to the maximum distance included in the original source image. This,
# in turn, means that circular banding artifacts will occur farther out and can be
# fully cropped out at the end.
def center_and_expand(center, src):
    return center_and_expand_to_dist(center, src, center_and_expand_get_dist(center, src.shape))


# Returns a function that normalizes to within a given range
def get_std_dev_scaler(min_recip, max_recip):
    return lambda sd: cv.normalize(sd, None, 1 / max_recip, 1 / min_recip, cv.NORM_MINMAX)

# Circle finding

# Assumes sun diameter is smaller than image, and sun isn't too small
def is_valid_circle(shape, radius):
    size = min(shape[0], shape[1])
    if 2 * radius > size or 2 * radius < 0.25 * size:
        return False
    return True


# Utility to convert ellipse data to center, radius
def get_circle_data(ellipse):
    if ellipse is None:
        return (0, 0), 0
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    radius = int(0.5 + ellipse[0][2] + (ellipse[0][3] + ellipse[0][4]) * 0.5)
    return center, radius


# Check that ellipse meets valid circle criteria
def is_valid_ellipse(shape, ellipse):
    (center, radius) = get_circle_data(ellipse)
    return is_valid_circle(shape, radius)


# Use Edge Drawing algorithm to find biggest valid circle, assumed to be the solar disk
def find_circle(src):
    # convert to 8bit grayscale for ellipse-detecting
    gray = (src / 256).astype(np.uint8)

    ed_params = cv.ximgproc_EdgeDrawing_Params()
    ed_params.MinPathLength = 300
    ed_params.PFmode = True
    ed_params.MinLineLength = 10
    ed_params.NFAValidation = False

    ed = cv.ximgproc.createEdgeDrawing()
    ed.setParams(ed_params)
    ed.detectEdges(gray)
    ellipses = ed.detectEllipses()

    if ellipses is None:
        return None

    # reject invalid ones *before* finding largest
    ellipses = [e for e in ellipses if is_valid_ellipse(src.shape, e)]
    if len(ellipses) == 0:
        return None

    # find ellipse with the biggest max axis
    return ellipses[np.array([e[0][2] + max(e[0][3], e[0][4]) for e in ellipses]).argmax()]


# Returns True plus center and radius in pixels, if solar disk circle is found
# If it fails at first, maybe that's due to a blurry high-res image, so try on a smaller version
def find_valid_circle(src):
    #(center, radius) = get_circle_data(find_circle(src))
    #print(center)
    #if not is_valid_circle(src.shape, radius):
        # try shrinking image
    thousands = math.ceil(min(src.shape[0], src.shape[1]) / 1000)
    for scale in range(2, thousands + 1):
        smaller = cv.resize(src, (int(src.shape[1] / scale), int(src.shape[0] / scale)))
        (small_center, small_radius) = get_circle_data(find_circle(smaller))
        (center, radius) = ((small_center[0] * scale + scale // 2, small_center[1] * scale + scale // 2),
                            small_radius * scale + scale // 2)
        if is_valid_circle(src.shape, radius):
            break
    return (True, center, radius) if is_valid_circle(src.shape, radius) else (False, None, None)

# Do full CNRGF enhancement from start to finish, displaying intermediate results. Takes a
# float 0-1 image with a centered solar disk.
#
# This uses a process I call Convolutional Normalizing Radial Graded Filter (CNRGF).
# CNRGF was developed largely independently of but was influenced by Druckmullerova's
# FNRGF technique. Instead of using a fourier series to approximate mean and stddev
# around each ring, CNRGF does a simple mean and stddev convolutional filter on a
# polar warped image, and then un-warps those results. This allows for a fairly simple
# and fast python implementation with similar effect of adaptively applying enhancement
# and addressing the radial gradient. CNRGF was developed for processing full-disk
# hydrogen alpha images, while FNRGF was developed for coronal images beyond 1 solar
# radius, but the problems have many similarities, and it should be possible to use the
# algorithms interchangeably for solar images more generally, including full disk
# white light images.
def cnrgf_enhance(src, src_center, n, min_recip, max_recip, min_clip, show_intermediate_1, show_intermediate_2, fn):
    (center, centered) = center_and_expand(src_center, src)
    #test to get and keep src_center from the first image in the series, but maybe more the expanded centered size
    mean_and_stddev = cnrgf_enhance_part1(centered, n, show_intermediate_1, fn)
    enhanced = cnrgf_enhance_part2(src, src_center, mean_and_stddev, get_std_dev_scaler(min_recip, max_recip),
                                   show_intermediate_2, fn)
    clipped = enhanced.clip(min=min_clip)
    normalized = cv.normalize(clipped, None, 0, 1, cv.NORM_MINMAX).clip(min=0).clip(max=1)
    return normalized

def read_image(fn):
    return cv.imread(fn, cv.IMREAD_UNCHANGED | cv.IMREAD_ANYDEPTH)


#
# Pixel format conversions

# Simple linear conversion from RGB to single-channel grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# Just replicate grayscale channel across RGB channels to make a three-channel grayscale image
def gray2rgb(im):
    return cv.merge([im, im, im])


# Colorize 0-1 float image with given RGB gamma values,
# then return as 0-1 float image with B and R swapped
def colorize_float_bgr(result, r, g, b):
    bgr = (np.power(result, b), np.power(result, g), np.power(result, r))
    return cv.merge(bgr)


# Colorize 0-1 float image with given RGB gamma values,
# then return as 0-65535 16 bit image with B and R swapped
def colorize16_bgr(result, r, g, b):
    return float01_to_16bit(colorize_float_bgr(result, r, g, b))


# Colorize 0-1 float image with given RGB gamma values, then return as 0-255 8 bit image in RGB format
def colorize8_rgb(im, r, g, b):
    rgb = (np.power(im, r), np.power(im, g), np.power(im, b))
    return float01_to_8bit(cv.merge(rgb))


# Given arbitrary image file, whether grayscale or RGB, 8 bit or 16 bit, or even 32 bit float
# return as 0-65535 16 bit single-channel grayscale
def force16_gray(im):
    im = rgb2gray(im) if len(im.shape) > 2 else im
    return cv.normalize(im.astype(np.float32), None, 0, 65535, cv.NORM_MINMAX).astype(np.uint16)


# Given a single-channel grayscale 16 bit image, return a three-channel 0-255 8 bit grayscale image
def gray16_to_rgb8(im):
    return gray2rgb((im / 256).astype(np.uint8))


# Convert image from float 0-1 to 16bit uint, works with grayscale or RGB
def float01_to_16bit(im):
    return (im * 65535).astype(np.uint16)


# Convert image from float 0-1 to 8bit uint, works with grayscale or RGB
def float01_to_8bit(im):
    return (im * 255).astype(np.uint8)


# Convert image from 16bit uint to float 0-1, works with grayscale or RGB
def to_float01_from_16bit(im):
    return im.astype(np.float32) / 65535.0


def adjust_gamma(image, gamma=1.0):
    """
    Manually applies gamma correction for 16-bit images.
    """
    if image.dtype == np.uint16:  # Check if the image is 16-bit
        max_val = 65535.0
    else:
        max_val = 255.0  # Default to 8-bit max value if not 16-bit

    invGamma = 1.0 / gamma
    # Normalize the image, apply gamma correction, and scale back
    corrected = ((image / max_val) ** invGamma) * max_val
    # Convert back to original depth
    corrected = np.clip(corrected, 0, max_val).astype(image.dtype)

    return corrected

def is_image_by_extension(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']  # Add more as needed
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def process_image_folder(folder_path,result_path):
    for filename in os.listdir(folder_path):
        print(filename)
        fullpath=os.path.join(folder_path, filename)
        if (is_image_by_extension(fullpath)):  # Output: True or False

            #image_path = '/Users/espensommereide/Dropbox/Projects/SUN/sunplanet_140148_pss.tiff'
            image = force16_gray(read_image(fullpath))
            
            # find the solar disk circle
            (is_valid, src_center, radius) = find_valid_circle(image)
            print(src_center)
            print(radius)
            if not (is_valid):
                # Get dimensions of the image
                height, width = image.shape[:2]

                # Calculate the center
                center_x = width // 2
                center_y = height // 2
                src_center=[center_x,center_y] #set real center
                print('manually set center of image')

            src = to_float01_from_16bit(image)

            enhanced_image = cnrgf_enhance(src, src_center, 6, 3.0, 5.0, 0.016, None, None, "")
            sixteenbit=float01_to_8bit(enhanced_image)
            gamma=adjust_gamma(sixteenbit,1.5)
            # Save the enhanced image
            result_image_path = os.path.join(result_path, filename)
            cv.imwrite(result_image_path, gamma)
            print('processed with sunrast: '+ result_image_path)

if __name__ == "__main__":
    # Example usage
    image_path = '/Users/espensommereide/Dropbox/Projects/SUN/sunplanet_114248_pss.tiff'
    image = force16_gray(read_image(image_path))
    
    # find the solar disk circle
    (is_valid, src_center, radius) = find_valid_circle(image)
    print(src_center)
    print(radius)
    if not (is_valid):
        src_center=[1000,1000]

    src = to_float01_from_16bit(image)

    enhanced_image = cnrgf_enhance(src, src_center, 6, 3.0, 5.0, 0.016, None, None, "")
    sixteenbit=float01_to_8bit(enhanced_image)
    gamma=adjust_gamma(sixteenbit,1.5)
    # Save the enhanced image
    cv.imwrite('/Users/espensommereide/Dropbox/Projects/SUN/sunplanet_114248_pss_cnrgf.png', gamma)

    # Example usage
    image_path = '/Users/espensommereide/Dropbox/Projects/SUN/sunplanet_140148_pss.tiff'
    image = force16_gray(read_image(image_path))
    
    # find the solar disk circle
    (is_valid, src_center, radius) = find_valid_circle(image)
    print(src_center)
    print(radius)
    if not (src_center):
        src_center=[1000,1000]

    src = to_float01_from_16bit(image)

    enhanced_image = cnrgf_enhance(src, src_center, 6, 3.0, 5.0, 0.016, None, None, "")
    sixteenbit=float01_to_8bit(enhanced_image)
    gamma=adjust_gamma(sixteenbit,1.5)
    # Save the enhanced image
    cv.imwrite('/Users/espensommereide/Dropbox/Projects/SUN/sunplanet_140148_pss_cnrgf.png', gamma)


    # Example usage
    image_path = '/Users/espensommereide/Dropbox/Projects/SUN/sunplanet_120333_pss.tiff'
    image = force16_gray(read_image(image_path))
    
    # find the solar disk circle
    (is_valid, src_center, radius) = find_valid_circle(image)
    print(src_center)
    print(radius)
    if not (src_center):
        src_center=[1000,1000]

    src = to_float01_from_16bit(image)

    enhanced_image = cnrgf_enhance(src, src_center, 6, 3.0, 5.0, 0.016, None, None, "")
    sixteenbit=float01_to_8bit(enhanced_image)
    gamma=adjust_gamma(sixteenbit,1.5)
    # Save the enhanced image
    cv.imwrite('/Users/espensommereide/Dropbox/Projects/SUN/sunplanet_120333_pss_cnrgf.png', gamma)
    
