from PIL import Image, ImageChops, ImageEnhance
import os

def blend_average(image1, image2):
    return ImageChops.blend(image1, image2, alpha=0.5)

def blend_soft_light(image1, image2):
    # Increased contrast for soft light effect
    contrast_enhancer = ImageEnhance.Contrast(image2)
    enhanced_image = contrast_enhancer.enhance(1.7)  # Increase the contrast; adjust this value as needed
    return Image.blend(image1, enhanced_image, alpha=0.5)

def blend_overlay(image1, image2):
    # Overlay blend mode approximation
    return ImageChops.overlay(image1, image2)

def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith("_real.png"):
            base = filename[:-9]  # Remove _real.png
            real_image_path = os.path.join(folder_path, filename)
            fake_image_path = os.path.join(folder_path, f"{base}_fake.png")
            print(fake_image_path)
            if os.path.exists(fake_image_path):
                real_image = Image.open(real_image_path).convert("RGBA")
                fake_image = Image.open(fake_image_path).convert("RGBA")

                # Apply blend modes
                average_result = blend_average(real_image, fake_image)
                soft_light_result = blend_soft_light(real_image, fake_image)
                overlay_result = blend_overlay(real_image, fake_image)

                # Save the results
                average_result.save(os.path.join(folder_path, f"{base}_average.png"))
                soft_light_result.save(os.path.join(folder_path, f"{base}_soft_light.png"))
                overlay_result.save(os.path.join(folder_path, f"{base}_overlay.png"))

def overlay_image(real_image,fake_image):
            #fake_image = Image.open(fake_image_path).convert("RGBA")

            # Apply blend modes
            #average_result = blend_average(real_image, fake_image)
            #soft_light_result = blend_soft_light(real_image, fake_image)
            overlay_result = blend_overlay(real_image, fake_image)

            # Return the results
            return overlay_result
            #return average_result,soft_light_result,overlay_result

if __name__ == "__main__":
    process_images("/Users/espensommereide/Developer/solar-AI-sharpen/results/solarblur2alpha512bw/test_latest/images")