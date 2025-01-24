# import torch
# from PIL import Image
# from transformers import AutoTokenizer, TextStreamer
# from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from conversation import conv_templates
# from model.builder import load_pretrained_model
# from mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# import os
# import re

# def extract_frame_number(filename):
#     """
#     Extracts the numerical part from a filename.

#     Args:
#         filename (str): The input filename.

#     Returns:
#         int: Extracted numerical part or float('inf') if not found.
#     """
#     # Extract numerical part from the filename
#     match = re.search(r'\d+', filename)
#     return int(match.group()) if match else float('inf')


# def generate_caption(model, tokenizer, image_processor, img_path):
#     """
#     Generates a caption for the given image using the provided model and tokenizer.

#     Args:
#         model: The MPLUG model for caption generation.
#         tokenizer: The tokenizer corresponding to the MPLUG model.
#         image_processor: The image processor used for image preprocessing.
#         img_path (str): The path to the input image.

#     Returns:
#         str: The generated caption in the format "directory/image_name ## generated_text".
#     """
#     # Extract the image directory, name, and timestamp
#     img_dir = os.path.basename(os.path.dirname(img_path))
#     img_name = os.path.basename(img_path)
#     timestamp = img_name.split()[1]  # Assuming the timestamp is separated by space in the filename

#     # Load image
#     image = Image.open(img_path).convert('RGB')
#     max_edge = max(image.size)
#     image = image.resize((max_edge, max_edge))
    
#     # Preprocess image
#     image_tensor = process_images([image], image_processor)
#     image_tensor = image_tensor.to(model.device, dtype=torch.float16)

#     # Create conversation context
#     conv = conv_templates["mplug_owl2"].copy()
#     inp = DEFAULT_IMAGE_TOKEN
#     conv.append_message(conv.roles[0], inp)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()
#     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
#     # Define stopping criteria
#     stop_str = conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     # Generate caption
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             do_sample=True,
#             temperature=0.7,
#             max_new_tokens=512,
#             streamer=streamer,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria]
#         )

#     generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    
#     # Combine directory, image name, timestamp, and generated text
#     caption = f"{img_name} ## {generated_text}\n"
#     return caption


# # def main(ucf_path, save_path, skip_frames=90):
# #     """
# #     Main function to process frames, generate captions, and save them to a text file.

# #     Args:
# #         ucf_path (str): The path to the UCF dataset directory.
# #         save_path (str): The path to the output captions text file.
# #         skip_frames (int): Number of frames to skip between generating captions (90 by default).

# #     Returns:
# #         None
# #     """
# #     # Load the MPLUG model and tokenizer
# #     model_path = 'MAGAer13/mplug-owl2-llama2-7b'
# #     model_name = get_model_name_from_path(model_path)
# #     tokenizer, model, image_processor, _ = load_pretrained_model(
# #         model_path, None, model_name, load_4bit=False, device="cuda", offload_folder="offload"
# #     )
# #     # Loop through all subdirectories in the UCF directory
# #     for root, dirs, files in os.walk(ucf_path):
# #         # Sort the files based on extracted numerical parts
# #         files.sort(key=extract_frame_number)

# #         frame_count = 0
# #         generate_caption_flag = True

# #         for file in files:
# #             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
# #                 img_path = os.path.join(root, file)

# #                 if generate_caption_flag:
# #                     # Generate caption for the current image
# #                     caption = generate_caption(model, tokenizer, image_processor, img_path)

# #                     # Save the caption to a text file
# #                     with open(save_path, 'a') as file:
# #                         file.write(caption)

# #                     generate_caption_flag = False

# #                 # Increment the frame counter
# #                 frame_count += 1

# #                 # If the specified number of frames have been processed, reset the counter
# #                 if frame_count >= skip_frames:
# #                     frame_count = 0
# #                     generate_caption_flag = True

# # if __name__ == "__main__":
# #     ucf_directory = '../Datasets/XD-Violance/XD_Violance_frames/'
# #     captions_save_path = 'XD_Violance_captions.txt'

#     main(ucf_directory, captions_save_path)
# def main(image_dir, save_path):
#     """
#     Main function to generate captions for images in a local directory.

#     Args:
#         image_dir (str): The path to the local image directory.
#         save_path (str): The path to the output captions text file.

#     Returns:
#         None
#     """
#     # Load the MPLUG model and tokenizer
#     model_path = 'MAGAer13/mplug-owl2-llama2-7b'
#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, _ = load_pretrained_model(
#         model_path, None, model_name, load_4bit=False, device="cuda", offload_folder="offload"
#     )

#     # Loop through all image files in the directory
#     for file in os.listdir(image_dir):
#         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             img_path = os.path.join(image_dir, file)

#             # Generate caption for the current image
#             caption = generate_caption(model, tokenizer, image_processor, img_path)

#             # Save the caption to a text file
#             with open(save_path, 'a') as f:
#                 f.write(f"{file} ## {caption}\n")

# if __name__ == "__main__":
#     image_directory = '1.png'
#     captions_save_path = 'single_captions.txt'

#     main(image_directory, captions_save_path)

import os
import torch
from PIL import Image
from transformers import AutoTokenizer, TextStreamer
from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from conversation import conv_templates
from model.builder import load_pretrained_model
from mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import re
import gc

def extract_frame_number(filename):
    """
    Extracts the numerical part from a filename.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def resize_image(image, max_size=1024):
    """
    Resize image to ensure it fits into GPU memory more easily.
    """
    max_edge = max(image.size)
    if max_edge > max_size:
        ratio = max_size / float(max_edge)
        new_size = tuple([int(dim * ratio) for dim in image.size])
        image = image.resize(new_size)
    return image

def generate_caption(model, tokenizer, image_processor, img_path):
    """
    Generates a caption for the given image using the provided model and tokenizer.
    """
    img_dir = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path)

    # Open and resize image to optimize memory usage
    image = Image.open(img_path).convert('RGB')
    image = resize_image(image)

    # Preprocess image and move to GPU
    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # Prepare prompt and input tokens
    conv = conv_templates["mplug_owl2"].copy()
    inp = DEFAULT_IMAGE_TOKEN
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.no_grad():
        # Generate caption
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    caption = f"{img_dir}/{img_name} ## {generated_text}\n"

    # Cleanup
    del image_tensor, input_ids, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    return caption


def get_last_processed_image(save_path):
    """
    Read the last processed caption from the output file.
    """
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                last_image_name = last_line.split(" ##")[0]
                return last_image_name
    return None


def main(ucf_path, save_path, skip_frames=90):
    """
    Main function to process frames, generate captions, and save them to a text file.
    """
    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, load_4bit=False, device="cuda", offload_folder="offload"
    )

    # Get the last processed image to continue from
    last_processed_image = get_last_processed_image(save_path)
    print(f"Last processed image: {last_processed_image}")  # Debug print

    start_processing = False

    for root, dirs, files in os.walk(ucf_path):
        files.sort(key=extract_frame_number)

        frame_count = 0
        generate_caption_flag = True

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(root, file)
                print(f"Current image path: {img_path}")  # Debug print

                # Skip images before the last processed one
                if last_processed_image and not start_processing:
                    if img_path == last_processed_image:
                        start_processing = True
                        print(f"Found last processed image. Resuming processing from: {img_path}")  # Debug print
                    else:
                        print(f"Skipping image: {img_path}")  # Debug print
                        continue  # Skip image

                if start_processing:
                    try:
                        print(f"Processing image: {img_path}")  # Debug print
                        caption = generate_caption(model, tokenizer, image_processor, img_path)

                        # Save the caption to a text file
                        with open(save_path, 'a') as f:
                            f.write(caption)
                        print(f"Caption saved: {caption.strip()}")  # Debug print

                        generate_caption_flag = False

                        # Clear memory
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
                        continue

                frame_count += 1

                if frame_count >= skip_frames:
                    frame_count = 0
                    generate_caption_flag = True

    # Final cleanup after all images are processed
    torch.cuda.empty_cache()
    gc.collect()
    print("Processing complete.")  # Debug print


if __name__ == "__main__":
    ucf_directory = '../Datasets/XD-Violance/XD_Violance_frames/'
    captions_save_path = 'XD_Violance_captions.txt'
    main(ucf_directory, captions_save_path)
