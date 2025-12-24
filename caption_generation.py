
# import os
# import torch
# from PIL import Image
# from transformers import AutoTokenizer, TextStreamer
# from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from conversation import conv_templates
# from model.builder import load_pretrained_model
# from mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# import re
# import gc
# import time

# def extract_frame_number(filename):
#     """
#     Extracts the numerical part from a filename.
#     """
#     match = re.search(r'\d+', filename)
#     return int(match.group()) if match else float('inf')

# def resize_image(image, max_size=1024):
#     """
#     Resize image to ensure it fits into GPU memory more easily.
#     """
#     max_edge = max(image.size)
#     if max_edge > max_size:
#         ratio = max_size / float(max_edge)
#         new_size = tuple([int(dim * ratio) for dim in image.size])
#         image = image.resize(new_size)
#     return image

# def generate_caption(model, tokenizer, image_processor, img_path):
#     """
#     Generates a caption for the given image using the provided model and tokenizer.
#     """
#     img_dir = os.path.basename(os.path.dirname(img_path))
#     img_name = os.path.basename(img_path)

#     # Open and resize image to optimize memory usage
#     image = Image.open(img_path).convert('RGB')
#     image = resize_image(image)

#     # Preprocess image and move to GPU
#     image_tensor = process_images([image], image_processor)
#     image_tensor = image_tensor.to(model.device, dtype=torch.float16)

#     # Prepare prompt and input tokens
#     conv = conv_templates["mplug_owl2"].copy()
#     inp = DEFAULT_IMAGE_TOKEN
#     conv.append_message(conv.roles[0], inp)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()
#     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

#     stop_str = conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#     with torch.no_grad():
#         # Generate caption
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             do_sample=True,
#             temperature=0.7,
#             max_new_tokens=512,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria],
#         )

#     generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
#     caption = f"{img_dir}/{img_name} ## {generated_text}\n"

#     # Cleanup
#     del image_tensor, input_ids, output_ids
#     torch.cuda.empty_cache()
#     gc.collect()

#     return caption

# def main(ucf_path, save_path, skip_frames=30, max_frames=float("inf")):
#     """
#     Main function to process a sample of frames, generate captions, and calculate average processing time.

#     Args:
#         ucf_path (str): Path to the directory containing frame images.
#         save_path (str): Path to save the generated captions.
#         skip_frames (int): Number of frames to skip between processed frames (sampling rate).
#         max_frames (int): Maximum number of frames to process for timing estimation.
#     """
#     model_path = 'MAGAer13/mplug-owl2-llama2-7b'
#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, _ = load_pretrained_model(
#         model_path, None, model_name, load_4bit=False, device="cuda", offload_folder="offload"
#     )

#     # Initialize timing and frame counter
#     start_time = time.time()  # Start timing for caption generation
#     processed_frames = 0  # Count frames actually processed

#     for root, dirs, files in os.walk(ucf_path):
#         files.sort(key=extract_frame_number)
#         frame_count = 0

#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                 img_path = os.path.join(root, file)

#                 # Process frame if it’s the selected one based on skip_frames
#                 if frame_count % skip_frames == 0:
#                     try:
#                         print(f"Processing image: {img_path}")
#                         caption = generate_caption(model, tokenizer, image_processor, img_path)

#                         # Save the caption to a text file
#                         with open(save_path, 'a') as f:
#                             f.write(caption)
#                         print(f"Caption saved: {caption.strip()}")

#                         processed_frames += 1  # Increment frame counter

#                         # Clear memory
#                         torch.cuda.empty_cache()
#                         gc.collect()

#                     except Exception as e:
#                         print(f"Error processing image {img_path}: {e}")
#                         continue

#                     # Stop after processing max_frames
#                     if processed_frames >= max_frames:
#                         break

#                 frame_count += 1

#         # Break outer loop if max_frames reached
#         if processed_frames >= max_frames:
#             break

#     # Calculate and print timing results
#     caption_time = time.time() - start_time  # End timing
#     print(f"Caption generation time for {processed_frames} frames: {caption_time:.2f} seconds")
#     if processed_frames > 0:
#         avg_time_per_frame = caption_time / processed_frames
#         print(f"Average time per frame: {avg_time_per_frame:.2f} seconds")
#     else:
#         print("No frames processed.")

#     # Final cleanup
#     torch.cuda.empty_cache()
#     gc.collect()
#     print("Processing complete.")

# if __name__ == "__main__":
#     ucf_directory = 'test_video-frames'
#     for skip_frames, save_path in [
#         (3, 'test_video_captions_3.txt'),
#         (5, 'test_video_captions_5.txt'),
#         (7, 'test_video_captions_7.txt')
#     ]:
#         print(f"\nRunning with skip_frames={skip_frames}")
#         main(ucf_directory, save_path, skip_frames, max_frames=10)

########################################## temporal selection #############################################

import os
import torch
from PIL import Image
from transformers import AutoTokenizer
from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from conversation import conv_templates
from model.builder import load_pretrained_model
from mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import re
import gc
import time

# ---------------------------
# Utility functions
# ---------------------------
def extract_frame_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def resize_image(image, max_size=1024):
    max_edge = max(image.size)
    if max_edge > max_size:
        ratio = max_size / float(max_edge)
        new_size = tuple([int(dim * ratio) for dim in image.size])
        image = image.resize(new_size)
    return image

def generate_caption(model, tokenizer, image_processor, img_path):
    img_dir = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path)

    image = Image.open(img_path).convert('RGB')
    image = resize_image(image)

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

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

    del image_tensor, input_ids, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    return caption

# ---------------------------
# Resume mechanism
# ---------------------------
def load_done_images(save_path):
    done = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                if "##" in line:
                    img = line.split("##")[0].strip()  # ex: RoadAccidents/...jpg
                    done.add(img)
    return done

# ---------------------------
# Main processing
# ---------------------------
def main(ucf_path, save_path, skip_frames=1):
    # Ensure output folder exists
    output_folder = 'temporal_selection'
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, save_path)

    start_time = time.time()
    processed_frames = 0

    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, load_4bit=False, device="cuda", offload_folder="offload"
    )

    done_images = load_done_images(save_path)

    for root, dirs, files in os.walk(ucf_path):
        files.sort(key=extract_frame_number)
        frame_count = 0

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(root, file)
                relative_name = os.path.basename(root) + "/" + file

                # --- RESUME CHECK ---
                if relative_name in done_images:
                    continue

                if frame_count % skip_frames == 0:
                    try:
                        print(f"Processing image: {img_path}")
                        caption = generate_caption(model, tokenizer, image_processor, img_path)
                        with open(save_path, 'a') as f:
                            f.write(caption)
                        processed_frames += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                frame_count += 1

    total_time = time.time() - start_time
    avg_time = total_time / processed_frames if processed_frames else 0
    log_msg = f"Processed {processed_frames} frames with skip_frames={skip_frames}\n" \
              f"Total time: {total_time:.2f}s, Avg per frame: {avg_time:.2f}s\n"

    print(log_msg)
    # Save timing log in temporal_selection folder
    log_file_path = os.path.join(output_folder, "caption_time_log.txt")
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_msg)

    torch.cuda.empty_cache()
    gc.collect()
    print("Processing complete.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    ucf_directory = '../Datasets/active_frames_output'
    save_filename = 'captions_all.txt'
    main(ucf_directory, save_filename, skip_frames=1)
