import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import torch.nn.functional as F

def read_captions(file_path):
    """
    Reads captions from a file and processes them.

    Args:
        file_path (str): The path to the file containing captions.

    Returns:
        dict: A dictionary mapping video names to processed captions.
    """
    captions_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        captions = line.split(':')
        video_name = captions[0].strip()
        frames_captions = [caption.strip() for caption in captions[1].split('##')]
        captions_dict[video_name] = frames_captions
    return captions_dict

def generate_embeddings(captions_dict, bert_model, tokenizer, output_file):
    """
    Generates embeddings for captions and saves them into a file.

    Args:
        captions_dict (dict): A dictionary mapping video names to processed captions.
        bert_model (AutoModel): The pre-trained BERT model.
        tokenizer (AutoTokenizer): The BERT tokenizer.
        output_file (str): The path to the output file for saving embeddings.

    Returns:
        None
    """
    embeddings_dict = {}
    for video_name, captions_list in tqdm(captions_dict.items()):
        embeddings_list = []
        for caption in captions_list:
            caption_tokens = tokenizer(caption, return_tensors="pt", truncation=True, padding=True)['input_ids']
            with torch.no_grad():
                caption_outputs = bert_model(input_ids=caption_tokens)
                caption_embeddings = caption_outputs.last_hidden_state[0, :, :]
            embeddings_list.append(caption_embeddings.numpy())
        embeddings_dict[video_name] = np.stack(embeddings_list)
    
    # Save embeddings to file
    np.savez(output_file, **embeddings_dict)
    print(f"Embeddings saved to {output_file}")

def load_pretrained_bert_model():
    """
    Loads a pre-trained BERT model and tokenizer.

    Returns:
        tuple: A tuple containing the loaded BERT model and tokenizer.
    """
    print("Starting the download")

    # Load pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Download done")
    return model, tokenizer

def get_target_word_embeddings(target_words, bert_model, tokenizer):
    """
    Obtain embeddings for target words using BERT model.

    Args:
        target_words (list): A list of target words.
        bert_model (AutoModel): The pre-trained BERT model.
        tokenizer (AutoTokenizer): The BERT tokenizer.

    Returns:
        np.array: An array containing embeddings for target words.
    """
    target_word_embeddings = []
    for word in target_words:
        # Tokenize the target word
        word_tokens = tokenizer(word, return_tensors="pt", truncation=True, padding=True)['input_ids']
        with torch.no_grad():
            # Get embeddings for the target word
            word_outputs = bert_model(input_ids=word_tokens)
            word_embedding = word_outputs.last_hidden_state[0, :, :]
            target_word_embeddings.append(word_embedding.numpy())
    return np.concatenate(target_word_embeddings, axis=0)

def calculate_similarity_from_embeddings(embeddings_file, target_word_embeddings, target_words):
    """
    Calculates similarity between pre-computed embeddings and target words.

    Args:
        embeddings_file (str): The path to the file containing pre-computed embeddings.
        target_word_embeddings (np.array): An array containing embeddings for target words.
        target_words (list): A list of target words for similarity calculation.

    Returns:
        dict: A dictionary mapping video names to a list of similarities.
    """
    similarities = {}
    embeddings_dict = np.load(embeddings_file, allow_pickle=True)
    for video_name, embeddings in embeddings_dict.items():
        for caption_embeddings in embeddings:
            # Ensure caption_embeddings is a 2D array
            caption_embeddings = np.atleast_2d(caption_embeddings)

            similarities_list = []
            for i in range(len(target_words)):
                # Ensure target_word_embeddings[i] is a 2D array
                target_word_embedding_i = np.atleast_2d(target_word_embeddings[i])
                
                # Calculate cosine similarity between the caption and the target word
                similarity = cosine_similarity(caption_embeddings, target_word_embedding_i)
                
                # Append the similarity to the list
                similarities_list.append((target_words[i], similarity[0][0]))

            # Sort the similarities in descending order
            sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)

            if video_name not in similarities:
                similarities[video_name] = []

            similarities[video_name].append({
            'scores': sorted_similarities
            })

    return similarities


def main_calculate_similarity(file_path, embeddings_file, target_words, output_file):
    """
    Main function for calculating caption similarities using pre-computed embeddings.

    Args:
        file_path (str): The path to the file containing captions.
        embeddings_file (str): The path to the file containing pre-computed embeddings.
        target_words (list): A list of target words for similarity calculation.
        output_file (str): The path to the output file for saving results.

    Returns:
        None
    """
    bert_model, tokenizer = load_pretrained_bert_model()
    target_word_embeddings = get_target_word_embeddings(target_words, bert_model, tokenizer)
    similarities = calculate_similarity_from_embeddings(embeddings_file, target_word_embeddings, target_words)

    with open(output_file, 'w') as out_file:
        for video_name, similarity_list in similarities.items():
            result_str = f"{video_name}: {sorted(similarity_list[0]['scores'], key=lambda x: x[1], reverse=True)}"
            print(result_str)
            out_file.write(result_str + '\n')


if __name__ == '__main__':
    file_path = "concatenated_captions.txt"
    embeddings_file = "captions_embeddings.npz"
    output_file = "similarities_testing.txt"
    classes = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal Videos", "Road Accidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    
#     main_generate_embeddings(file_path, embeddings_file)
    main_calculate_similarity(file_path, embeddings_file, classes, output_file)
