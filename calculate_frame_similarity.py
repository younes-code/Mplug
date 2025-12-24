import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
from nltk.corpus import stopwords

def read_captions(file_path):
    captions_dict = {}
    valid_names = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Normal_Videos_", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    pattern = r'^(' + '|'.join(valid_names) + r')([^/]*)'

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                category_name = match.group(1)
            video_name, caption = line.split(':', 1)  # Split at the first colon
            video_name = match.group(1) + match.group(2).split('.')[0]  # Extract the video name with its extension removed
            print("video name",video_name)
            caption = caption.strip()
            print("caption",caption)

            # Removing stop words and converting to lowercase
            stop_words = set(stopwords.words('english'))
            cleaned_caption = ' '.join([word.lower().strip() for word in caption.split() if word.lower().strip() not in stop_words])

            if video_name not in captions_dict:
                captions_dict[video_name] = []
            captions_dict[video_name].append(cleaned_caption)
    return captions_dict

def calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings, target_words):
    similarities = {}
    for video_name, embeddings_list in tqdm(embeddings_dict.items(), desc="Calculating similarities", unit="videos"):
        best_similarity = -1
        best_embedding = None
        best_caption = None
        similarities_ordered = []

        for idx, embedding in enumerate(embeddings_list):
            # Ensure embedding is a 2D array
            if len(embedding.shape) == 1:
                embedding = np.expand_dims(embedding, axis=0)

            # Calculate cosine similarity between the embedding and each target word embedding
            similarities_list = [cosine_similarity(embedding, target_embedding.reshape(1, -1)).item() for target_embedding in target_word_embeddings]

            # Find the highest similarity
            max_similarity = max(similarities_list)

            # Update best similarity, best embedding, and best caption if the current embedding has a higher similarity
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_embedding = embedding
                best_caption = idx

            # Store the similarities ordered to the ground truth
            similarities_ordered.append(sorted(zip(target_words, similarities_list), key=lambda x: x[1], reverse=True))

        similarities[video_name] = {
            'embedding': best_embedding,
            'best_similarity': best_similarity,
            'best_caption': best_caption,
            'similarities_ordered': similarities_ordered
        }

    return similarities


def get_target_word_embeddings(target_words, model):
    target_word_embeddings = []
    for word in target_words:
        # Encode the target word
        word_embedding = model.encode([word])[0]
        target_word_embeddings.append(word_embedding)
    return target_word_embeddings

def main_calculate_similarity(file_path, embeddings_file, target_words, output_file, best_embeddings_file, best_captions_file):
    # Read captions
    captions_dict = read_captions(file_path)

    # Load pre-computed embeddings
    embeddings_dict = np.load(embeddings_file, allow_pickle=True)

    # Initialize SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Get embeddings for target words
    target_word_embeddings = get_target_word_embeddings(target_words, model)

    # Calculate similarities using pre-computed embeddings
    similarities = calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings, target_words)

    print("Keys in captions_dict:", captions_dict.keys())
    print("Keys in embeddings_dict:", embeddings_dict.keys())
    
    # Save similarities to output file
    with open(output_file, 'w') as out_file:
        for video_name, similarity_data in similarities.items():
            # Find the list with the highest similarity
            max_similarity_list = max(similarity_data['similarities_ordered'], key=lambda x: x[0][1])
            # Write video name
            out_file.write(f"{video_name}: {max_similarity_list}\n")

    # Save best embeddings to new .npz file
    best_embeddings = {}
    for video_name, similarity_data in similarities.items():
        if video_name in embeddings_dict:
            best_embeddings[video_name] = similarity_data['embedding']
        else:
            print(f"Warning: Embeddings not found for video {video_name}")
    np.savez(best_embeddings_file, **best_embeddings)

    # Save best captions to file
    with open(best_captions_file, 'w') as captions_file:
        for video_name, similarity_data in similarities.items():
            best_caption_index = similarity_data['best_caption']
            best_caption = captions_dict[video_name][best_caption_index]
            captions_file.write(f"{video_name}: Best Caption = {best_caption}\n")




if __name__ == '__main__':
    file_path = "captions.txt"
    embeddings_file = "three_frames_captions_embeddings.npz"
    output_file = "three_frames_frame_similarities.txt"
    best_embeddings_file = "three_frames_best_embeddings.npz"
    best_captions_file = "three_frames_best_captions.txt"
    target_words = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Normal_Videos_", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
    main_calculate_similarity(file_path, embeddings_file, target_words, output_file, best_embeddings_file, best_captions_file)
