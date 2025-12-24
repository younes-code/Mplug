# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import re
# from nltk.corpus import stopwords

# def read_captions(file_path):
#     captions_dict = {}
#     valid_names = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Normal_Videos_", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
#     pattern = r'^(' + '|'.join(valid_names) + r')([^/]*)'

#     with open(file_path, 'r') as file:
#         for line in file:
#             match = re.match(pattern, line)
#             if match:
#                 category_name = match.group(1)
#             video_name, caption = line.split(':', 1)  # Split at the first colon
#             video_name = video_name.strip()
#             caption = caption.strip()

#             # Removing stop words and converting to lowercase
#             stop_words = set(stopwords.words('english'))
#             cleaned_caption = ' '.join([word.lower().strip() for word in caption.split() if word.lower().strip() not in stop_words])

#             captions_dict[video_name] = [cleaned_caption]

#     return captions_dict

# def get_target_word_embeddings(target_word_lists, model):
#     """
#     Obtain embeddings for target words using SentenceTransformer model.

#     Args:
#         target_word_lists (list): A list of strings, each containing a target word and related words.
#         model (SentenceTransformer): The SentenceTransformer model.

#     Returns:
#         list: A list of tuples containing target words and their embeddings.
#     """
#     target_word_embeddings = []
#     for words in target_word_lists:
#         word_list = words.split(',')
#         target_word = word_list[0].strip()  # Ensure the target word is stripped of any leading/trailing spaces
#         sentence = ' '.join(word_list)
#         # Encode the sentence
#         sentence_embedding = model.encode([sentence])[0]
#         target_word_embeddings.append((target_word, sentence_embedding))
        
#     return target_word_embeddings


# def calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings):
#     """
#     Calculates similarity between pre-computed embeddings and target word sentences.

#     Args:
#         embeddings_dict (dict): A dictionary mapping video names to pre-computed embeddings.
#         target_word_embeddings (list): A list of tuples containing target words and their embeddings.

#     Returns:
#         dict: A dictionary mapping video names to a list of similarities.
#     """
#     similarities = {}
#     for video_name, embeddings in embeddings_dict.items():
#         for caption_embeddings in embeddings:
#             if len(caption_embeddings.shape) == 1:
#                 caption_embeddings = np.expand_dims(caption_embeddings, axis=0)

#             similarities_list = []
#             for target_word, target_word_embedding in target_word_embeddings:
#                 target_word_embedding = np.atleast_2d(target_word_embedding)
#                 similarity = cosine_similarity(caption_embeddings, target_word_embedding)
#                 similarities_list.append((target_word, similarity[0][0]))

#             sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)
#             if video_name not in similarities:
#                 similarities[video_name] = []

#             similarities[video_name].append({
#                 'similarities': sorted_similarities
#             })

#     return similarities

# def main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file):
#     """
#     Main function for calculating caption similarities using pre-computed embeddings.

#     Args:
#         file_path (str): The path to the file containing captions.
#         embeddings_file (str): The path to the file containing pre-computed embeddings.
#         target_word_lists (list): A list of strings, each containing a target word and related words.
#         output_file (str): The path to the output file for saving results.

#     Returns:
#         None
#     """
#     # Load embeddings from the file
#     embeddings_dict = np.load(embeddings_file)
#     model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#     # Get target word embeddings
#     target_word_embeddings = get_target_word_embeddings(target_word_lists, model)
#     similarities = calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings)

#     with open(output_file, 'w') as out_file:
#         for video_name, similarity_list in similarities.items():
#             result_str = f"{video_name}: {sorted(similarity_list[0]['similarities'], key=lambda x: x[1], reverse=True)}"
#             out_file.write(result_str + '\n')


# if __name__ == '__main__':
#     file_path = "concatenated_captions_distilbert.npz.txt"
#     embeddings_file = "concatenated_captions_distilbert.npz"
#     output_file = "concatenated_captions_distilbert_similarities.txt"
#     target_word_lists = ["Abuse,Abuse,maltreatment, mistreatment, cruelty, harm, exploitation, violence, harassment, injury, oppression, torment", 
#                          "Arrest,detain, apprehend, capture, take into custody, seize,law enforcement, police, handcuffs, booking, incarceration, detention", 
#                          "Arson,fire-raising, incendiarism, pyromania, torching,fire, blaze, ignition, burning, conflagration, fire-setting", 
#                          "Assault,attack, physical attack, aggression,violence, strike", 
#                          "Burglary,break-in, robbery, housebreaking, theft,intrusion, trespassing, stealing, larceny, heist, pilfering", 
#                          "Explosion,blast, detonation, eruption, blowup,bomb, burst, combustion, detonate, rupture, outburst", 
#                          "Fighting,brawling, combat, conflict, struggle, tussle,altercation, skirmish, clash, scuffle, fray, melee",
#                          "Accident, accident ", 
#                          "Normal Videos,routine, ordinary, standard, typical, regular,everyday, usual, common, mundane, conventional, normalcy", 
#                          "Robbery,theft, heist, larceny, hold-up, mugging,burglary, stealing, pilfering, banditry, thievery, looting", 
#                          "Shooting,gunfire, gunfight, gunplay, firing, discharge,firearm, bullet, shootout, sniper, marksman, ballistics", 
#                          "Shoplifting,theft, pilfering, larceny, stealing,shop theft, retail theft, snatching, shop burglarizing, petty theft", 
#                          "Stealing,theft, larceny, robbery, pilfering, thievery,embezzlement, burglary, shoplifting, swiping, looting, filching", 
#                          "Vandalism,Vandalism,destruction, defacement, sabotage,graffiti, wrecking, ruin"]
#     main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file)


# # import torch
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity
# # from sentence_transformers import SentenceTransformer
# # from tqdm import tqdm
# # import re
# # from nltk.corpus import stopwords
# # import time

# # def read_captions(file_path):
# #     captions_dict = {}
# #     valid_names = ["test_category"]
# #     pattern = r'^(' + '|'.join(valid_names) + r')([^/]*)'
# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             match = re.match(pattern, line)
# #             if match:
# #                 category_name = match.group(1)
# #             video_name, caption = line.split(':', 1)
# #             video_name = video_name.strip()
# #             caption = caption.strip()
# #             stop_words = set(stopwords.words('english'))
# #             cleaned_caption = ' '.join([word.lower().strip() for word in caption.split() if word.lower().strip() not in stop_words])
# #             captions_dict[video_name] = [cleaned_caption]
# #     return captions_dict

# # def get_target_word_embeddings(target_word_lists, model):
# #     target_word_embeddings = []
# #     for words in target_word_lists:
# #         word_list = words.split(',')
# #         target_word = word_list[0].strip()
# #         sentence = ' '.join(word_list)
# #         sentence_embedding = model.encode([sentence])[0]
# #         target_word_embeddings.append((target_word, sentence_embedding))
# #     return target_word_embeddings

# # def calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings):
# #     similarities = {}
# #     for video_name, embeddings in embeddings_dict.items():
# #         for caption_embeddings in embeddings:
# #             if len(caption_embeddings.shape) == 1:
# #                 caption_embeddings = np.expand_dims(caption_embeddings, axis=0)
# #             similarities_list = []
# #             for target_word, target_word_embedding in target_word_embeddings:
# #                 target_word_embedding = np.atleast_2d(target_word_embedding)
# #                 similarity = cosine_similarity(caption_embeddings, target_word_embedding)
# #                 similarities_list.append((target_word, similarity[0][0]))
# #             sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)
# #             if video_name not in similarities:
# #                 similarities[video_name] = []
# #             similarities[video_name].append({
# #                 'similarities': sorted_similarities
# #             })
# #     return similarities

# # def main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file):
# #     embeddings_dict = np.load(embeddings_file)
# #     model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# #     target_word_embeddings = get_target_word_embeddings(target_word_lists, model)
# #     start_time = time.time()
# #     similarities = calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings)
# #     with open(output_file, 'w') as out_file:
# #         for video_name, similarity_list in similarities.items():
# #             result_str = f"{video_name}: {sorted(similarity_list[0]['similarities'], key=lambda x: x[1], reverse=True)}"
# #             out_file.write(result_str + '\n')
# #     similarity_time = time.time() - start_time
# #     print(f"Similarity calculation time for {len(embeddings_dict)} embeddings: {similarity_time:.2f} seconds")

# # if __name__ == '__main__':
# #     target_word_lists = [
# #         "Abuse,Abuse,maltreatment,mistreatment,cruelty,harm,exploitation,violence,harassment,injury,oppression,torment",
# #         "Arrest,detain,apprehend,capture,take into custody,seize,law enforcement,police,handcuffs,booking,incarceration,detention",
# #         "Arson,fire-raising,incendiarism,pyromania,torching,fire,blaze,ignition,burning,conflagration,fire-setting",
# #         "Assault,attack,physical attack,aggression,violence,strike",
# #         "Burglary,break-in,robbery,housebreaking,theft,intrusion,trespassing,stealing,larceny,heist,pilfering",
# #         "Explosion,blast,detonation,eruption,blowup,bomb,burst,combustion,detonate,rupture,outburst",
# #         "Fighting,brawling,combat,conflict,struggle,tussle,alteraltarion,skirmish,clash,scuffle,fray,melee",
# #         "Accident,accident",
# #         "Normal Videos,routine,ordinary,standard,typical,regular,everyday,usual,common,mundane,conventional,normalcy",
# #         "Robbery,theft,heist,larceny,hold-up,mugging,burglary,stealing,pilfering,banditry,thievery,looting",
# #         "Shooting,gunfire,gunfight,gunplay,firing,discharge,firearm,bullet,shootout,sniper,marksman,ballistics",
# #         "Shoplifting,theft,pilfering,larceny,stealing,shop theft,retail theft,snatching,shop burglarizing,petty theft",
# #         "Stealing,theft,larceny,robbery,pilfering,thievery,embezzlement,burglary,shoplifting,swiping,looting,filching",
# #         "Vandalism,Vandalism,destruction,defacement,sabotage,graffiti,wrecking,ruin"
# #     ]
# #     for file_path, embeddings_file, output_file in [
# #         ('test_video_captions_3s_concatenated.txt', 'embeddings_3s.npz', 'similarities_3s.txt'),

# #     ]:
# #         print(f"\nProcessing {file_path}")
# #         main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file)


# # import torch
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity
# # from sentence_transformers import SentenceTransformer
# # from tqdm import tqdm
# # import re
# # from nltk.corpus import stopwords
# # import time

# # def read_captions(file_path):
# #     """
# #     Read captions from a file and extract class (normal/abnormal) and cleaned caption.
    
# #     Args:
# #         file_path (str): Path to the caption file.
    
# #     Returns:
# #         dict: Dictionary mapping video names to cleaned captions.
# #     """
# #     captions_dict = {}
# #     pattern = r'^(normal|abnormal)_scene_\d+_scenario_\d+'  # Match normal_scene_* or abnormal_scene_*
    
# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             match = re.match(pattern, line)
# #             if not match:
# #                 continue
# #             class_name = match.group(1)  # 'normal' or 'abnormal'
# #             video_name, caption = line.split(':', 1)  # Split at first colon
# #             video_name = video_name.strip()
# #             caption = caption.strip()
            
# #             # Remove stopwords and convert to lowercase
# #             stop_words = set(stopwords.words('english'))
# #             cleaned_caption = ' '.join([word.lower().strip() for word in caption.split() if word.lower().strip() not in stop_words])
            
# #             captions_dict[video_name] = [cleaned_caption]
    
# #     return captions_dict

# # def get_target_word_embeddings(target_word_lists, model):
# #     """
# #     Obtain embeddings for target words using SentenceTransformer model.
    
# #     Args:
# #         target_word_lists (list): List of strings with target word and related words.
# #         model (SentenceTransformer): SentenceTransformer model.
    
# #     Returns:
# #         list: List of tuples containing target words and their embeddings.
# #     """
# #     target_word_embeddings = []
# #     for words in target_word_lists:
# #         word_list = words.split(',')
# #         target_word = word_list[0].strip()
# #         sentence = ' '.join(word_list)
# #         sentence_embedding = model.encode([sentence])[0]
# #         target_word_embeddings.append((target_word, sentence_embedding))
    
# #     return target_word_embeddings

# # def calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings):
# #     """
# #     Calculate similarity between pre-computed embeddings and target word embeddings.
    
# #     Args:
# #         embeddings_dict (dict): Dictionary mapping video names to embeddings.
# #         target_word_embeddings (list): List of tuples with target words and embeddings.
    
# #     Returns:
# #         dict: Dictionary mapping video names to similarity scores.
# #     """
# #     similarities = {}
# #     for video_name, embeddings in embeddings_dict.items():
# #         for caption_embeddings in embeddings:
# #             if len(caption_embeddings.shape) == 1:
# #                 caption_embeddings = np.expand_dims(caption_embeddings, axis=0)
            
# #             similarities_list = []
# #             for target_word, target_word_embedding in target_word_embeddings:
# #                 target_word_embedding = np.atleast_2d(target_word_embedding)
# #                 similarity = cosine_similarity(caption_embeddings, target_word_embedding)
# #                 similarities_list.append((target_word, similarity[0][0]))
            
# #             sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)
# #             if video_name not in similarities:
# #                 similarities[video_name] = []
            
# #             similarities[video_name].append({
# #                 'similarities': sorted_similarities
# #             })
    
# #     return similarities

# # def main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file):
# #     """
# #     Main function to calculate caption similarities using pre-computed embeddings.
    
# #     Args:
# #         file_path (str): Path to caption file.
# #         embeddings_file (str): Path to embeddings file.
# #         target_word_lists (list): List of target words with related terms.
# #         output_file (str): Path to output file.
    
# #     Returns:
# #         None
# #     """
# #     # Start timing
# #     start_time = time.time()
    
# #     # Load embeddings
# #     embeddings_dict = np.load(embeddings_file)
# #     model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
# #     # Get target word embeddings
# #     target_word_embeddings = get_target_word_embeddings(target_word_lists, model)
    
# #     # Calculate similarities
# #     similarities = calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings)
    
# #     # Write results
# #     with open(output_file, 'w') as out_file:
# #         for video_name, similarity_list in similarities.items():
# #             result_str = f"{video_name}: {sorted(similarity_list[0]['similarities'], key=lambda x: x[1], reverse=True)}"
# #             out_file.write(result_str + '\n')
    
# #     # Print timing
# #     similarity_time = time.time() - start_time
# #     print(f"Similarity calculation time for {len(embeddings_dict)} embeddings: {similarity_time:.2f} seconds")

# # if __name__ == '__main__':
# #     file_path = "captions_interval_7s_concatenated.txt"
# #     embeddings_file = "UBnormal_captions_concatenated.npz"
# #     output_file = "UBnormal_captions_concatenated_similarities.txt"
# #     target_word_lists = [
# #         "normal,routine,ordinary,standard,typical,regular,everyday,usual,common,mundane,conventional,normalcy,calm,peaceful",
# #         "abnormal,unusual,irregular,atypical,anomalous,deviant,uncommon,peculiar,strange,bizarre,surreal,odd,exceptional"
# #     ]
# #     main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file)


############################################################################################
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
            video_name = video_name.strip()
            caption = caption.strip()

            # Removing stop words and converting to lowercase
            stop_words = set(stopwords.words('english'))
            cleaned_caption = ' '.join([word.lower().strip() for word in caption.split() if word.lower().strip() not in stop_words])

            captions_dict[video_name] = [cleaned_caption]

    return captions_dict

def get_target_word_embeddings(target_word_lists, model):
    """
    Obtain embeddings for target words using SentenceTransformer model.

    Args:
        target_word_lists (list): A list of strings, each containing a target word and related words.
        model (SentenceTransformer): The SentenceTransformer model.

    Returns:
        list: A list of tuples containing target words and their embeddings.
    """
    target_word_embeddings = []
    for words in target_word_lists:
        word_list = words.split(',')
        target_word = word_list[0].strip()
        sentence = ' '.join(word_list)
        sentence_embedding = model.encode([sentence])[0]
        target_word_embeddings.append((target_word, sentence_embedding))
        
    return target_word_embeddings

def calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings):
    """
    Calculates similarity between pre-computed embeddings and target word sentences using Euclidean distance.

    Args:
        embeddings_dict (dict): A dictionary mapping video names to pre-computed embeddings.
        target_word_embeddings (list): A list of tuples containing target words and their embeddings.

    Returns:
        dict: A dictionary mapping video names to a list of similarities.
    """
    similarities = {}
    for video_name, embeddings in embeddings_dict.items():
        for caption_embeddings in embeddings:
            if len(caption_embeddings.shape) == 1:
                caption_embeddings = np.expand_dims(caption_embeddings, axis=0)

            similarities_list = []
            for target_word, target_word_embedding in target_word_embeddings:
                target_word_embedding = np.atleast_2d(target_word_embedding)
                # Compute Euclidean distance
                distance = np.linalg.norm(caption_embeddings - target_word_embedding, axis=1)
                # Convert distance to similarity (lower distance -> higher similarity)
                similarity = 1 / (1 + distance)
                similarities_list.append((target_word, similarity[0]))

            sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)
            if video_name not in similarities:
                similarities[video_name] = []

            similarities[video_name].append({
                'similarities': sorted_similarities
            })

    return similarities

def main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file):
    """
    Main function for calculating caption similarities using pre-computed embeddings.

    Args:
        file_path (str): The path to the file containing captions.
        embeddings_file (str): The path to the file containing pre-computed embeddings.
        target_word_lists (list): A list of strings, each containing a target word and related words.
        output_file (str): The path to the output file for saving results.

    Returns:
        None
    """
    # Load embeddings from the file
    embeddings_dict = np.load(embeddings_file)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Get target word embeddings
    target_word_embeddings = get_target_word_embeddings(target_word_lists, model)
    similarities = calculate_similarity_from_embeddings(embeddings_dict, target_word_embeddings)

    with open(output_file, 'w') as out_file:
        for video_name, similarity_list in similarities.items():
            result_str = f"{video_name}: {sorted(similarity_list[0]['similarities'], key=lambda x: x[1], reverse=True)}"
            out_file.write(result_str + '\n')

if __name__ == '__main__':
    file_path = "concatenated_captions_distilbert.npz.txt"
    embeddings_file = "ucf_concatenated_captions_embeddings.npz"
    output_file = "ucf_concatenated_captions_Euclidean_similarities.txt"
    target_word_lists = [
        "Abuse,Abuse,maltreatment,mistreatment,cruelty,harm,exploitation,violence,harassment,injury,oppression,torment",
        "Arrest,detain,apprehend,capture,take into custody,seize,law enforcement,police,handcuffs,booking,incarceration,detention",
        "Arson,fire-raising,incendiarism,pyromania,torching,fire,blaze,ignition,burning,conflagration,fire-setting",
        "Assault,attack,physical attack,aggression,violence,strike",
        "Burglary,break-in,robbery,housebreaking,theft,intrusion,trespassing,stealing,larceny,heist,pilfering",
        "Explosion,blast,detonation,eruption,blowup,bomb,burst,combustion,detonate,rupture,outburst",
        "Fighting,brawling,combat,conflict,struggle,tussle,altercation,skirmish,clash,scuffle,fray,melee",
        "Accident,accident",
        "Normal Videos,routine,ordinary,standard,typical,regular,everyday,usual,common,mundane,conventional,normalcy",
        "Robbery,theft,heist,larceny,hold-up,mugging,burglary,stealing,pilfering,banditry,thievery,looting",
        "Shooting,gunfire,gunfight,gunplay,firing,discharge,firearm,bullet,shootout,sniper,marksman,ballistics",
        "Shoplifting,theft,pilfering,larceny,stealing,shop theft,retail theft,snatching,shop burglarizing,petty theft",
        "Stealing,theft,larceny,robbery,pilfering,thievery,embezzlement,burglary,shoplifting,swiping,looting,filching",
        "Vandalism,Vandalism,destruction,defacement,sabotage,graffiti,wrecking,ruin"
    ]
    main_calculate_similarity(file_path, embeddings_file, target_word_lists, output_file)