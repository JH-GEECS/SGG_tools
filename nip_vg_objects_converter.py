import glob
import os
import pandas as pd
import time
import spellchecker
import numpy as np

def glove_example_code():
    import gensim.downloader

    print(list(gensim.downloader.info()['models'].keys()))
    glove_vectors = gensim.downloader.load('glove-twitter-25')
    test = 1
    glove_vectors.most_similar('twitter')
    glove_vectors.distance('sky', 'trucks')

def nips_dataset_collector(root_dir):
    word_counts = {}
    csv_files = glob.glob(os.path.join(root_dir, '*.csv'))

    for each_csv_file in csv_files:
        df = pd.read_csv(each_csv_file)
        word_list = df['class'].tolist()
        for word in word_list:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    return word_counts

def word_statistics_writer(word_list: dict, path_for_file):
    output_df = pd.DataFrame(word_list.items(), columns=['word', 'count'])
    output_df['index'] = output_df.index
    output_df = output_df[['index', 'word', 'count']]
    output_df.to_csv(os.path.join(path_for_file, 'word_statistics.csv'), index=False)

# label에 spelling error가 있어서 이를 교정하기 위함임
def word_refiner(statistics_path, path_to_csv_file):
    df = pd.read_csv(path_to_csv_file)
    spell = spellchecker.SpellChecker()
    #df['word'] = df['word'].apply(lambda x: spell.correction(x))

    for index, row in df.iterrows():
        if len(row['word'].split(' ')) == 1:
            df.at[index, 'word'] = spell.correction(row['word'])

    df = df.drop(columns=['index'])
    df = df.groupby('word').sum()
    df['word'] = df.index
    df = df.reset_index(drop=True)
    df['index'] = df.index
    output_df = df[['index', 'word', 'count']]
    output_df.to_csv(os.path.join(statistics_path, 'word_statistics_refined.csv'), index=False)

# 이 부분에서는 visual genome에서 쓰인 objects 들을 glove vector로 변환하여,
# nips에 쓰인 label과 비교하여, 가장 유사한 단어를 찾아내는 작업을 함


def nip2vg_converter(statistics_path, refined_csv_path):
    from nltk.corpus import wordnet
    # import nltk
    # nltk.download('wordnet')

    df = pd.read_csv(refined_csv_path)

    label_to_idx = {"kite": 69, "pant": 87, "bowl": 18, "laptop": 72, "paper": 88, "motorcycle": 80, "railing": 103, "chair": 28, "windshield": 146, "tire": 130, "cup": 34, "bench": 10, "tail": 127, "bike": 11, "board": 13, "orange": 86, "hat": 60, "finger": 46, "plate": 97, "woman": 149, "handle": 59, "branch": 21, "food": 49, "bear": 8, "vase": 140, "vegetable": 141, "giraffe": 52, "desk": 36, "lady": 70, "towel": 132, "glove": 55, "bag": 4, "nose": 84, "rock": 104, "guy": 56, "shoe": 112, "sneaker": 120, "fence": 45, "people": 90, "house": 65, "seat": 108, "hair": 57, "street": 124, "roof": 105, "racket": 102, "logo": 77, "girl": 53, "arm": 3, "flower": 48, "leaf": 73, "clock": 30, "hill": 63, "bird": 12, "umbrella": 139, "leg": 74, "screen": 107, "men": 79, "sink": 116, "trunk": 138, "post": 100, "sidewalk": 114, "box": 19, "boy": 20, "cow": 33, "skateboard": 117, "plane": 95, "stand": 123, "pillow": 93, "ski": 118, "wire": 148, "toilet": 131, "pot": 101, "sign": 115, "number": 85, "pole": 99, "table": 126, "boat": 14, "sheep": 109, "horse": 64, "eye": 43, "sock": 122, "window": 145, "vehicle": 142, "curtain": 35, "kid": 68, "banana": 5, "engine": 42, "head": 61, "door": 38, "bus": 23, "cabinet": 24, "glass": 54, "flag": 47, "train": 135, "child": 29, "ear": 40, "surfboard": 125, "room": 106, "player": 98, "car": 26, "cap": 25, "tree": 136, "bed": 9, "cat": 27, "coat": 31, "skier": 119, "zebra": 150, "fork": 50, "drawer": 39, "airplane": 1, "helmet": 62, "shirt": 111, "paw": 89, "boot": 16, "snow": 121, "lamp": 71, "book": 15, "animal": 2, "elephant": 41, "tile": 129, "tie": 128, "beach": 7, "pizza": 94, "wheel": 144, "plant": 96, "tower": 133, "mountain": 81, "track": 134, "hand": 58, "fruit": 51, "mouth": 82, "letter": 75, "shelf": 110, "wave": 143, "man": 78, "building": 22, "short": 113, "neck": 83, "phone": 92, "light": 76, "counter": 32, "dog": 37, "face": 44, "jacket": 66, "person": 91, "truck": 137, "bottle": 17, "basket": 6, "jean": 67, "wing": 147}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    df['word_3'] = ''
    df['word_3_idx'] = 0
    for index, row in df.iterrows():
        start_time = time.time()
        print('word ', index, ' start')
        if row['word_2'] in label_to_idx.keys():
            df.at[index, 'word_3'] = row['word_2']
            df.at[index, 'word_3_idx'] = label_to_idx[row['word_2']]
            print('word ', index, ' done @ ',time.time() - start_time)
        else:
            target_word = row['word_2']
            max_sim = 0.0
            most_similar_word = ''
            most_similar_word_in_category = ''

            synsets1 = wordnet.synsets(target_word)
            for word in label_to_idx.keys():
                synsets2 = wordnet.synsets(word)

                for synset1 in synsets1:
                    for synset2 in synsets2:
                        similarity = synset1.path_similarity(synset2)
                        if similarity is not None and similarity > max_sim:
                            max_sim = similarity
                            most_similar_word = synset2.name().split('.')[0]
                            if most_similar_word in label_to_idx.keys():
                                most_similar_word_in_category = most_similar_word
            if most_similar_word_in_category == '':
                df.at[index, 'word_3'] = 0
                df.at[index, 'word_3_idx'] = 0
                print('Fail!!! word ', index, ' done @ ', time.time() - start_time)
            else:
                df.at[index, 'word_3'] = most_similar_word_in_category
                df.at[index, 'word_3_idx'] = label_to_idx[most_similar_word_in_category]
                print('word ', index, ' done @ ', time.time() - start_time)
    df.to_csv(os.path.join(statistics_path, 'word_statistics_refined5.csv'), index=False)

# handcraft로 가장 semantically similar한 word table을 만들어 주는 code
def nip2vg_converted_refiner(statistics_path, refined_csv_path, label_to_fix_path):
    import gensim.downloader
    glove_vectors = gensim.downloader.load('glove-twitter-25')

    df = pd.read_csv(refined_csv_path)
    vg_to_idx = {"kite": 69, "pant": 87, "bowl": 18, "laptop": 72, "paper": 88, "motorcycle": 80, "railing": 103, "chair": 28, "windshield": 146, "tire": 130, "cup": 34, "bench": 10, "tail": 127, "bike": 11, "board": 13, "orange": 86, "hat": 60, "finger": 46, "plate": 97, "woman": 149, "handle": 59, "branch": 21, "food": 49, "bear": 8, "vase": 140, "vegetable": 141, "giraffe": 52, "desk": 36, "lady": 70, "towel": 132, "glove": 55, "bag": 4, "nose": 84, "rock": 104, "guy": 56, "shoe": 112, "sneaker": 120, "fence": 45, "people": 90, "house": 65, "seat": 108, "hair": 57, "street": 124, "roof": 105, "racket": 102, "logo": 77, "girl": 53, "arm": 3, "flower": 48, "leaf": 73, "clock": 30, "hill": 63, "bird": 12, "umbrella": 139, "leg": 74, "screen": 107, "men": 79, "sink": 116, "trunk": 138, "post": 100, "sidewalk": 114, "box": 19, "boy": 20, "cow": 33, "skateboard": 117, "plane": 95, "stand": 123, "pillow": 93, "ski": 118, "wire": 148, "toilet": 131, "pot": 101, "sign": 115, "number": 85, "pole": 99, "table": 126, "boat": 14, "sheep": 109, "horse": 64, "eye": 43, "sock": 122, "window": 145, "vehicle": 142, "curtain": 35, "kid": 68, "banana": 5, "engine": 42, "head": 61, "door": 38, "bus": 23, "cabinet": 24, "glass": 54, "flag": 47, "train": 135, "child": 29, "ear": 40, "surfboard": 125, "room": 106, "player": 98, "car": 26, "cap": 25, "tree": 136, "bed": 9, "cat": 27, "coat": 31, "skier": 119, "zebra": 150, "fork": 50, "drawer": 39, "airplane": 1, "helmet": 62, "shirt": 111, "paw": 89, "boot": 16, "snow": 121, "lamp": 71, "book": 15, "animal": 2, "elephant": 41, "tile": 129, "tie": 128, "beach": 7, "pizza": 94, "wheel": 144, "plant": 96, "tower": 133, "mountain": 81, "track": 134, "hand": 58, "fruit": 51, "mouth": 82, "letter": 75, "shelf": 110, "wave": 143, "man": 78, "building": 22, "short": 113, "neck": 83, "phone": 92, "light": 76, "counter": 32, "dog": 37, "face": 44, "jacket": 66, "person": 91, "truck": 137, "bottle": 17, "basket": 6, "jean": 67, "wing": 147}
    idx_to_vg = {v: k for k, v in vg_to_idx.items()}

    # label에 중복처리가 안되어 있음, 따라서 dataframe을 그대로 이용 해야 한다.
    # 아래의 dict code는 폐기 처리한다.
    """
    label_to_idx = {}
    for idx, word in enumerate(df['word_2']):
        label_to_idx[word] = idx+1
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    """

    similiarity_matrix = np.zeros((len(df['word_2']), len(idx_to_vg)), dtype=np.float32)
    for i in range(len(df['word_2'])):
        for j in idx_to_vg.keys():
            similiarity_matrix[i][j-1] = glove_vectors.similarity(df['word_2'].iloc[i], idx_to_vg[j])

    argsort_similarity = np.argsort(-similiarity_matrix, axis=1)
    arg_top5_similarity = argsort_similarity[:, :10]

    idx_to_label_df = pd.DataFrame(df['word_2'])
    idx_to_label_df[['top1', 'top2', 'top3', 'top4', 'top5', 'top6', 'top7', 'top8', 'top9', 'top10']] = pd.DataFrame(arg_top5_similarity, index=idx_to_label_df.index)
    # todo 와 lambda식 미쳤네
    for i in range(1, 11):
        idx_to_label_df['top{}'.format(i)] = idx_to_label_df['top{}'.format(i)].apply(lambda x: idx_to_vg[x+1])

    idx_to_label_df_req_process = idx_to_label_df[df['req_refine'] == True]
    idx_to_label_df_req_process.to_csv(label_to_fix_path)

def nip_refine_vg_map(orgin, map_file, result):
    df_map = pd.read_csv(map_file)
    df_map = df_map[['word_2', 'word_3']]
    df_map_dict = df_map.set_index('word_2')['word_3'].to_dict()

    df_origin = pd.read_csv(orgin)
    df_origin['word_4'] = df_origin['word_2'].apply(lambda x: df_map_dict.get(x, x))

    new_word = []
    for word in df_origin['word_2']:
        if word not in df_map_dict.keys():
            new_word.append(word)

    df_origin.to_csv(result)
    test = 1

def final_converter(orgin, arrival):
    label_to_idx = {"kite": 69, "pant": 87, "bowl": 18, "laptop": 72, "paper": 88, "motorcycle": 80, "railing": 103, "chair": 28, "windshield": 146, "tire": 130, "cup": 34, "bench": 10, "tail": 127, "bike": 11, "board": 13, "orange": 86, "hat": 60, "finger": 46, "plate": 97, "woman": 149, "handle": 59, "branch": 21, "food": 49, "bear": 8, "vase": 140, "vegetable": 141, "giraffe": 52, "desk": 36, "lady": 70, "towel": 132, "glove": 55, "bag": 4, "nose": 84, "rock": 104, "guy": 56, "shoe": 112, "sneaker": 120, "fence": 45, "people": 90, "house": 65, "seat": 108, "hair": 57, "street": 124, "roof": 105, "racket": 102, "logo": 77, "girl": 53, "arm": 3, "flower": 48, "leaf": 73, "clock": 30, "hill": 63, "bird": 12, "umbrella": 139, "leg": 74, "screen": 107, "men": 79, "sink": 116, "trunk": 138, "post": 100, "sidewalk": 114, "box": 19, "boy": 20, "cow": 33, "skateboard": 117, "plane": 95, "stand": 123, "pillow": 93, "ski": 118, "wire": 148, "toilet": 131, "pot": 101, "sign": 115, "number": 85, "pole": 99, "table": 126, "boat": 14, "sheep": 109, "horse": 64, "eye": 43, "sock": 122, "window": 145, "vehicle": 142, "curtain": 35, "kid": 68, "banana": 5, "engine": 42, "head": 61, "door": 38, "bus": 23, "cabinet": 24, "glass": 54, "flag": 47, "train": 135, "child": 29, "ear": 40, "surfboard": 125, "room": 106, "player": 98, "car": 26, "cap": 25, "tree": 136, "bed": 9, "cat": 27, "coat": 31, "skier": 119, "zebra": 150, "fork": 50, "drawer": 39, "airplane": 1, "helmet": 62, "shirt": 111, "paw": 89, "boot": 16, "snow": 121, "lamp": 71, "book": 15, "animal": 2, "elephant": 41, "tile": 129, "tie": 128, "beach": 7, "pizza": 94, "wheel": 144, "plant": 96, "tower": 133, "mountain": 81, "track": 134, "hand": 58, "fruit": 51, "mouth": 82, "letter": 75, "shelf": 110, "wave": 143, "man": 78, "building": 22, "short": 113, "neck": 83, "phone": 92, "light": 76, "counter": 32, "dog": 37, "face": 44, "jacket": 66, "person": 91, "truck": 137, "bottle": 17, "basket": 6, "jean": 67, "wing": 147}

    df_origin = pd.read_csv(orgin)
    df_origin['word_4_idx'] = df_origin['word_4'].apply(lambda x: label_to_idx[x])

    df_origin.to_csv(arrival)


if __name__ == '__main__':
    start_time = time.time()
    print('start')

    # 처음 모든 단어를 획득하기 위한 code
    """
    big_one_path = r'Z:\assistant\assistant_deploy\obj_det_anot_spc'
    statistics_path = r'Z:\assistant\assistant_deploy\word_conversion'
    word_list = nips_dataset_collector(big_one_path)
    word_statistics_writer(word_list, statistics_path)
    """


    # spelling error가 있는 단어를 교정한 code
    """
    statistics_path = r'Z:\assistant\assistant_deploy\word_conversion'
    statistics_file_path = r'Z:\assistant\assistant_deploy\word_conversion\word_statistics.csv'
    word_refiner(statistics_path, statistics_file_path)
    """
    """
    # scence graph model의 class label을 glove vector로 변환하기 위한 code
    statistics_path = r'Z:\assistant\assistant_deploy\word_conversion'
    once_refined_statistics_path = r'Z:\assistant\assistant_deploy\word_conversion\word_statistics_refined.csv'
    nip2vg_converter(statistics_path, once_refined_statistics_path)
    """
    original_file = r'Z:\assistant\assistant_deploy\word_conversion\word_refined_final.csv'
    map_file = r'C:\model_temp\raw\word_statistics_refined5-map.csv'
    result_file = r'Z:\assistant\assistant_deploy\word_conversion\word_refined_final_done.csv'
    final_converter(original_file, result_file)

    """
    statistics_path = r'C:\model_temp\raw'
    twice_refined_statistics_path = r'C:\model_temp\raw\word_statistics_refined5-pre_hand_process.csv'
    label_to_fix_path = r'C:\model_temp\raw\word_statistics_refined5-pre_hand_process_req_list.csv'
    nip2vg_converted_refiner(statistics_path, twice_refined_statistics_path, label_to_fix_path)
    """

    print(f'done @ {time.time() - start_time}')

