"""
하나의 csv file을 읽어 들인다.
class class_tmp attribute xmin ymin xmax ymax
c1    c_t_1     a_1       1    2    3    4
c2
c3

그리고 그에 corresponding 하는 사진 data를 읽어서 H, W를 구한다.
...

1. 먼저 아래와 같이 mixing을 해준다.
sub	sub_tmp	sub_attri	sxmin	symin	sxmax	symax x_centre_1 y_centre_1 image_ratio_1	obj	obj_tmp	obj_attri	oxmin	oymin	oxmax	oymax x_centre_2 y_centre_2	rel image_ratio_2
c1                                                      c2
c1                                                      c3
c2                                                      c3
c2                                                      c4
c3                                                      c4

2. 이후에, x_centre_1, y_centre_1, image_ratio_1, x_centre_2, y_centre_2, image_ratio_2
다만 각각의 중심부는 normalize 해야 할 것이다.

3. 정렬한다.
어떻게 할지는 아직 정해진 것은 없으나,
d((x_centre_1, y_centre_1), (x_centre_2, y_centre_2))가 작으면 작을 수록 좋다.

4. 최종적으로 필요한 data는 sub sub_tmp sub_attri obj obj_tmp obj_attri rel 이다.

5. depth 정보까지 얻을 수 있게 되었으므로, 각각의 이름에 맞는 사진, depth map을 가지고 와서 45 55 사이의 median depth의 avg를 이용하여
해당 물체의 평균 deoth를 구한다.
즉, x,y,z 정보가 주어지기 떄문에 공간 상에서 두 물체간의 relation에 대해서 파악할 수 있게 된다.

6. classifier를 지속적으로 개선해 나가야 하는데, 먼저 가장 간단하게 parameter 50개 짜리 부터 시작을 한다.
가장 간단한 형태의 classifer는 다음과 같은 형태를 취할 것이다.
act(C_1 * D(o_1, o_2) + C_2 * log10(max(S(o_2)/S(o_1), S(o_1)/S(o_2))) + C_3 * log10(|depth(o_1) - depth(o_2)|) + C_4 * (embed(O_1))*(embed(O_2))/|embed(O_1)|*|embed(O_2)|)

objective = binary cross entropy
최대는 1000개 짜리 parameter를 이용한 model정도가 될것이다. 최대 image가 1000장이고, 가지는 predicate는 10 ~  20개정도 이므로

향후 방향
few-shot learning을 이용해야 한다.

5. 후처리기를 만들어야 한다.
rel이 " "이 아니라면, semantic을 true로 변경하고 필요한 값만을 저장한다.
"""
import pandas as pd
import glob
import os
import cv2
import numpy as np
import time
from tqdm import tqdm

# object annoation 단계에서 이미 corruption이 발생했을 가능성이 많다.
import spellchecker


# 실제 combination을 만들기 전에 spell checker를 통하여 철자 오류를 수정한다.
# 또한 이부분에서 추가 사양으로 필요한 것이 word normalization이 필요하다.
def spell_checker(original, processed, debug_dir, norm_dict):
    start = time.time()
    print(f'spell check for object annotations started')
    word_dict_df = pd.read_csv(norm_dict)

    spell = spellchecker.SpellChecker()
    obj_annot_csv = glob.glob(os.path.join(original, '*.csv'))
    os.makedirs(processed, exist_ok=True)
    word_counter = {}

    # 해당 함수의 초기 사양은 단순히 spelling check만을 하였는데, word normalization도 수행하도록 코드를 수정하였다.
    for each_csv in tqdm(obj_annot_csv):

        df = pd.read_csv(each_csv)
        df['class_refined'] = ""
        # 여기는 row iterations으로 가야함.
        for idx, row in df.iterrows():
            each_class = row['class']
            word_counter[each_class] = word_counter.get(each_class, 0) + 1
            corrected_word = []
            for each_word in each_class.split(' '):
                corrected_word.append(spell.correction(each_word))
            df.at[idx, 'class'] = ' '.join(corrected_word)
            # class normalization
            # 사전을 다시 만들어야 해서 해당 기능은 잠시 중지
            # df.at[idx, 'class_refined'] = word_dict_df.loc[word_dict_df['word'] == ' '.join(corrected_word), 'word_4'].values[0]

        df.drop(columns=['attribute'], inplace=True)
        df.to_csv(os.path.join(processed, os.path.basename(each_csv)), index=False)

    word_statistics_dict = {'class': word_counter.keys(), 'count': word_counter.values()}
    word_statistics = pd.DataFrame(word_statistics_dict)
    # 전체 raw dataset의 단어 분포에 대해서 알 수 있다.
    # 또한 해방 부분을 통해서 오타가 존재하는지의 여부를 빠르게 알 수 있다.
    word_statistics.to_csv(os.path.join(debug_dir, 'word_statistics.csv'), index=False)
    print(f'done @ {time.time() - start}')
    print(f'word statistics saved at {os.path.join(debug_dir, "word_statistics.csv")}')
    return word_statistics


def word_extender(new_raw_list_df, prev_list, output_dir):
    start = time.time()
    print(f'word normalizer generator for object annotations started')

    import inflect
    from nltk.corpus import wordnet
    p = inflect.engine()
    label_to_idx = {"kite": 69, "pant": 87, "bowl": 18, "laptop": 72, "paper": 88, "motorcycle": 80, "railing": 103,
                    "chair": 28, "windshield": 146, "tire": 130, "cup": 34, "bench": 10, "tail": 127, "bike": 11,
                    "board": 13, "orange": 86, "hat": 60, "finger": 46, "plate": 97, "woman": 149, "handle": 59,
                    "branch": 21, "food": 49, "bear": 8, "vase": 140, "vegetable": 141, "giraffe": 52, "desk": 36,
                    "lady": 70, "towel": 132, "glove": 55, "bag": 4, "nose": 84, "rock": 104, "guy": 56, "shoe": 112,
                    "sneaker": 120, "fence": 45, "people": 90, "house": 65, "seat": 108, "hair": 57, "street": 124,
                    "roof": 105, "racket": 102, "logo": 77, "girl": 53, "arm": 3, "flower": 48, "leaf": 73, "clock": 30,
                    "hill": 63, "bird": 12, "umbrella": 139, "leg": 74, "screen": 107, "men": 79, "sink": 116,
                    "trunk": 138, "post": 100, "sidewalk": 114, "box": 19, "boy": 20, "cow": 33, "skateboard": 117,
                    "plane": 95, "stand": 123, "pillow": 93, "ski": 118, "wire": 148, "toilet": 131, "pot": 101,
                    "sign": 115, "number": 85, "pole": 99, "table": 126, "boat": 14, "sheep": 109, "horse": 64,
                    "eye": 43, "sock": 122, "window": 145, "vehicle": 142, "curtain": 35, "kid": 68, "banana": 5,
                    "engine": 42, "head": 61, "door": 38, "bus": 23, "cabinet": 24, "glass": 54, "flag": 47,
                    "train": 135, "child": 29, "ear": 40, "surfboard": 125, "room": 106, "player": 98, "car": 26,
                    "cap": 25, "tree": 136, "bed": 9, "cat": 27, "coat": 31, "skier": 119, "zebra": 150, "fork": 50,
                    "drawer": 39, "airplane": 1, "helmet": 62, "shirt": 111, "paw": 89, "boot": 16, "snow": 121,
                    "lamp": 71, "book": 15, "animal": 2, "elephant": 41, "tile": 129, "tie": 128, "beach": 7,
                    "pizza": 94, "wheel": 144, "plant": 96, "tower": 133, "mountain": 81, "track": 134, "hand": 58,
                    "fruit": 51, "mouth": 82, "letter": 75, "shelf": 110, "wave": 143, "man": 78, "building": 22,
                    "short": 113, "neck": 83, "phone": 92, "light": 76, "counter": 32, "dog": 37, "face": 44,
                    "jacket": 66, "person": 91, "truck": 137, "bottle": 17, "basket": 6, "jean": 67, "wing": 147}

    df_prev = pd.read_csv(prev_list).copy()
    df_new_raw = new_raw_list_df

    # 새로운 row를 추가하는 데, 만일 해당 class가 있다면 pass한다.
    # 하지만 해당 단어가 없다면
    # 1. 먼저 단수형으로 만들고, 유무를 검사한다.
    # 2. 그 이후에는 df에 있는 단어들 중에서 가장 유사한 놈을 추출하고, 그것을 word 4로서 추가 한뒤에, idx도 추가한다.

    for index, row in tqdm(df_new_raw.iterrows(), total=df_new_raw.shape[0]):
        if df_prev['word'].isin([row['class']]).any():
            continue
        else:
            if df_prev['word'].isin([p.singular_noun(row['class'])]).any():
                specific_row = df_prev[df_prev['word'] == p.singular_noun(row['class'])]
                word_norm = specific_row['word_4'].tolist()[0]
                df_row = pd.DataFrame.from_dict([{'word': row['class'], 'word_2': None, 'word_3': None,
                                                  'word_4': word_norm, 'word_4_idx': label_to_idx[word_norm]}])
                df_prev = pd.concat([df_prev, df_row], ignore_index=True)
            else:
                target_word = row['class']
                max_sim = 0.0
                most_similar_word = ''
                most_similar_word_in_category = ''

                synsets1 = wordnet.synsets(target_word)
                for word in df_prev['word'].tolist():
                    synsets2 = wordnet.synsets(word)
                    for synset1 in synsets1:
                        for synset2 in synsets2:
                            similarity = synset1.path_similarity(synset2)
                            if similarity is not None and similarity > max_sim:
                                max_sim = similarity
                                most_similar_word = synset2.name().split('.')[0]
                                if most_similar_word in df_prev['word'].tolist():
                                    most_similar_word_in_category = most_similar_word
                if most_similar_word_in_category == '':
                    word_norm = 'FAIL'
                    word_norm_idx = 1 # todo 실 사용 시에 무조건 수작업으로 변경해주어야 한다.
                    df_row = pd.DataFrame.from_dict([{'word': row['class'], 'word_2': None, 'word_3': 'FAIL',
                                                      'word_4': 'wing', 'word_4_idx': 147}])

                else:
                    # 여기 code가 정규화는 작성되지 않았음
                    specific_row = df_prev[df_prev['word'] == most_similar_word_in_category]
                    word_norm = specific_row['word_4'].tolist()[0]
                    word_norm_idx = label_to_idx[word_norm]
                    df_row = pd.DataFrame.from_dict([{'word': row['class'], 'word_2': None, 'word_3': None, 'word_4': word_norm,'word_4_idx': word_norm_idx}])
                df_prev = pd.concat([df_prev, df_row], ignore_index=True)

    df_prev.to_csv(os.path.join(output_dir, 'word_refine_extend.csv'), index=False)
    return df_prev

def word_normalizer(original, processed, norm_dict):
    start = time.time()
    print(f'word norm for object annotations started')
    word_dict_df = norm_dict

    obj_annot_csv = glob.glob(os.path.join(original, '*.csv'))
    os.makedirs(processed, exist_ok=True)

    # word normalization도 수행하도록 코드를 수정하였다.
    for each_csv in tqdm(obj_annot_csv):
        df = pd.read_csv(each_csv)
        df['class_refined'] = ""
        # 여기는 row iterations으로 가야함.
        for idx, row in df.iterrows():
            each_class = row['class']
            # class normalization
            df.at[idx, 'class_refined'] = word_dict_df.loc[word_dict_df['word'] == each_class, 'word_4'].values[0]
        df.to_csv(os.path.join(processed, os.path.basename(each_csv)), index=False)
    print(f'done @ {time.time() - start}')


def combinator(dataset_root, input_csv, output_csv):
    start = time.time()
    print(f'combinator for object relations started')

    os.makedirs(os.path.join(dataset_root, output_csv), exist_ok=True)
    input_csv_files = sorted(glob.glob(os.path.join(dataset_root, input_csv, "*.csv")))

    # images가 더 많아서 분류작업이 필요함
    for each_input_csv in tqdm(input_csv_files):
        # start_iter = time.time()
        # print("start", each_input_csv)
        # 2. read csv
        df = pd.read_csv(each_input_csv)

        df['key'] = 1
        df['index'] = df.index
        # df.drop('class_confidence', axis=1, inplace=True)

        # 철자 오류로 model에 문제 생기는 것 미연에 방지하기 위하여,
        # spell = spellchecker.SpellChecker()
        # df['class'] = df['class'].apply(lambda x: spell.correction(x))

        # compute the Cartesian product of the DataFrame based on the key column
        combinations = pd.merge(df, df, on='key', suffixes=('_sub', '_obj'))

        # remove the key column and duplicate rows
        combinations = combinations[combinations['index_sub'] != combinations['index_obj']]
        """
        # 전체에 대하여 하기위해서 자기 자신과의 combination만 제거하기
        combinations = combinations[(combinations['xmax_sub'] - combinations['xmin_sub']) * (
                    combinations['ymax_sub'] - combinations['ymin_sub'])
                                    <= (combinations['xmax_obj'] - combinations['xmin_obj']) * (
                                                combinations['ymax_obj'] - combinations['ymin_obj'])]
        """
        combinations.drop('key', axis=1, inplace=True)

        # 여기서 부터 짜주면 될듯
        # 추가 부여 사양

        # combinations['img_name'] = df['img_name'].iloc[0]
        combinations['rel'] = ""
        combinations['rel_score'] = 0.0
        # semantic은 backward compatibilty를 위해서 전부 True로 하여 남겨둠
        combinations['semantic'] = True

        # 일단은 귀찮으니까 hardcoding

        columns = list(combinations.columns)

        combinations.drop('img_name_obj', axis=1, inplace=True)
        combinations.rename(columns={'img_name_sub': 'img_name'}, inplace=True)
        combinations = combinations[[
            'img_name', 'class_sub', 'class_tmp_sub', 'class_obj', 'class_tmp_obj',
            'index_sub', 'index_obj', 'rel', 'rel_score', 'semantic',
            'xmin_sub', 'ymin_sub', 'xmax_sub', 'ymax_sub',
            'xmin_obj', 'ymin_obj', 'xmax_obj', 'ymax_obj',
            'class_refined_sub', 'class_refined_obj',
            'attribute_refine_sub', 'attribute_refine_obj',
            'class_refined_tmp_sub', 'class_refined_tmp_obj'
        ]]

        # 6. save
        name = each_input_csv.split("\\")[-1].split(".")[0]
        combinations.to_csv(os.path.join(dataset_root, output_csv, name + '.csv'), index=False)
        # print(f'done @ {time.time() - start_iter}')
    print(f'done @ {time.time() - start}')


if __name__ == "__main__":
    print("start")
    start = time.time()

    # todo 하단의 3개 작성하기
    # image와 사진에 대하여 전체 root dir
    dataset_root = r"E:\23.04.04\Input"
    # raw csv dir
    input_csv = "CSV"
    # final output csv dir
    output_csv = "CSV_test"
    # 단어 정규화를 위한 사전 파일 위치
    # norm_dict = r"Z:\iet_sgg_trial1\PENET\sgg_service\word_refined_final_done.csv"

    # spell check 및 word statistics를 작성하기 위한 code
    # 처음에 spell checker를 통하여 철자 오류를 수정한다.
    # 그후 해당 dataframe에서 바로 단어 정규화를 실시해서 붙여준다.
    # pre_output_csv = "spell_check_obj_csv"
    # pre2_output_csv = "normed_obj_csv"

    # word_statistics_df = spell_checker(os.path.join(dataset_root, input_csv), os.path.join(dataset_root, pre_output_csv), dataset_root, norm_dict)
    # word_statistics_df = pd.read_csv(os.path.join(dataset_root, "word_statistics.csv"))
    # norm_dict_extend = pd.read_csv(os.path.join(dataset_root, "word_refine_extend.csv"))
    # norm_dict_extend = word_extender(word_statistics_df, norm_dict, dataset_root)
    # word_normalizer(os.path.join(dataset_root, pre_output_csv), os.path.join(dataset_root, pre2_output_csv),
    #                 norm_dict_extend)
    combinator(dataset_root, input_csv, output_csv)

    print(f'done @ {time.time() - start}')
