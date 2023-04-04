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

print("start")
start = time.time()

dataset_root = r"Z:\assistant\dataset"
pictures = "input_image_raw_trim"
depth = "input_Image_depth_trial1"
input_csv = "input_CSV_trim"
output_csv = "output_CSV_trial3"

os.makedirs(os.path.join(dataset_root, output_csv), exist_ok=True)
images_files = sorted(glob.glob(os.path.join(dataset_root, pictures, "*.jpg")))
# 나중에 변경필요
images_depth_files = sorted(glob.glob(os.path.join(dataset_root, depth, "*.png")))
input_csv_files = sorted(glob.glob(os.path.join(dataset_root, input_csv, "*.csv")))

# images가 더 많아서 분류작업이 필요함
for each_input_csv in input_csv_files:
    # 1. read image
    image_path_raw = each_input_csv.split('\\')[-1].split('.')[0].split('_')
    each_image_path = os.path.join(dataset_root, pictures, image_path_raw[0] + "_" + image_path_raw[1] + ".jpg")

    img = cv2.imread(each_image_path)
    H, W, _ = img.shape

    # 2. read csv
    df = pd.read_csv(each_input_csv)

    # 3. spatial information, (x,y)
    df["x_centre"] = (df["xmax"] + df["xmin"]) / 2
    df["y_centre"] = (df["ymax"] + df["ymin"]) / 2
    df["image_ratio"] = ((df["xmax"] - df["xmin"]) * (df["ymax"] - df["ymin"])) / (W * H)
    df["x_centre_norm"] = df["x_centre"] / W
    df["y_centre_norm"] = df["y_centre"] / H

    # 4-1. depth 정보의 입력 (z)
    file_name = os.path.basename(each_image_path).split(".")[0]
    depth_file = os.path.join(dataset_root, depth, file_name + ".png")  # should check depth
    if depth_file in images_depth_files:
        for idx, row in df.iterrows():
            depth_image = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)

            x_min = int(row["xmin"])
            y_min = int(row["ymin"])
            x_max = int(row["xmax"])
            y_max = int(row["ymax"])

            # z value를 65536 -> 1.0으로 먼저 normalize한 다음에 이 value에 역수를 취한다.
            # 그러면 상대적인 scale을 가진 depth float value 1 ~ 100의 값들을 얻을 수 있다.
            # 그런데 그냥 depth로 normalize 또 하자
            with np.errstate(divide='ignore', invalid='ignore'):
                depth_image_converted = 1 / (depth_image / 65536) # divide zero error가 나오기는 하는데, 치환할 것이다.
                depth_image_converted[np.isinf(depth_image_converted)] = 1
                depth_image_converted = depth_image_converted / depth_image_converted.max()

            depth_slice = depth_image_converted[y_min:y_max, x_min:x_max]
            df.loc[idx, "avg_depth"] = np.average(np.quantile(depth_slice, [0.45, 0.55]))

    df['key'] = 1
    df['row_num'] = np.arange(df.shape[0])
    df['index'] = df.index

    # compute the Cartesian product of the DataFrame based on the key column
    combinations = pd.merge(df, df, on='key', suffixes=('_sub', '_obj'))

    # remove the key column and duplicate rows
    combinations = combinations[combinations['row_num_sub'] < combinations['row_num_obj']]
    combinations.drop('key', axis=1, inplace=True)
    combinations.drop('row_num_sub', axis=1, inplace=True)
    combinations.drop('row_num_obj', axis=1, inplace=True)

    # 여기서 부터 짜주면 될듯
    # 4. sort

    combinations["cross_size_ratio_1"] = combinations["image_ratio_sub"] / combinations["image_ratio_obj"]
    combinations["cross_size_ratio_2"] = combinations["image_ratio_obj"] / combinations["image_ratio_sub"]

    combinations["cross_size_ratio"] = np.log10(combinations[["cross_size_ratio_1", "cross_size_ratio_2"]].max(axis=1))

    combinations.drop('cross_size_ratio_1', axis=1, inplace=True)
    combinations.drop('cross_size_ratio_2', axis=1, inplace=True)

    combinations['norm_distance_w_size'] = np.sqrt(
        (combinations["x_centre_norm_sub"] - combinations["x_centre_norm_obj"]) ** 2 + (
                combinations["y_centre_norm_sub"] - combinations["y_centre_norm_obj"]) ** 2
        +(combinations["avg_depth_sub"] - combinations["avg_depth_obj"])**2 ) * combinations["cross_size_ratio"]

    #
    combinations['rel'] = " "


    combinations['semantic'] = False

    # 일단은 귀찮으니까 hardcoding

    # combinations = combinations.sort_values(by=["norm_distance_w_size"])
    combinations = combinations[
        ['class_sub', 'class_tmp_sub', 'attribute_refine_sub', 'class_obj', 'class_tmp_obj', 'attribute_refine_obj',
         'rel', 'semantic', 'norm_distance_w_size', 'cross_size_ratio', 'xmin_sub',
         'ymin_sub', 'xmax_sub', 'ymax_sub', 'x_centre_sub', 'y_centre_sub', 'avg_depth_sub',
         'image_ratio_sub', 'x_centre_norm_sub', 'y_centre_norm_sub',
         'xmin_obj', 'ymin_obj', 'xmax_obj', 'ymax_obj', 'x_centre_obj', 'y_centre_obj', 'avg_depth_obj',
         'image_ratio_obj', 'x_centre_norm_obj', 'y_centre_norm_obj', 'index_sub', 'index_obj']]

    # 6. save
    name = each_input_csv.split("\\")[-1].split(".")[0].split("_")[0]
    name_2 = each_input_csv.split("\\")[-1].split(".")[0].split("_")[1]
    combinations.to_csv(os.path.join(dataset_root, output_csv, name + '_' + name_2 + '.csv'), index=False)

print(f'done @ {time.time() - start}')
