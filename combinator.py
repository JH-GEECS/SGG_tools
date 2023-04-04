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

# object annoation 단계에서 이미 corruption이 발생했을 가능성이 많다.
import spellchecker

print("start")
start = time.time()

dataset_root = r"Z:\assistant\assistant_deploy"
input_csv = "obj_det_anot"
output_csv = "rel_pred_anot_spc"

os.makedirs(os.path.join(dataset_root, output_csv), exist_ok=True)

input_csv_files = sorted(glob.glob(os.path.join(dataset_root, input_csv, "*.csv")))

# images가 더 많아서 분류작업이 필요함
for each_input_csv in input_csv_files:
    start_iter = time.time()
    print("start", each_input_csv)
    # 2. read csv
    df = pd.read_csv(each_input_csv)

    df['key'] = 1
    df['index'] = df.index
    df.drop('class_confidence', axis=1, inplace=True)

    # 철자 오류로 model에 문제 생기는 것 미연에 방지하기 위하여,
    # spell = spellchecker.SpellChecker()
    # df['class'] = df['class'].apply(lambda x: spell.correction(x))

    # compute the Cartesian product of the DataFrame based on the key column
    combinations = pd.merge(df, df, on='key', suffixes=('_sub', '_obj'))


    # remove the key column and duplicate rows
    combinations = combinations[combinations['index_sub'] != combinations['index_obj']]
    combinations = combinations[(combinations['xmax_sub'] - combinations['xmin_sub'])*(combinations['ymax_sub'] - combinations['ymin_sub'])
                                <= (combinations['xmax_obj'] - combinations['xmin_obj'])*(combinations['ymax_obj'] - combinations['ymin_obj'])]

    combinations.drop('key', axis=1, inplace=True)

    # 여기서 부터 짜주면 될듯

    combinations['rel'] = " "
    combinations['semantic'] = False

    # 일단은 귀찮으니까 hardcoding

    columns = list(combinations.columns)

    combinations.drop('img_name_obj', axis=1, inplace=True)
    combinations = combinations[[
        'class_sub', 'class_tmp_sub', 'class_obj', 'class_tmp_obj',
        'index_sub',  'index_obj', 'rel', 'semantic',
        'xmin_sub', 'ymin_sub', 'xmax_sub', 'ymax_sub',
        'xmin_obj', 'ymin_obj', 'xmax_obj', 'ymax_obj', 'img_name_sub']]

    # 6. save
    name = each_input_csv.split("\\")[-1].split(".")[0]
    combinations.to_csv(os.path.join(dataset_root, output_csv, name + '.csv'), index=False)
    print(f'done @ {time.time() - start_iter}')

print(f'done @ {time.time() - start}')
