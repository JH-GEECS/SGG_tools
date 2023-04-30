import csv
import json
import pandas as pd
import glob
import os
import numpy as np
import time
import csv

"""
1. prediction json file을 row by row로 읽는다.
2.

"""
# 쉬움 2의 output 사용
prediction_result_path = r'E:\23.04.04\Input\custom_prediction_PE.json'
# 쉬움 파일 이름 그대로 사용
idx_to_name_json_path = r'E:\23.04.04\Input\image_data.json'
# 쉬움 파일 이름 그대로 사용
idx_to_words_json_path = r'E:\23.04.04\Input\VG-SGG-dicts-with-attri.json'
# predicate 있기 전의 csv 파일들이 있는 경로
csv_dir = r'E:\23.04.04\Input\CSV_test'
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
# predicate 작성한 것을 작성할 최대 경로
destination_dir = r'E:\23.04.04\Output'
os.makedirs(destination_dir, exist_ok=True)

manual_handle_list = []

with open(idx_to_words_json_path, 'r') as f:
    idx_to_words = json.load(f)
    # 여기서 integer value들을 각각의 predicate word로 바꿔 주어야 한다.
    idx_to_predicates = idx_to_words['idx_to_predicate']

with open(idx_to_name_json_path, 'r') as f:
    #  각각의 image index에 대하여, 이를 file이름으로 convert해주는 dict이다.
    idx_to_name = json.load(f)

with open(prediction_result_path, 'r') as f:
    prediction_result = json.load(f)
    for key in prediction_result:
        start_init = time.time()
        print(f'start the file {idx_to_name[int(key)]["image_id"]}')
        # 각각의 file을 읽기.
        each_csv = pd.read_csv(os.path.join(csv_dir, idx_to_name[int(key)]['image_id'] + '.csv'))
        # todo 여기서 새로운 column을 추가해주어야한다.
        each_csv[['pred_1','pred_2','pred_3','pred_4','pred_5']] = ""
        each_csv[['pred_1_conf', 'pred_2_conf', 'pred_3_conf', 'pred_4_conf', 'pred_5_conf']] = [0.0, 0.0, 0.0, 0.0, 0.0]
        # row by row로 읽어서, prediction 결과를 추가해준다.
        for idx, row in each_csv.iterrows():
            if row['semantic']:
                # 전체 확률 분포 가져오기

                # 이부분 예외 처리 필요, 중요!!! label은 존재하는데 model이 검출하지 못하는 경우가 존재하는데 이러한 경우에는 manual input으로 error bound로 해주어야 한다.
                try:
                    each_rel_tupe = [row['index_sub'], row['index_obj']]
                    row_of_pred = prediction_result[key]['rel_pairs'].index(each_rel_tupe)
                    each_preds = np.asarray(prediction_result[key]['rel_all_scores'][row_of_pred])[1:]
                    arg_pred_sorted_descent = np.argsort(each_preds)[::-1]
                    top_5_preds = arg_pred_sorted_descent[:5]
                    top_5_pred_prob = each_preds[top_5_preds]
                    top_5_predicates = [idx_to_predicates[str(pred + 1)] for pred in top_5_preds]
                    each_csv.loc[idx, ['pred_1','pred_2','pred_3','pred_4','pred_5']] = top_5_predicates
                    each_csv.loc[idx, ['pred_1_conf', 'pred_2_conf', 'pred_3_conf', 'pred_4_conf', 'pred_5_conf']] = top_5_pred_prob
                except ValueError as e:
                    print(f'error @ {idx_to_name[int(key)]["image_id"]}')
                    manual_handle_list.append(idx_to_name[int(key)]["image_id"])
                    print(e)
                    each_csv.loc[idx, ['pred_1','pred_2','pred_3','pred_4','pred_5']] = 'manual input required'
            else:
                continue
        print(f'finished the file {idx_to_name[int(key)]["image_id"]} @ {time.time() - start_init}')
        each_csv.to_csv(os.path.join(destination_dir, idx_to_name[int(key)]['image_id'] + '.csv'), index=False)

with open(os.path.join(destination_dir, 'manual_handle_list.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(manual_handle_list)
