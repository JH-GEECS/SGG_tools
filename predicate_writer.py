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

prediction_result_path = r'Z:\assistant\assistant_deploy\custom_prediction.json'
idx_to_name_json_path = r'Z:\assistant\assistant_deploy\image_data_rel.json'
idx_to_words_json_path = r'Z:\assistant\assistant_deploy\VG-SGG-dicts-with-attri.json'

csv_dir = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\results'
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

destination_dir = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\result_with_preds'

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
        # row by row로 읽어서, prediction 결과를 추가해준다.
        for idx, row in each_csv.iterrows():
            if row['semantic']:
                # 전체 확률 분포 가져오기

                # 이부분 예외 처리 필요, 중요!!! label은 존재하는데 model이 검출하지 못하는 경우가 존재하는데 이러한 경우에는 manual input으로 error bound로 해주어야 한다.
                try:
                    each_rel_tupe = [row['index_sub'], row['index_obj']]
                    row_of_pred = prediction_result[key]['rel_pairs'].index(each_rel_tupe)
                    each_preds = np.asarray(prediction_result[key]['rel_all_scores'][row_of_pred])[1:]
                    top_5_preds = np.argsort(each_preds)[-5:]
                    top_5_predicates = [idx_to_predicates[str(pred + 1)] for pred in reversed(top_5_preds)]
                    each_csv.loc[idx, ['pred_1','pred_2','pred_3','pred_4','pred_5']] = top_5_predicates
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
