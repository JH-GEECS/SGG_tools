# 일시적으로 빈칸이 되어있는 file을 다루기 위하여

import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import time

def blanker_setter(origin_csv, pred_csv, dest_csv):
    origin_csv_list = glob.glob(os.path.join(origin_csv, '*.csv'))
    pred_csv_list = glob.glob(os.path.join(pred_csv, '*.csv'))

    for idx, each_csv in enumerate(pred_csv_list):
        start_time = time.time()
        print(f'{idx}th csv: {each_csv}')
        each_df = pd.read_csv(each_csv)

        if not each_df.isnull().values.any():
            continue
        else:
            for idx, row in each_df.iterrows():
                if pd.isna(row['index_sub']):
                    each_org_df = pd.read_csv(os.path.join(origin_csv, os.path.basename(each_csv).split('.')[0]+'.csv'))
                    each_df.loc[idx, ['class_sub', 'class_tmp_sub', 'index_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                'ymax_sub']] = each_org_df.loc[idx, ['class_sub', 'class_tmp_sub', 'index_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                'ymax_sub']].tolist()
                    each_df.loc[idx, ['class_obj', 'class_tmp_obj', 'index_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                         'ymax_obj']] = each_org_df.loc[idx, ['class_sub', 'class_tmp_sub', 'index_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                'ymax_sub']].tolist()
                else:
                    continue

        # each csv 저장 하기
        each_df.to_csv(os.path.join(dest_csv, os.path.basename(each_csv)), index=False)
        print(f'{idx}th csv: {each_csv} is done, {time.time() - start_time} seconds')

if __name__ == '__main__':
    origin_csv = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\result_with_preds'
    pred_csv = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\finished_job_20230320'
    dest_csv = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\finished_job_20230320_error_handle'
    blanker_setter(origin_csv, pred_csv, dest_csv)
