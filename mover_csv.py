# model parallel 구동을 위하여 image를 다른 곳으로 배분해주는 model이다.
import glob
import shutil
import os
import csv


def mover(orginal, processed, arrival):
    org_csv_list = glob.glob(os.path.join(orginal, '*.csv'))
    processed_csv_list = glob.glob(os.path.join(processed, '*.csv'))

    for org_csv_path in org_csv_list:
        csv_name = os.path.basename(org_csv_path).split('.')[0]
        original_csv_path = os.path.join(orginal, csv_name + '.csv')
        processed_csv_path = os.path.join(processed, csv_name + '.csv')
        arrival_csv_path = os.path.join(arrival, csv_name + '.csv')

        if processed_csv_path in processed_csv_list:
            shutil.copy(original_csv_path, arrival_csv_path)
            os.remove(original_csv_path)
            print(f'{original_csv_path} moved to {arrival_csv_path}.')


if __name__ == '__main__':
    original = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\a_jobs_doing_now\job_20230324_part'
    processed = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\a_finished_job'
    arrival = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\a_finished_original_job'

    mover(original, processed, arrival)
    print('done')
