# model parallel 구동을 위하여 image를 다른 곳으로 배분해주는 model이다.
import glob
import shutil
import os
import csv

def mover(orginal, exist_csv, processed):

    org_img_list = glob.glob(os.path.join(orginal, '*.jpg'))
    csv_img_list = glob.glob(os.path.join(exist_csv, '*.png'))

    for depth_file_path in csv_img_list:
        csv_img_name = os.path.basename(depth_file_path).split('.')[0]
        orginal_img_path = os.path.join(orginal, csv_img_name + '.jpg')
        processed_img_path = os.path.join(processed, csv_img_name + '.jpg')

        shutil.copy(orginal_img_path, processed_img_path)
        print(f'{orginal_img_path} copied to {processed_img_path}.')


if __name__ == '__main__':
    original = r'C:\model_temp\dataset\annotated_imgs-use'
    exist_csv = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\finished_job_20230320_error_handle'
    arrival = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\visualize_20230323_mask'


    mover(original, exist_csv, arrival)
    print('done')