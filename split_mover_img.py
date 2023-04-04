#label이 완료된 파일을 옮기기 위한 용도이다.
import glob
import shutil
import os
import csv

def mover(orginal, processed):

    org_img_list = glob.glob(os.path.join(orginal, '*.jpg'))

    for org_file_path in org_img_list:
        org_img_name = os.path.basename(org_file_path).split('.')[0]
        processed_img_path = os.path.join(processed, org_img_name + '.jpg')

        if not os.path.exists(org_file_path):
            continue

        shutil.copy(org_file_path, processed_img_path)
        os.remove(org_file_path)
        print(f'{org_file_path} moved to {processed_img_path}.')


if __name__ == '__main__':
    original = r'Z:\assistant\assistant_deploy\image'
    processed = r'Z:\assistant\assistant_deploy\image_parse_temp5'


    mover(original, processed)
    print('done')