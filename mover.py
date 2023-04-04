#label이 완료된 파일을 옮기기 위한 용도이다.
import glob
import shutil
import os
import csv

def mover(file_list_path, depart_dir, arrival_dir):
    with open(file_list_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)

        file_list = [row[0] for row in reader]
    file_list = [file_name for file_name in file_list]
    file_list = [os.path.join(depart_dir, file_name) for file_name in file_list]
    for file_path in file_list:
        shutil.copy(file_path, arrival_dir)
        shutil.copy(file_path, os.path.join(depart_dir, 'processed'))
        os.remove(file_path)
        print(f'{file_path} moved to {arrival_dir}.')


if __name__ == '__main__':
    file_list_path = r'C:\model_temp\dataset\progress_report\prometus-use-2.csv'
    depart_dir = r'C:\model_temp\dataset\prometeus-egg1_bak-use'
    arrival_dir = r'C:\model_temp\dataset\prometus-use-2'

    mover(file_list_path, depart_dir, arrival_dir)
    print('done')