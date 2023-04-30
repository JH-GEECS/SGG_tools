import csv
import pandas as pd
import glob
import os
from tqdm import tqdm

def triplet_remove_statistics(csv_dir, dest_path):
    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))

    # key는 [class_sub, class_obj]가 되고, value는 True이거나 false가 된다.
    statistics = {}

    # 해당 for 문을 통해서 전체 csv 접근
    for csv_idx, each_csv in tqdm(enumerate(csv_list), total=len(csv_list)):
        each_df = pd.read_csv(each_csv)
        for idx, row in each_df.iterrows():
            vector = (row['sclass'], row['oclass'])
            vector_transpose = (row['oclass'], row['sclass'])
            # predicate가 작성이 안되면 useless를 True로 해주기
            # 하나의 dataframe에 대해서 한 vector에 대해서 predicate가 없고, vector_transpose도 없는 경우만, useless를 True로 해준다.
            useless = (not isinstance(row['rel_refined'], str)) and \
                      (len(each_df[(each_df['sclass'] == vector[1]) & (each_df['oclass'] == vector[0])]) == 0)

            if useless:
                if (vector in statistics) and (statistics[vector] == False):
                    continue
                else:
                    statistics[vector] = True

            # tuple 처럼 처리하기 위해서
            else:
                if (vector in statistics) or (vector_transpose in statistics):
                    statistics[vector] = False
                    statistics[vector_transpose] = False

    os.makedirs(dest_path, exist_ok=True)
    csv_path = os.path.join(dest_path, 'triplet_remove_statistics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_sub', 'class_obj', 'useless'])
        for key, value in statistics.items():
            if value:
                writer.writerow([key[0], key[1], value])

    test = 1

if __name__ =='__main__':
    csv_dir = r'Z:\assistant\23.04.14\Files\Output_CSV_test'
    dest_path = r'E:\23.04.14\Output2'
    triplet_remove_statistics(csv_dir, dest_path)
