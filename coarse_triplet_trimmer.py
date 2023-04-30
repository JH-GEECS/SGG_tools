import csv
import pandas as pd
import glob
import os
from tqdm import tqdm

def triplet_remove_statistics(csv_dir, dest_dir):

    if csv_dir == dest_dir:
        raise ValueError("csv_dir and dest_dir must be different.")

    os.makedirs(dest_dir, exist_ok=True)

    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))

    # 해당 for 문을 통해서 전체 csv 접근
    for csv_idx, each_csv in tqdm(enumerate(csv_list), total=len(csv_list)):
        each_df = pd.read_csv(each_csv)
        each_df = each_df.drop('rel_refined', axis=1)
        dispose_condition = each_df['rel_refined.1'].isnull()
        each_df.loc[dispose_condition, 'rel_refined.1'] = "Disposal"
        each_df = each_df.rename(columns={'rel_refined.1': 'rel_refined'})
        each_df.to_csv(os.path.join(dest_dir, os.path.basename(each_csv)), index=False)

if __name__ == "__main__":
    csv_dir = r'C:\soft_links\nip_label\230423\230423_B_MEDIC_1000_200_CSV_output'
    dest_dir = r'C:\soft_links\nip_label\230423\230423_B_MEDIC_1000_200_CSV_output_refined_v2'
    triplet_remove_statistics(csv_dir, dest_dir)
