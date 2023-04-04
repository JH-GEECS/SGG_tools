import numpy
import pandas as pd
import csv
import os
import glob

def statistics_writer(obj_path, rel_path):

    obj_dir = glob.glob(os.path.join(obj_path, '*.csv'))
    rel_dir = glob.glob(os.path.join(rel_path, '*.csv'))

    statistics_df = pd.DataFrame()
    """
    img_name
    num_obj
    num_rels
    freq_obj
    """
    img_name = []
    num_obj = []
    num_rel = []
    top3_obj = []

    for each_rel in rel_dir:
        rel_df = pd.read_csv(each_rel)
        each_obj_file_name = os.path.basename(each_rel).split('.')[0] + '_Original.csv'
        obj_df = pd.read_csv(os.path.join(obj_path, each_obj_file_name))

        img_name.append(os.path.basename(each_rel).split('.')[0])
        num_obj.append(obj_df['class'].count())
        num_rel.append(rel_df[rel_df['semantic'] == True]['semantic'].count())


    statistics_df['img_name'] = img_name
    statistics_df['num_obj'] = num_obj
    statistics_df['num_rel'] = num_rel

    statistics_df.to_csv(os.path.join(rel_path, 'statistics.csv'), index=False)
    test = 1


if __name__ == '__main__':
    obj_path = r'C:\model_temp\raw\obj_anot_1'
    rel_dir = r'C:\model_temp\dataset\prometeus-deploy-trial3\results'
    statistics_writer(obj_path, rel_dir)