import h5py
import os
import json
import glob
import pandas as pd
import numpy as np
import time

# 해당 부분에서는 h5 file을 분석하느 부분이다.
def h5_analyser(path_image_data_json, path_image_id_to_idx_json,
                object_dir, rel_model_anot, path_original_h5, path_result_h5,
                object_conversion_table):

    with h5py.File(path_result_h5, 'w') as f2:
        with h5py.File(path_original_h5, 'r') as f1:
            for name, dataset in f1.items():
                basis = dataset.shape
                if len(basis) == 1:
                    new_dataset = f2.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dataset.dtype)
                else:
                    new_dataset = f2.create_dataset(name, shape=(0, basis[1]), maxshape=(None, basis[1]), dtype=dataset.dtype)

        # data writer part cf) image_data.iloc[0]
        with open(path_image_data_json, 'r') as f:
            image_data = json.load(f)
            image_data = pd.DataFrame(image_data)

        # dict 형이므로, 이를 이용하여 idx를 찾는다.
        with open(path_image_id_to_idx_json, 'r') as f:
            image_name_to_idx = json.load(f)

        # object의 class를 network가 받아들일 수 있는 것으로 변형해주는 부분
        obj_conv = pd.read_csv(object_conversion_table)
        obj_conv = obj_conv.set_index(['word'])


        # 여기서부터 중요함 annotated 된 것에 한하여 이를 검사를 해야함
        # 또한 매번 image가 추가 될때 마다, path_image_data_json와 image_name_to_idx를 잘 정의 해야함
        # path_image_data_json만 cumulative하게 잘 정의 한다면 문제는 없다.

        # 이제 annotated 된것에 한하여 h5로 convert code를 작성하고자 한다.
        object_detection_anot = sorted(glob.glob(os.path.join(object_dir, '*.csv')))
        rel_model_anot_dir = rel_model_anot
        rel_model_anot = sorted(glob.glob(os.path.join(rel_model_anot, '*.csv')))

        # 아무래도 전체적으로는 numpy을 통해서 처리하고 이를 저장하는 형태로 하는 것이 좋을 것으로 생각된다.

        num_imgs = 0
        num_objs = 0
        num_rels = 0

        big_boxes1024_list = []
        big_boxes512_list = []

        # 여기는 partitioning을 위한 곳
        big_imgs_bbox_list_raw = []
        big_imgs_rel_list_raw = []

        big_bboxes_labels_list = []
        big_rels_idx_comb_list_start = []
        big_rels_idx_comb_list_residual = []

        for idx, each_object_detection_anot in enumerate(object_detection_anot):
            start_time = time.time()
            print(f'{idx}th csv processing, {each_object_detection_anot}')



            file_name = os.path.basename(each_object_detection_anot).split('.')[0]
            # file_name = file_name.split('_')[0] + '_' + file_name.split('_')[1]

            # detection annotation에서 semantic relation이 0이면 skip
            if os.path.join(rel_model_anot_dir, file_name + '.csv') in rel_model_anot:
                rel_anot_df = pd.read_csv(os.path.join(rel_model_anot_dir, file_name + '.csv'))
                if rel_anot_df[rel_anot_df['semantic'] == True]['semantic'].count() == 0:
                    print(f'{idx}th csv no rel!!!')
                    continue


            num_imgs += 1

            # image size의 경우에는 integer conversion 필수

            det_anot_df = pd.read_csv(each_object_detection_anot)
            img_width, img_height = image_data.loc[image_name_to_idx[file_name], ['width', 'height']].values
            obj_width = (det_anot_df['xmax'] - det_anot_df['xmin']).values
            obj_height = (det_anot_df['ymax'] - det_anot_df['ymin']).values
            obj_xc = ((det_anot_df['xmax'] + det_anot_df['xmin']) / 2).values
            obj_yc = ((det_anot_df['ymax'] + det_anot_df['ymin']) / 2).values

            bbox_coords = np.vstack((obj_xc, obj_yc, obj_width, obj_height)).T
            big_boxes1024_list.extend((bbox_coords * 1024 / max(img_width, img_height)).astype(np.int32))
            big_boxes512_list.extend((bbox_coords * 512 / max(img_width, img_height)).astype(np.int32))

            obj_len = len(det_anot_df)
            big_imgs_bbox_list_raw.append([obj_len])

            # 오직 true relatio만을 저장해야 한다. 여기 매우 신경써서 작성하기
            if os.path.join(rel_model_anot_dir ,file_name+'.csv') in rel_model_anot:
                rel_anot_df = pd.read_csv(os.path.join(rel_model_anot_dir ,file_name+'.csv'))
                # rel_len = len(rel_anot_df)
                # big_imgs_rel_list_raw.append([rel_len])

                # todo 여기 meaning을 모르겠다.
                # rel_labels = [[obj_conv.loc[x][-1]] for x in rel_anot_df['rel'].values]
                # big_bboxes_labels_list.extend(rel_labels)

                semantic_rel_df = rel_anot_df[rel_anot_df['semantic'] == True]
                rel_len = len(semantic_rel_df)
                big_imgs_rel_list_raw.append([rel_len])
                rel_obj_idxs = semantic_rel_df[['index_sub', 'index_obj']].values
                big_rels_idx_comb_list_residual.extend(rel_obj_idxs)

                rel_obj_idxs_cum = rel_obj_idxs + num_objs
                big_rels_idx_comb_list_start.extend(rel_obj_idxs_cum)
                num_rels += rel_len

                print(f'done a file, {time.time() - start_time} seconds')

            num_objs += obj_len  # 누계를 이용한 방법
            # 여기에 문제가 있다. lantern?
            obj_labels = [[obj_conv.loc[x][-1]] for x in det_anot_df['class'].values]  # extend로 붙여야 할듯
            big_bboxes_labels_list.extend(obj_labels)

        ### 매우 중요 !!! 위의 list들을 저장하는 과정을 거쳐야 한다!!!

        nrows = f2['boxes_1024'].shape[0]
        f2['boxes_1024'].resize((nrows + num_objs, 4))
        f2['boxes_1024'][nrows:] = np.array(big_boxes1024_list, dtype=np.int32)

        nrows = f2['boxes_512'].shape[0]
        f2['boxes_512'].resize((nrows + num_objs, 4))
        f2['boxes_512'][nrows:] = np.array(big_boxes1024_list, dtype=np.int32)

        nrows = f2['labels'].shape[0]
        f2['labels'].resize((nrows + num_objs,1))
        f2['labels'][nrows:] = np.array(big_bboxes_labels_list, dtype=np.int64)

        nrows = f2['img_to_first_box'].shape[0]
        f2['img_to_first_box'].resize((nrows + num_imgs,))
        f2['img_to_first_box'][nrows:] = (np.cumsum(np.array([[0]] + big_imgs_bbox_list_raw, dtype=np.int32)))[:-1]

        nrows = f2['img_to_last_box'].shape[0]
        f2['img_to_last_box'].resize((nrows + num_imgs,))
        f2['img_to_last_box'][nrows:] = (np.cumsum(np.array([[0]] + big_imgs_bbox_list_raw, dtype=np.int32)))[1:] - 1

        nrows = f2['img_to_first_rel'].shape[0]
        f2['img_to_first_rel'].resize((nrows + num_imgs,))
        f2['img_to_first_rel'][nrows:] = (np.cumsum(np.array([[0]] + big_imgs_rel_list_raw, dtype=np.int32)))[:-1]

        nrows = f2['img_to_last_rel'].shape[0]
        f2['img_to_last_rel'].resize((nrows + num_imgs,))
        f2['img_to_last_rel'][nrows:] = (np.cumsum(np.array([[0]] + big_imgs_rel_list_raw, dtype=np.int32)))[1:] - 1

        nrows = f2['relationships'].shape[0]
        f2['relationships'].resize((nrows + num_rels, 2))
        f2['relationships'][nrows:] = np.array(big_rels_idx_comb_list_start, dtype=np.int32)

        # 이 아래 부분은 거의 상수값의 느낌으로 넣어야 하므로 이렇게 한다.
        nrows = f2['active_object_mask'].shape[0]
        f2['active_object_mask'].resize((nrows + num_objs, 1))
        f2['active_object_mask'][nrows:] = np.ones((num_objs, 1), dtype=bool)

        nrows = f2['attributes'].shape[0]
        f2['attributes'].resize((nrows + num_objs, 10))
        f2['attributes'][nrows:] = np.zeros((num_objs, 10), dtype=np.int64)

        nrows = f2['predicates'].shape[0]
        f2['predicates'].resize((nrows + num_rels, 1))
        temp = np.ones((num_rels, 1), dtype=np.int64) * 32
        f2['predicates'][nrows:] = temp

        nrows = f2['split'].shape[0]
        f2['split'].resize((nrows + num_imgs,))
        temp = np.ones((num_imgs, ), dtype=np.int32) * 2
        f2['split'][nrows:] = temp

        """
        # get the current number of rows in the dataset
        nrows = dataset.shape[0]
        
        # resize the dataset to accommodate the new rows
        dataset.resize((nrows + new_data.shape[0], new_data.shape[1]))
        
        # add the new rows to the dataset
        dataset[nrows:, :] = new_data
        
        """
        test_trigger = True

def json_analyser(dir, path):
    with open(path, 'r') as f:
        data = json.load(f)
        test = 1

if __name__ == "__main__":

    path_image_data_json = r'Z:\assistant\assistant_deploy\image_data_rel.json'
    path_image_id_to_idx_json = r'Z:\assistant\assistant_deploy\image_name_to_idx_rel.json'
    object_dir = r'Z:\assistant\assistant_deploy\obj_det_anot_spc_2'
    rel_model_anot = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\results'
    path_original_h5 = r'Z:\Scene-Graph-Benchmark.pytorch\datasets\vg\VG-SGG-with-attri.h5'
    path_result_h5 = r'Z:\assistant\assistant_deploy\VG-SGG-with-attri.h5'
    object_conversion_table = r'Z:\assistant\assistant_deploy\word_refined_final_done.csv'

    # path = os.path.join(dir, h5_path, h5_path_2)
    # have to load,
    h5_analyser(path_image_data_json, path_image_id_to_idx_json,
                object_dir, rel_model_anot, path_original_h5, path_result_h5,
                object_conversion_table)