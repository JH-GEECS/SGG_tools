import json
import os
from PIL import Image
import glob
import time
import pandas as pd
from tqdm import tqdm
import h5py
import os
import json
import glob
import numpy as np


# 해당 부분에서는 h5 file을 분석하느 부분이다.
def h5_analyser(path_image_data_json, path_image_id_to_idx_json,
                object_dir, rel_model_anot, path_original_h5, path_result_h5, index_dict):
    with h5py.File(path_result_h5, 'w') as f2:
        with h5py.File(path_original_h5, 'r') as f1:
            for name, dataset in f1.items():
                basis = dataset.shape
                if len(basis) == 1:
                    new_dataset = f2.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dataset.dtype)
                else:
                    new_dataset = f2.create_dataset(name, shape=(0, basis[1]), maxshape=(None, basis[1]),
                                                    dtype=dataset.dtype)

        # data writer part cf) image_data.iloc[0]
        with open(path_image_data_json, 'r') as f:
            image_data = json.load(f)
            image_data = pd.DataFrame(image_data)

        # dict 형이므로, 이를 이용하여 idx를 찾는다.
        with open(path_image_id_to_idx_json, 'r') as f:
            image_name_to_idx = json.load(f)

        # object의 class를 network가 받아들일 수 있는 것으로 변형해주는 부분
        with open(index_dict, 'r') as f:
            index_dict = json.load(f)
        # obj_conv = obj_conv.set_index(['word'])

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

        for idx, each_object_detection_anot in tqdm(enumerate(object_detection_anot), total=len(object_detection_anot)):
            start_time = time.time()

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
            if os.path.join(rel_model_anot_dir, file_name + '.csv') in rel_model_anot:
                rel_anot_df = pd.read_csv(os.path.join(rel_model_anot_dir, file_name + '.csv'))
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


            num_objs += obj_len  # 누계를 이용한 방법
            # 여기에 문제가 있다. lantern? -> OOD case
            # 다행이도 별 문제 없다. object anot 에서 하는 것이므로 이부분은 spell_checked를 넣으면 될듯하다.
            obj_labels = [[index_dict['label_to_idx'][x]] for x in det_anot_df['class_refined'].values]  # extend로 붙여야 할듯
            big_bboxes_labels_list.extend(obj_labels)

        ### 매우 중요 !!! 위의 list들을 저장하는 과정을 거쳐야 한다!!!

        nrows = f2['boxes_1024'].shape[0]
        f2['boxes_1024'].resize((nrows + num_objs, 4))
        f2['boxes_1024'][nrows:] = np.array(big_boxes1024_list, dtype=np.int32)

        nrows = f2['boxes_512'].shape[0]
        f2['boxes_512'].resize((nrows + num_objs, 4))
        f2['boxes_512'][nrows:] = np.array(big_boxes1024_list, dtype=np.int32)

        nrows = f2['labels'].shape[0]
        f2['labels'].resize((nrows + num_objs, 1))
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
        temp = np.ones((num_imgs,), dtype=np.int32) * 2
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


# 매우 중요 relationship이 0개인 image에 대해서는 idx 부여하는 것을 skip 해야 한다.

def image_to_json_for_rel(image_dir, file_path, file_path_2, rel_dir):
    # pandas 기준으로 해야할 것으로 생각된다.
    with open(file_path, 'w') as f1:
        with open(file_path_2, 'w') as f2:
            visual_genome_list = []
            image_name_to_idx = {}

            csv_path_list = sorted(glob.glob(os.path.join(rel_dir, '*.csv')))

            cum_idx = 0
            for idx, csv_path in tqdm(enumerate(csv_path_list), total=len(csv_path_list)):
                start_time = time.time()
                each_pd = pd.read_csv(csv_path)
                if each_pd[each_pd['semantic'] == True]['semantic'].count() == 0:
                    continue
                else:
                    image_name = os.path.basename(csv_path).split('.')[0]
                    image_path = os.path.join(image_dir, image_name + '.jpg')
                    each_img = Image.open(image_path)
                    width, height = each_img.size

                    image_dict = {}
                    image_dict['width'] = width
                    image_dict['height'] = height
                    image_dict['url'] = ''
                    image_dict['image_id'] = image_name

                    image_dict['coco_id'] = None
                    image_dict['flickr_id'] = None
                    image_dict['anti_prop'] = 0.0
                    visual_genome_list.append(image_dict)
                    image_name_to_idx[image_name] = cum_idx
                    cum_idx += 1

            json.dump(image_name_to_idx, f2)
        json.dump(visual_genome_list, f1)


if __name__ == "__main__":
    # 제작할 데이터의 root
    root_dir = r'E:\23.04.04\Input'

    # 이미지 폴더, CSV rel 폴더
    image_dir = 'Image'
    rel_dir = 'CSV_test'

    # 수정 하지 않기
    file_name_1 = 'image_data.json'
    file_name_2 = 'image_name_to_idx.json'

    #image_to_json_for_rel(os.path.join(root_dir, image_dir), os.path.join(root_dir, 'image_data.json'),
    #                      os.path.join(root_dir, 'image_name_to_idx.json'), os.path.join(root_dir, rel_dir))

    # 여기는 크게 문제 없음 위에서 나온거 그대로
    path_image_data_json = r'Z:\iet_sgg_trial1\PENET\sgg_service\Input\image_data.json'
    path_image_id_to_idx_json = r'Z:\iet_sgg_trial1\PENET\sgg_service\Input\image_name_to_idx.json'
    
    # 여기 문제(완료, spelling check된 obj anot을 넣으면 된다.)
    object_dir = r'E:\23.04.04\Input\CSV'
    # 여기도 문제(완료, rel annot combination csv file을 넣으면 된다.)
    rel_model_anot = r'E:\23.04.04\Input\CSV_test'
    # 이건 괜찮(파일 그대로)
    path_original_h5 = r'Z:\iet_sgg_trial1\PENET\sgg_service\Input\example.h5'
    # 이건 괜찮(이것도 파일명은 유지 해야함)
    path_result_h5 = r'E:\23.04.04\Input\VG-SGG-with-attri.h5'
    index_dict = r'E:\23.04.04\Input\VG-SGG-dicts-with-attri.json'

    h5_analyser(path_image_data_json, path_image_id_to_idx_json,
                object_dir, rel_model_anot, path_original_h5, path_result_h5, index_dict)