import json
import os
from PIL import Image
import glob
import time
import pandas as pd
import time

# 매우 중요 relationship이 0개인 image에 대해서는 idx 부여하는 것을 skip 해야 한다.

# 그런데 전체 image에 대해서 접근하는 것도 필요하다.

# 여기에서는 image dir에서 image들을 받아 드린 이후에, 이들을 sort한다
# 이후 visual genome styple json file과 image_name to idx dict를 만든다.
# 이후 2 개의 file들을 저장한다.
def image_to_json(image_dir, file_path, file_path_2):
    with open(file_path, 'w') as f1:
        with open(file_path_2, 'w') as f2:
            image_list = glob.glob(os.path.join(image_dir, '*.jpg'))
            image_list.sort()
            visual_genome_list = []
            image_name_to_idx = {}
            for idx, image_path in enumerate(image_list):
                start_time = time.time()
                print(f'start {idx}th image')
                file_name = os.path.basename(image_path).split('.')[0]
                image_name_to_idx[file_name] = idx  # dict 생성기

                # image opener
                each_img = Image.open(image_path)
                width, height = each_img.size

                image_dict = {}
                image_dict['width'] = width
                image_dict['height'] = height
                image_dict['url'] = ''
                image_dict['image_id'] = file_name

                image_dict['coco_id'] = None
                image_dict['flickr_id'] = None
                image_dict['anti_prop'] = 0.0
                visual_genome_list.append(image_dict)
                print(f'done {idx}th image @ {time.time() - start_time}')

            # json write
            json.dump(image_name_to_idx, f2)
        json.dump(visual_genome_list, f1)

def image_to_json_for_rel(image_dir, file_path, file_path_2, rel_dir):
    # pandas 기준으로 해야할 것으로 생각된다.
    with open(file_path, 'w') as f1:
        with open(file_path_2, 'w') as f2:
            visual_genome_list = []
            image_name_to_idx = {}

            csv_path_list = sorted(glob.glob(os.path.join(rel_dir, '*.csv')))

            cum_idx = 0
            for idx, csv_path in enumerate(csv_path_list):
                start_time = time.time()
                print(f'start {idx}th image')
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
                print(f'done {idx}th image @ {time.time() - start_time}')

            json.dump(image_name_to_idx, f2)
        json.dump(visual_genome_list, f1)


if __name__ == "__main__":
    image_dir = r'Z:\bak_sgb\datasets\vg\debug_20230425\Image'
    root_dir = r'Z:\bak_sgb\datasets\vg\debug_20230425'
    rel_dir = r'Z:\bak_sgb\datasets\vg\debug_20230425\CSV_test'
    file_name_1 = 'image_data_rel.json'
    file_name_2 = 'image_name_to_idx_rel.json'

    # image_to_json(image_dir, os.path.join(root_dir, file_name_1), os.path.join(root_dir, file_name_2))
    image_to_json_for_rel(image_dir, os.path.join(root_dir, file_name_1), os.path.join(root_dir, file_name_2), rel_dir)
