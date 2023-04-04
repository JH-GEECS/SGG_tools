import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import time


def cv2_predicate_writer(csv_dir, image_dir, dest_dir):
    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))
    img_list = glob.glob(os.path.join(image_dir, '*.jpg'))

    for csv_idx, each_csv in enumerate(csv_list):
        start_time = time.time()
        print(f'start {each_csv}th images, each_csv: {each_csv}')
        each_df = pd.read_csv(each_csv)

        each_img = cv2.imread(os.path.join(image_dir, os.path.basename(each_csv).split('.')[0] + '.jpg'))
        H_raw, W_raw, _ = each_img.shape
        each_img = cv2.resize(each_img, (1280, 720))
        H, W, _ = each_img.shape
        H_factor, W_factor = H / H_raw, W / W_raw

        writen_obj = {}
        sentinel = True

        for idx, row in each_df.iterrows():

            if not row['semantic']:
                continue

            center_sub_x, center_sub_y = int(W_factor * (row['xmin_sub'] + row['xmax_sub']) / 2), int(
                H_factor * (row['ymin_sub'] + row['ymax_sub']) / 2)
            center_obj_x, center_obj_y = int(W_factor * (row['xmin_obj'] + row['xmax_obj']) / 2), int(
                H_factor * (row['ymin_obj'] + row['ymax_obj']) / 2)

            if sentinel:
                cv2.rectangle(each_img, (int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                              (int(W_factor * row['xmax_sub']), int(H_factor * row['ymax_sub'])), (0, 0, 255), 1)
                cv2.putText(each_img,
                            text=f"{row['class_tmp_sub']}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org= (int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                            fontScale=0.5,
                            color=(0, 0, 255),
                            thickness=2,
                            bottomLeftOrigin=False)
                writen_obj[row['class_tmp_sub']] = True
            else:
                if row['class_tmp_sub'] in writen_obj.keys():
                    pass
                else:
                    cv2.rectangle(each_img, (int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                                  (int(W_factor * row['xmax_sub']), int(H_factor * row['ymax_sub'])), (0, 0, 255), 1)
                    cv2.putText(each_img,
                                text=f"{row['class_tmp_sub']}",
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                org=(int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                                fontScale=0.5,
                                color=(0, 0, 255),
                                thickness=2,
                                bottomLeftOrigin=False)
                    writen_obj[row['class_tmp_sub']] = True


            if sentinel:
                cv2.rectangle(each_img, (int(W_factor * row['xmin_obj']), int(H_factor * row['ymin_obj'])),
                              (int(W_factor * row['xmax_obj']), int(H_factor * row['ymax_obj'])), (255, 0, 0), 1)

                cv2.putText(each_img,
                            text=f"{row['class_tmp_obj']}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W_factor * row['xmin_obj']), int(H_factor * row['ymin_obj'])),
                            fontScale=0.5,
                            color=(255, 0, 0),
                            thickness=2,
                            bottomLeftOrigin=False)
                writen_obj[row['class_tmp_obj']] = True
            else:
                if row['class_tmp_obj'] in writen_obj.keys():
                    pass
                else:
                    cv2.rectangle(each_img, (int(W_factor * row['xmin_obj']), int(H_factor * row['ymin_obj'])),
                                  (int(W_factor * row['xmax_obj']), int(H_factor * row['ymax_obj'])), (255, 0, ), 1)
                    cv2.putText(each_img,
                                text=f"{row['class_tmp_obj']}",
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                org=(int(W_factor * row['xmin_obj']), int(H_factor * row['ymin_obj'])),
                                fontScale=0.5,
                                color=(255, 0, 0),
                                thickness=2,
                                bottomLeftOrigin=False)
                    writen_obj[row['class_tmp_obj']] = True



            cv2.line(each_img,
                     (int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                     (int(W_factor * row['xmin_obj']), int(H_factor * row['ymin_obj'])),
                     (0, 255, 0),
                     2)


            cv2.putText(
                each_img,
                text=f"{row['rel']}",
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                org=(int(W_factor * (row['xmin_sub'] * 1.5 +row['xmin_obj'] * 0.5)/2),
                     int(H_factor * (row['ymin_sub'] * 1.5 +row['ymin_obj'] * 0.5)/2)),
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=2,
                bottomLeftOrigin=False
            )

            sentinel = False

        cv2.putText(each_img,
                    text=f'num predicates: {each_df[each_df["semantic"] == True]["index"].count()} ',
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    org=(int(W*0.01), int(H*0.99)),
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2,
                    bottomLeftOrigin=False)
        cv2.imwrite(os.path.join(dest_dir, os.path.basename(each_csv).split('.')[0] + '.png') , each_img)
        print(f'end {each_csv}th images, each_csv: {each_csv}, time: {time.time() - start_time}')

if __name__ == '__main__':
    csvdir = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\a_transfer\20230326\finished_job_20230320'
    image_dir = r'Z:\assistant\assistant_deploy\image_processed'
    dest_dir = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\a_transfer\20230326\images_20230320'
    cv2_predicate_writer(csvdir, image_dir, dest_dir)