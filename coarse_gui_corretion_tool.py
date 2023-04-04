import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

"""
해당 program의 목적
1. 특정 폴더에 있는 csv file을 pandas로 읽어 드린 다음에
2. 각각의 row에 대하여 predicate를 불러 오고
3. opencv를 통해서 image를 합성하는 task를 진행한다.
    이때, 합성된 image에는 전체 남은 image와 한 image에 대하여 남은 annotation의 수를 표시한다.
    하단에는, top5의 annotation을 표시한다.
4. opencv를 통해서 key 입력을 받는다면
    1, 2, 3, 4, 5, I, O, P를 읽어 들이고
    1, 2, 3, 4, 5의 경우에는 바로 다음으로
    I와 O의 경우에는 console에서 입력을 받고
    P의 경우에는 semantic false로 수정하고 다음으로 넘어간다.

"""

def cv2_loader(csv_dir, image_dir, dest_dir):
    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))
    img_list = glob.glob(os.path.join(image_dir, '*.jpg'))

    for csv_idx, each_csv in enumerate(csv_list):
        each_df = pd.read_csv(each_csv)
        for idx, row in each_df.iterrows():

            # 여기서 sky skip 하기
            if row['semantic']:
                skip_list = {'sky', 'fence', 'sign', 'reflection', 'grass'}
                if row['class_sub'] in skip_list or row['class_obj'] in skip_list:
                    each_df.loc[idx, 'semantic'] = False
                    continue

                each_img = cv2.imread(os.path.join(image_dir, os.path.basename(each_csv).split('.')[0] + '.jpg'))
                H_raw, W_raw, _ = each_img.shape
                each_img = cv2.resize(each_img, (1280, 720))
                H, W, _ = each_img.shape

                H_factor, W_factor = H / H_raw, W / W_raw

                predicates = row[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].tolist()

                # subject는 빨강, object는 파랑으로 합성한다.
                # 또한 subject와 object까지 작성하면 좋을듯 하다.
                cv2.rectangle(each_img, (int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                              (int(W_factor * row['xmax_sub']), int(H_factor * row['ymax_sub'])), (0, 0, 255), 3)
                cv2.putText(each_img,
                            text=f"{row['class_tmp_sub']}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W*0.01), int(H*0.25)),
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2,
                            bottomLeftOrigin=False)

                cv2.rectangle(each_img, (int(W_factor * row['xmin_obj']), int(H_factor * row['ymin_obj'])),
                              (int(W_factor * row['xmax_obj']), int(H_factor * row['ymax_obj'])), (255, 0, 0), 3)
                cv2.putText(each_img,
                            text=f"{row['class_tmp_obj']}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W * 0.01), int(H * 0.75)),
                            fontScale=1,
                            color=(255, 0, 0),
                            thickness=2,
                            bottomLeftOrigin=False)

                # predicates를 image의 하단에 합성한다.
                # todo 다시쓰기
                # predicate는 초록으로 한다.
                cv2.putText(each_img,
                            text=f"predcaites: 1.{row['pred_1']}, 2.{row['pred_2']}, 3.{row['pred_3']}, 4.{row['pred_4']}, 5.{row['pred_5']}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W*0.01), int(H*0.1)),
                            fontScale=1,
                            color=(0, 255, 0),
                            thickness=2,
                            bottomLeftOrigin=False)

                # statistics는 상단에 검은색으로 한다.
                cv2.putText(each_img,
                            text=f'image: {csv_idx}/{len(csv_list)}, annot: {idx}/{len(each_df)} ',
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W*0.01), int(H*0.99)),
                            fontScale=2,
                            color=(255, 255, 255),
                            thickness=2,
                            bottomLeftOrigin=False)

                # predicate 입력을 받는다.
                string_mode = False
                string = ""
                raw_img = each_img.copy()
                while True:
                    cv2.imshow('image', each_img)
                    key = cv2.waitKey(0)

                    # 강제 종료를 하기 위해서
                    if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                        print('force quit')
                        return 0

                    if string_mode:
                        # 윈도우여서 13이 enter
                        if key == 13:
                            string_mode = False
                            each_df.loc[idx, 'rel'] = string.strip()
                            print('string mode off')
                            string = ""
                            break
                        # 여기는 backspace
                        elif key == 8:
                            if len(string) > 0:
                                string = string[:-1]
                            else:
                                continue
                            print(string)
                        # 여기는 esc
                        elif key == 27:
                            string_mode = False
                            string = ""
                            print('string mode off')
                            cv2.destroyAllWindows()
                            each_img = raw_img.copy()
                            cv2.imshow('image', raw_img)
                        else:
                            string += chr(key)
                            print(string)

                    else:
                        if key == ord('1'):
                            print('1')
                            each_df.loc[idx, 'rel'] = predicates[0]
                            break
                        elif key == ord('2'):
                            each_df.loc[idx, 'rel'] = predicates[1]
                            print('2')
                            break
                        elif key == ord('3'):
                            each_df.loc[idx, 'rel'] = predicates[2]
                            print('3')
                            break
                        elif key == ord('4'):
                            each_df.loc[idx, 'rel'] = predicates[3]
                            print('4')
                            break
                        elif key == ord('5'):
                            each_df.loc[idx, 'rel'] = predicates[4]
                            print('5')
                            break
                        elif key == ord('i'):
                            print('i')
                            string_mode = True
                            cv2.destroyAllWindows()
                            cv2.putText(each_img,
                                        text=f"manual input mode",
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        org=(int(W * 0.01), int(H * 0.5)),
                                        fontScale=1,
                                        color=(255, 255, 255),
                                        thickness=2,
                                        bottomLeftOrigin=False)
                            cv2.imshow('image', each_img)
                        elif key == ord('o'):
                            temp = row[['class_sub', 'class_tmp_sub', 'index_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub', 'ymax_sub']].tolist()
                            each_df.loc[idx, ['class_sub', 'class_tmp_sub', 'index_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub', 'ymax_sub']] = \
                                row[['class_obj', 'class_tmp_obj', 'index_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj', 'ymax_obj']].tolist()
                            each_df.loc[idx, ['class_obj', 'class_tmp_obj', 'index_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj', 'ymax_obj']] = temp
                            print('o')
                            string_mode = True
                            cv2.destroyAllWindows()
                            cv2.putText(each_img,
                                        text=f"transposed input mode",
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        org=(int(W * 0.01), int(H * 0.5)),
                                        fontScale=1,
                                        color=(255, 255, 255),
                                        thickness=2,
                                        bottomLeftOrigin=False)
                            cv2.imshow('image', each_img)
                        elif key == ord('p'):
                            each_df.loc[idx, 'semantic'] = False
                            print('p')
                            break
                        else:
                            print('please write appropriate key')



                cv2.destroyAllWindows()
            else:
                continue
        each_df.to_csv(os.path.join(dest_dir, os.path.basename(each_csv)), index=False)

if __name__ == '__main__':
    csv_dir = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\a_jobs_doing_now\job_20230324'
    image_dir = r'Z:\assistant\assistant_deploy\image_processed'
    dest_dir = r'Z:\assistant\assistant_deploy\rel_pred_anot_spc\a_finished_job'
    cv2_loader(csv_dir, image_dir, dest_dir)