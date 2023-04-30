import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import bayesian_predicate_predicator_improve as bpp
import scipy.sparse as sp


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
5. NB를 구현하였기 때문에, 6, 7번째 recommendatation에 대하여 bayesian classifier의 output이 들어 갈 수 있도록 변형 한다.
"""

def predicate_giver(row, vectorizer, NB_classifier):
    predicates = row[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']].tolist()
    # 전체 traversal 시에 7개면 끝, 아니면 가장 잘 쓰이는 2개 붙여주기 구현한다.
    row_x_sub_int = vectorizer.transform([row['class_sub']])
    row_x_obj_int = vectorizer.transform([row['class_obj']])
    X = sp.hstack([row_x_sub_int, row_x_obj_int])
    # 여기에서는 row wise한 data만을 입력으로하기 때문에 0번을 넣는다.
    NB_pred = NB_classifier.predict(X, topk=5)[0]
    for each_pred in NB_pred:
        if each_pred not in predicates:
            predicates.append(each_pred)
        if len(predicates) == 7:
            break

    if len(predicates) == 6:
        predicates.append('in')
    elif len(predicates) == 5:
        predicates.append('in')
        predicates.append('on')
    else:
        pass

    return predicates

def cv2_loader(csv_dir, image_dir, dest_dir, vectorizer, NB_classifier, HW_size=(1600, 900)):
    
    os.makedirs(dest_dir, exist_ok=True)
    
    csv_list = glob.glob(os.path.join(csv_dir, '*.csv'))
    img_list = glob.glob(os.path.join(image_dir, '*.jpg'))

    csv_list_in_size_raw = sorted([(each_csv, os.path.getsize(each_csv)) for each_csv in csv_list], key=lambda x: x[1])
    csv_list_in_size = [each_csv[0] for each_csv in csv_list_in_size_raw]

    # 해당 for 문을 통해서 전체 csv 접근
    for csv_idx, each_csv in enumerate(csv_list_in_size):
        each_df = pd.read_csv(each_csv)
        # 해당 for 문을 통해서 각각의 csv 접근
        """
        # todo
        0. semantic row의 생성
        1. 크기 관계를 통해서 relation 걸러내기, semantic false로 변경 -> 이건 대강 된듯
        2. 같은 class인 경우 skip하기 -> 여기는 코드로 구현 해주기
        3. class_sub가 사람이고, class_obj가 사람이 아닌 경우, transpose하기
        4. confidence 기반으로 cut 하기 여기는 조금 실험적
        """
        # 나의 프로그램에 맞추기 위한 수단
        each_df['semantic'] = True
        each_df['rel'] = ""
        # dataframe의 column rename하기
        # attribute도 바꾸고 최종적으로 dict바꾸기
        column_map_dict = {
            "sclass": "class_sub",
            "sxmin": "xmin_sub",
            "symin": "ymin_sub",
            "sxmax": "xmax_sub",
            "symax": "ymax_sub",
            "sclass_tmp": "class_tmp_sub",
            "sattribute_refine": "attribute_refine_sub",
            "sclass_refined": "class_refined_sub",
            "sclass_refined_tmp": "class_refined_tmp_sub",
            "oclass": "class_obj",
            "oxmin": "xmin_obj",
            "oymin": "ymin_obj",
            "oxmax": "xmax_obj",
            "oymax": "ymax_obj",
            "oclass_tmp": "class_tmp_obj",
            "oattribute_refine": "attribute_refine_obj",
            "oclass_refined": "class_refined_obj",
            "oclass_refined_tmp": "class_refined_tmp_obj",
        }
        each_df = each_df.rename(
            columns=column_map_dict
        )

        # transpose checker
        for idx, row in each_df.iterrows():
            # {object_1, object_2}와 {object_2, object_1}가 존재한다면, 크기가 큰것이 object가 되는 것이 맞다.
            # 이때 유의해야 할 것이, iterator를 통해서 접근할 시에 특정 값이 변경되어도 접근이 안된다는 점이다.
            # 먼저 datafarme단에서 처리를 해야 겠다.

            duplicate_checker = each_df[(each_df['class_tmp_obj'] == row['class_tmp_sub']) & (
                    each_df['class_tmp_sub'] == row['class_tmp_obj'])]
            if len(duplicate_checker) > 0:
                size_of_sub_now = (row['xmax_sub'] - row['xmin_sub']) * (row['ymax_sub'] - row['ymin_sub'])
                size_of_obj_now = (row['xmax_obj'] - row['xmin_obj']) * (row['ymax_obj'] - row['ymin_obj'])

                # 지금 상황에서 sub가 obj보다 크다면 지금을 꺼준다.
                if size_of_sub_now >= size_of_obj_now:
                    each_df.loc[idx, 'semantic'] = False

        # 중복 열 check
        each_df = each_df.drop_duplicates()

        for idx, row in each_df.iterrows():

            # 여기서 sky skip 하기
            if each_df.loc[idx, 'semantic']:
                # skip list 기반 skip
                skip_list = {'sky', 'reflection', 'cloud'}
                if row['class_sub'] in skip_list or row['class_obj'] in skip_list:
                    each_df.loc[idx, 'semantic'] = False
                    continue

                # 중복 물체 기반 skip
                human_list = ['baby', 'child', 'girl', 'boy', 'adult', 'person', 'woman', 'man', 'people']
                if (not (row['class_sub'] in human_list or row['class_obj'] in human_list)) and (
                        row['class_sub'] == row['class_obj']):
                    each_df.loc[idx, 'semantic'] = False
                    continue

                each_img = cv2.imread(os.path.join(image_dir, os.path.basename(each_csv).split('.')[0] + '.jpg'))
                H_raw, W_raw, _ = each_img.shape
                each_img = cv2.resize(each_img, HW_size)
                raw_img_pre = each_img.copy()
                H, W, _ = each_img.shape

                H_factor, W_factor = H / H_raw, W / W_raw

                # 이 지점을 통해서 predicate vector를 compose해주는 것이 좋을 것이다.
                """
                1. 만일 water class가 있는 경우라면 standing in, walking in이 추가되면 좋을 것이다.
                2. 만일 road class가 있는 경우라면 aganist가 추가 되면 좋을 것이다.
                """

                # 여기서 obj가 사람이고, sub보다 크다면 tranpose하는 것으로서 정의한다.
                # dataframe은 변경 되었는데, row data는 변경이 되지 않았기 때문이다.
                # 인간이 쥐고 있어야 하는데 왜 안되지?
                # 여기는 기존에는 index 였는데, 어쩔 수 없이 변경한다. e.g.) index_sub -> class_refined_sub

                if row['class_obj'] in human_list and row['class_sub'] not in human_list:
                    area_obj = (row['xmax_obj'] - row['xmin_obj']) * (row['ymax_obj'] - row['ymin_obj'])
                    area_sub = (row['xmax_sub'] - row['xmin_sub']) * (row['ymax_sub'] - row['ymin_sub'])
                    if area_obj > area_sub:
                        temp = row[['class_sub', 'class_tmp_sub', 'class_refined_sub', 'class_refined_tmp_sub',
                                    'attribute_refine_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                    'ymax_sub']].tolist()
                        # original dataframe 변경 code
                        each_df.loc[idx, ['class_sub', 'class_tmp_sub', 'class_refined_sub', 'class_refined_tmp_sub',
                                          'attribute_refine_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                          'ymax_sub']] = \
                            row[['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                                 'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                                 'ymax_obj']].tolist()
                        each_df.loc[idx, ['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                                          'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                                          'ymax_obj']] = temp
                        # row 변경 code
                        row[['class_sub', 'class_tmp_sub', 'class_refined_sub', 'class_refined_tmp_sub',
                             'attribute_refine_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                             'ymax_sub']] = \
                            row[['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                                 'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                                 'ymax_obj']].tolist()
                        row[['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                             'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                             'ymax_obj']] = temp

                # 일단 이 부분이 NB로부터 정보를 받아 올 수 있게 수정한다.
                # functionalize해서 이 부분 밖으로 뺴기
                predicates = predicate_giver(row, vectorizer, NB_classifier)

                """ 
                # NB 구현이 성공하였기 때문에 이 부분은 deprecate.
                water_list = ['water']
                road_list = ['road', 'grass', 'field']

                if row['class_sub'] in water_list or row['class_obj'] in water_list:
                    predicates.append('standing in')
                    predicates.append('walking in')
                elif (row['class_sub'] in human_list and row['class_obj'] in road_list) or (row['class_sub'] in road_list and row['class_obj'] in human_list):
                    predicates.append('standing on')
                    predicates.append('walking on')
                elif row['class_sub'] in road_list or row['class_obj'] in road_list:
                    predicates.append('against')
                    predicates.append('on')
                else:
                    predicates.append('on')
                    predicates.append('in')
                """

                # subject는 빨강, object는 파랑으로 합성한다.
                # 또한 subject와 object까지 작성하면 좋을듯 하다.
                cv2.rectangle(each_img, (int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                              (int(W_factor * row['xmax_sub']), int(H_factor * row['ymax_sub'])), (0, 0, 255), 3)
                cv2.putText(each_img,
                            text=f"{row['class_tmp_sub']}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W * 0.01), int(H * 0.25)),
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

                cv2.putText(each_img,
                            text=f"predcaites: 1.{predicates[0]}, 2.{predicates[1]}, 3.{predicates[2]}, 4.{predicates[3]}, 5.{predicates[4]}, 6.{predicates[5]}, 7.{predicates[6]}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W * 0.01), int(H * 0.1)),
                            fontScale=1,
                            color=(0, 255, 0),
                            thickness=2,
                            bottomLeftOrigin=False)

                # statistics는 상단에 검은색으로 한다.
                cv2.putText(each_img,
                            text=f'image: {csv_idx}/{len(csv_list)}, annot: {idx}/{len(each_df)} ',
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            org=(int(W * 0.01), int(H * 0.99)),
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
                        # 여기 약간 바꿔주기
                        elif key == ord('6'):
                            each_df.loc[idx, 'rel'] = predicates[5]
                            print('6')
                            break
                        elif key == ord('7'):
                            each_df.loc[idx, 'rel'] = predicates[6]
                            print('7')
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
                            # 중요 row transpose operations
                            temp = row[['class_sub', 'class_tmp_sub', 'class_refined_sub', 'class_refined_tmp_sub',
                                        'attribute_refine_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                        'ymax_sub']].tolist()
                            # original dataframe 변경 code
                            each_df.loc[
                                idx, ['class_sub', 'class_tmp_sub', 'class_refined_sub', 'class_refined_tmp_sub',
                                      'attribute_refine_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                      'ymax_sub']] = \
                                row[['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                                     'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                                     'ymax_obj']].tolist()
                            each_df.loc[
                                idx, ['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                                      'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                                      'ymax_obj']] = temp
                            # row 변경 code
                            row[['class_sub', 'class_tmp_sub', 'class_refined_sub', 'class_refined_tmp_sub',
                                 'attribute_refine_sub', 'xmin_sub', 'ymin_sub', 'xmax_sub',
                                 'ymax_sub']] = \
                                row[['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                                     'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                                     'ymax_obj']].tolist()
                            row[['class_obj', 'class_tmp_obj', 'class_refined_obj', 'class_refined_tmp_obj',
                                 'attribute_refine_obj', 'xmin_obj', 'ymin_obj', 'xmax_obj',
                                 'ymax_obj']] = temp

                            print('o')
                            # string_mode = True # string mode를 없애고, 단순히 transpose만 한다.
                            # transposed 기준으로 새롭게 image 작성 필요
                            # 다음 부터는 class로서 함수를 작성한다.

                            # 새롭게 정의된 sub, obj 기준으로 다시 작성
                            cv2.destroyAllWindows()
                            each_img = raw_img_pre.copy()
                            cv2.rectangle(each_img, (int(W_factor * row['xmin_sub']), int(H_factor * row['ymin_sub'])),
                                          (int(W_factor * row['xmax_sub']), int(H_factor * row['ymax_sub'])),
                                          (0, 0, 255), 3)
                            cv2.putText(each_img,
                                        text=f"{row['class_tmp_sub']}",
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        org=(int(W * 0.01), int(H * 0.25)),
                                        fontScale=1,
                                        color=(0, 0, 255),
                                        thickness=2,
                                        bottomLeftOrigin=False)

                            cv2.rectangle(each_img, (int(W_factor * row['xmin_obj']), int(H_factor * row['ymin_obj'])),
                                          (int(W_factor * row['xmax_obj']), int(H_factor * row['ymax_obj'])),
                                          (255, 0, 0), 3)
                            cv2.putText(each_img,
                                        text=f"{row['class_tmp_obj']}",
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        org=(int(W * 0.01), int(H * 0.75)),
                                        fontScale=1,
                                        color=(255, 0, 0),
                                        thickness=2,
                                        bottomLeftOrigin=False)
                            # predicates를 image의 하단에 합성한다.
                            # predicate는 초록으로 한다.

                            # transpose relation에 대해서는 한번 다시 predicate를 가져오는 것이 좋을 듯 하다.
                            predicates = predicate_giver(row, vectorizer, NB_classifier)
                            cv2.putText(each_img,
                                        text=f"predcaites: 1.{predicates[0]}, 2.{predicates[1]}, 3.{predicates[2]}, 4.{predicates[3]}, 5.{predicates[4]}, 6.{predicates[5]}, 7.{predicates[6]}",
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        org=(int(W * 0.01), int(H * 0.1)),
                                        fontScale=1,
                                        color=(0, 255, 0),
                                        thickness=2,
                                        bottomLeftOrigin=False)
                            # statistics는 상단에 검은색으로 한다.
                            cv2.putText(each_img,
                                        text=f'image: {csv_idx}/{len(csv_list)}, annot: {idx}/{len(each_df)} ',
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        org=(int(W * 0.01), int(H * 0.99)),
                                        fontScale=2,
                                        color=(255, 255, 255),
                                        thickness=2,
                                        bottomLeftOrigin=False)
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
                            each_df.loc[idx, 'rel'] = ''
                            print('p')
                            break
                        else:
                            print('please write appropriate key')
                cv2.destroyAllWindows()
            else:
                continue
        # 여기서 기존의 cloumn 명 수정 필요
        swapped_dict = {value: key for key, value in column_map_dict.items()}
        swapped_dict['rel'] = 'rel_refined'
        each_df = each_df.rename(columns=swapped_dict)
        each_df = each_df.drop('semantic', axis=1)

        # 중복열 버리기
        each_df = each_df.drop_duplicates()
        each_df.to_csv(os.path.join(dest_dir, os.path.basename(each_csv)), index=False)


if __name__ == '__main__':
    """
    csv_dir = r'D:\23.04.14\Files\230421_Predicate_200'
    image_dir = r'D:\23.04.14\Files\Input_Image'    
    dest_dir = r'D:\23.04.14\Files\230421_Predicate_200_Output_split_2'
    """

    # 아무래도 cv2_loader에 NB를 embeddding하는 것보다는, main 단에서 잘 처리 해 놓는 것이 더 좋을 것으로 보인다.

    # todo data lake 적절하게 지정 필요
    train_data_dir = r'C:\soft_links\nip_label\train_data_lake'

    X, y, vectorizer = bpp.csv_files_extractor(train_data_dir)
    NB_classifier = bpp.naive_bayes_classifier(pretrained_model=None)
    NB_classifier.train(X, y)

    csv_dir = r'C:\soft_links\nip_label\230423\230423_B_MEDIC_1000_200_CSV'
    image_dir = r'C:\soft_links\nip_label\230423\230423_B_MEDIC_1000_200_Imag'
    dest_dir = r'C:\soft_links\nip_label\230423\230423_B_MEDIC_1000_200_CSV_output'

    cv2_loader(csv_dir, image_dir, dest_dir, vectorizer, NB_classifier, HW_size=(1600, 900))
