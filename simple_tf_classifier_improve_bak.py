"""
1. 매우 간단한 classifier를 만드는 것을 목적으로 한다.
2. csv file에서 object간의 거리, 크기 비율, depth 차이, semantic embedding을 이용하여 classifier를 만든다.
3. 이후에, 이를 이용하여, object간의 relation이 유의미 하면 True, 아니면 False로 분류한다.

"""
import numpy as np
# classifier model part
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import gensim.downloader
import cv2
import time
import pickle

# dataloader
import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class tensor_X_str(Enum):
    avg_depth_sub = 0
    x_centre_norm_sub = 1
    y_centre_norm_sub = 2
    avg_depth_obj = 3
    x_centre_norm_obj = 4
    y_centre_norm_obj = 5
    cross_size_ratio = 6
    word_cos_similarities = 7


class Image_Semantic_AwareClassifier_v2(nn.Module):
    def __init__(self):
        super(Image_Semantic_AwareClassifier_v2, self).__init__()
        _vgg16 = torchvision.models.vgg16(pretrained=True)

        # that feeds class_sub, class_obj binary masks, depth , RGB
        _first_feature_layer = [
            nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ]
        _features = list(_vgg16.features.children())[1:]

        self.features = nn.Sequential(*_first_feature_layer, *_features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.latent_extractor = nn.Sequential(
            *list(_vgg16.classifier.children())[:-1],
            nn.Linear(4096, 4096))

        # word vector의 정보량이 너무 방대해서 단순한 MLP로는 적절한 추론을 하지 못하는 것으로 생각된다.
        self.InS_classifier = nn.Sequential(
            nn.Linear(4146, 100),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        x[0] = self.features(x[0])
        x[0] = self.avgpool(x[0])
        x[0] = torch.flatten(x[0], 1)
        x = torch.cat((self.latent_extractor(x[0]), torch.flatten(x[1], 1)), dim=1)
        x = self.InS_classifier(x)
        return x

class Image_Semantic_AwareClassifier_v3(nn.Module):
    def __init__(self):
        super(Image_Semantic_AwareClassifier_v3, self).__init__()

        _eff_net = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)

        _first_feature_layer = [
            nn.Conv2d(6, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        ]

        _feature = list(_eff_net.features.children())[1:]
        self.features = nn.Sequential(*_first_feature_layer, *_feature)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)


        self.InS_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1586, 150),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(150, 2)
        )

    def forward(self, x):
        x[0] = self.features(x[0])
        x[0] = self.avgpool(x[0]).squeeze(dim=2).squeeze(dim=2)
        x = torch.cat((x[0], torch.flatten(x[1], 1)), dim=1)
        x = self.InS_classifier(x)
        return x

## code for deploy perpose

model_checkpoint_path = r'C:\model_temp\checkpoint\prometeus_v3_checkpoint_0.7229344844818115.pt'
model = Image_Semantic_AwareClassifier_v3()
model.load_state_dict(torch.load(model_checkpoint_path)['model_state_dict'])
model.to(device)
model.eval()


# world embedder 따로 빼야할 듯

# 여기는 학습용으로 만들고
# todo 차후에 deployment 관점에서 사용하기 위해서 파일별로 get하고 write하는 dataset을 만들어야 한다.
class CSV_Dataset(Dataset):
    def __init__(self, root_dir):
        """

        :param root_dir: csv file이 있는 directory
        """
        self.glove_vectors = gensim.downloader.load('glove-twitter-25')
        self.root_dir = root_dir
        self.csv_files = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
        csv_list = []
        for each_csv in self.csv_files:
            csv_list.append(pd.read_csv(each_csv))
        # huge dataframe from each image captioning data
        self.data = pd.concat(csv_list, ignore_index=True)

        # orignal file의 위치를 찾는 방법에 대해서 고안해놔야 한다.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = row[["avg_depth_sub", "x_centre_norm_sub", "y_centre_norm_sub",
                 "avg_depth_obj", "x_centre_norm_obj", "y_centre_norm_obj",
                 "cross_size_ratio"]].values.astype('float32')

        # 여기서 하는 것은 word2vec에 의한 두 단어의 거리에 대한 값을 append 해준다.
        # 작을 수록 유사도가 존재한다는 의미이다.

        words_class_sub = row["class_sub"].split(" ")
        class_sub_vec = []
        for word in words_class_sub:
            class_sub_vec.append(self.glove_vectors.get_vector(word))
        class_sub_vec = np.array(class_sub_vec).sum(axis=0)

        words_class_obj = row["class_obj"].split(" ")
        class_obj_vec = []
        for word in words_class_obj:
            class_obj_vec.append(self.glove_vectors.get_vector(word))
        class_obj_vec = np.array(class_obj_vec).sum(axis=0)

        x = np.append(x, (np.dot(class_sub_vec, class_obj_vec) / (
                np.linalg.norm(class_sub_vec) * np.linalg.norm(class_obj_vec))).astype('float32'))

        # 내가 하고 싶은 것은 data가 의미론적으로 의미가 있는지를 알고 싶은 것이다.
        y = row["semantic"].astype('long')
        y = torch.nn.functional.one_hot(torch.tensor(y).to(torch.int64), num_classes=2)

        return torch.from_numpy(x), y


class CSV_Image_Dataset(Dataset):
    def __init__(self, root_dir, data_sub_dir):
        """

        :param root_dir: csv file이 있는 directory
        예시) C:\model_temp\dataset
        data_sub_dir[0] = RGB
        data_sub_dir[1] = depth
        data_sub_dir[2] = csv
        """
        # self.glove_vectors = gensim.downloader.load('glove-twitter-25')
        self.root_dir = root_dir
        self.rgb_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[0], "*.jpg")))
        self.depth_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[1], "*.png")))
        self.csv_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[2], "*.csv")))
        csv_list = []
        for each_csv in self.csv_files:
            df = pd.read_csv(each_csv)

            image_path_raw = each_csv.split('\\')[-1].split('.')[0].split('_')
            each_rgb_image_path = os.path.join(root_dir, data_sub_dir[0],
                                               image_path_raw[0] + "_" + image_path_raw[1] + ".jpg")
            each_depth_image_path = os.path.join(root_dir, data_sub_dir[1],
                                                 image_path_raw[0] + "_" + image_path_raw[1] + ".png")
            df["rgb_file_path"] = each_rgb_image_path
            df["depth_file_path"] = each_depth_image_path
            csv_list.append(df)
        # huge dataframe from each image captioning data
        self.data = pd.concat(csv_list, ignore_index=True)

        # orignal file의 위치를 찾는 방법에 대해서 고안해놔야 한다.

    def __len__(self):
        return len(self.data)

    def _create_mask(self, row, H, W):
        sub_mask = np.zeros((H, W), dtype=np.float32)
        obj_mask = np.zeros((H, W), dtype=np.float32)
        sub_mask[int(row["ymin_sub"]):int(row["ymax_sub"]) + 1,
        int(row["xmin_sub"]):int(row["xmax_sub"]) + 1] = 1
        obj_mask[int(row["ymin_obj"]):int(row["ymax_obj"]) + 1,
        int(row["xmin_obj"]):int(row["xmax_obj"]) + 1] = 1

        return sub_mask, obj_mask

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        rgb = cv2.imread(row["rgb_file_path"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(row["depth_file_path"], cv2.IMREAD_ANYDEPTH)
        depth = np.array(depth).astype(np.float32) / np.iinfo(np.uint16).max

        H, W = rgb.shape[:2]

        sub_mask, obj_mask = self._create_mask(row, H, W)

        # manual transform이므로 나중에는 수정 필요
        rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        rgb = rgb_transform(rgb)

        # manual transform이므로 나중에는 수정 필요
        tensorize_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dps = tensorize_transform(
            np.concatenate((np.expand_dims(sub_mask, axis=-1), np.expand_dims(obj_mask, axis=-1),
                            np.expand_dims(depth, axis=-1)),
                           axis=-1)
        )

        x = torch.cat((rgb, dps), dim=0)  # (3, H, W) + (3, H, W) = (6, H, W)
        x = x.unsqueeze(0)  # (H, W, 6) -> (1, 6, H, W)

        # manual transform이므로 나중에는 수정 필요
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)

        # 내가 하고 싶은 것은 data가 의미론적으로 의미가 있는지를 알고 싶은 것이다.
        y = row["semantic"].astype('long')
        y = torch.nn.functional.one_hot(torch.tensor(y).to(torch.int64), num_classes=2).float()

        return x.squeeze(0), y


class CSV_Image_Semantic_Dataset(Dataset):
    def __init__(self, root_dir, data_sub_dir):
        """

        :param root_dir: csv file이 있는 directory
        예시) C:\model_temp\dataset
        data_sub_dir[0] = RGB
        data_sub_dir[1] = depth
        data_sub_dir[2] = csv
        """
        self.glove_vectors = gensim.downloader.load('glove-twitter-25')
        self.root_dir = root_dir
        self.rgb_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[0], "*.jpg")))
        self.depth_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[1], "*.png")))
        self.csv_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[2], "*.csv")))
        csv_list = []
        for each_csv in self.csv_files:
            df = pd.read_csv(each_csv)

            image_path_raw = each_csv.split('\\')[-1].split('.')[0].split('_')
            each_rgb_image_path = os.path.join(root_dir, data_sub_dir[0],
                                               image_path_raw[0] + "_" + image_path_raw[1] + ".jpg")
            each_depth_image_path = os.path.join(root_dir, data_sub_dir[1],
                                                 image_path_raw[0] + "_" + image_path_raw[1] + ".png")
            df["rgb_file_path"] = each_rgb_image_path
            df["depth_file_path"] = each_depth_image_path
            csv_list.append(df)
        # huge dataframe from each image captioning data
        self.data = pd.concat(csv_list, ignore_index=True)

        # orignal file의 위치를 찾는 방법에 대해서 고안해놔야 한다.

    def __len__(self):
        return len(self.data)

    def _create_mask(self, row, H, W):
        sub_mask = np.zeros((H, W), dtype=np.float32)
        obj_mask = np.zeros((H, W), dtype=np.float32)
        sub_mask[int(row["ymin_sub"]):int(row["ymax_sub"]) + 1,
        int(row["xmin_sub"]):int(row["xmax_sub"]) + 1] = 1
        obj_mask[int(row["ymin_obj"]):int(row["ymax_obj"]) + 1,
        int(row["xmin_obj"]):int(row["xmax_obj"]) + 1] = 1

        return sub_mask, obj_mask

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        rgb = cv2.imread(row["rgb_file_path"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(row["depth_file_path"], cv2.IMREAD_ANYDEPTH)
        depth = np.array(depth).astype(np.float32) / np.iinfo(np.uint16).max

        H, W = rgb.shape[:2]

        sub_mask, obj_mask = self._create_mask(row, H, W)

        # manual transform이므로 나중에는 수정 필요
        rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        rgb = rgb_transform(rgb)

        # manual transform이므로 나중에는 수정 필요
        tensorize_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dps = tensorize_transform(
            np.concatenate((np.expand_dims(sub_mask, axis=-1), np.expand_dims(obj_mask, axis=-1),
                            np.expand_dims(depth, axis=-1)),
                           axis=-1)
        )

        x = torch.cat((rgb, dps), dim=0)  # (3, H, W) + (3, H, W) = (6, H, W)
        x = x.unsqueeze(0)  # (H, W, 6) -> (1, 6, H, W)

        # manual transform이므로 나중에는 수정 필요
        x = torch.nn.functional.interpolate(x, size=(300, 300), mode='bilinear', align_corners=True)

        # 내가 하고 싶은 것은 data가 의미론적으로 의미가 있는지를 알고 싶은 것이다.
        y = row["semantic"].astype('long')
        y = torch.nn.functional.one_hot(torch.tensor(y).to(torch.int64), num_classes=2).float()

        # word semantic meaning
        words_class_sub = row["class_sub"].split(" ")
        class_sub_vec = []
        for each_word in words_class_sub:
            class_sub_vec.append(self.glove_vectors[each_word])
        class_sub_vec = np.array(class_sub_vec).mean(axis=0)

        words_class_obj = row["class_obj"].split(" ")
        class_obj_vec = []
        for each_word in words_class_obj:
            class_obj_vec.append(self.glove_vectors[each_word])
        class_obj_vec = np.array(class_obj_vec).mean(axis=0)

        vector = torch.tensor(np.vstack([class_sub_vec, class_obj_vec]), dtype=torch.float32)
        vector = nn.functional.normalize(vector, p=2, dim=1)

        return x.squeeze(0), vector, y


class CSV_Image_Semantic_Dataset_Deployer(Dataset):
    def __init__(self, root_dir, data_sub_dir):
        """

        :param root_dir: csv file이 있는 directory
        예시) C:\model_temp\dataset
        data_sub_dir[0] = RGB
        data_sub_dir[1] = depth
        data_sub_dir[2] = csv
        """
        self.glove_vectors = gensim.downloader.load('glove-twitter-25')
        self.root_dir = root_dir
        self.data_sub_dir = data_sub_dir
        self.rgb_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[0], "*.jpg")))
        self.depth_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[1], "*.png")))
        self.csv_files = sorted(glob.glob(os.path.join(root_dir, data_sub_dir[2], "*.csv")))

        os.makedirs(os.path.join(self.root_dir,self.data_sub_dir[2],
                               'results'), exist_ok=True)

        # model load
        # self.model = model
        """
        self.model = Image_Semantic_AwareClassifier_v2()
        self.model.load_state_dict(torch.load(model_checkpoint_path)['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        """

        test = 1
        # orignal file의 위치를 찾는 방법에 대해서 고안해놔야 한다.

    def __len__(self):
        return len(self.csv_files)

    def _create_mask(self, row, H, W):
        sub_mask = np.zeros((H, W), dtype=np.float32)
        obj_mask = np.zeros((H, W), dtype=np.float32)
        sub_mask[int(row["ymin_sub"]):int(row["ymax_sub"]) + 1,
        int(row["xmin_sub"]):int(row["xmax_sub"]) + 1] = 1
        obj_mask[int(row["ymin_obj"]):int(row["ymax_obj"]) + 1,
        int(row["xmin_obj"]):int(row["xmax_obj"]) + 1] = 1

        return sub_mask, obj_mask

    def __getitem__(self, idx):

        start_time = time.time()
        csv_file = self.csv_files[idx]  # 여기에서 glob으로 읽어옴
        # 중요
        df = pd.read_csv(csv_file)
        df["confidence"] = 0.0 # -2
        df['index'] = df.index # -1

        image_path_raw = csv_file.split('\\')[-1].split('.')[0].split('_')
        each_rgb_image_path = os.path.join(self.root_dir, self.data_sub_dir[0],
                                           image_path_raw[0] + "_" + image_path_raw[1] + ".jpg")
        each_depth_image_path = os.path.join(self.root_dir, self.data_sub_dir[1],
                                             image_path_raw[0] + "_" + image_path_raw[1] + ".png")
        # df["rgb_file_path"] = each_rgb_image_path
        # df["depth_file_path"] = each_depth_image_path

        # row = self.data.iloc[idx] # 마스크는 계속 합성 해야하기는 하는데 RGB, depth는 재활용 for a single csv

        rgb = cv2.imread(each_rgb_image_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(each_depth_image_path, cv2.IMREAD_ANYDEPTH)
        depth = np.array(depth).astype(np.float32) / np.iinfo(np.uint16).max

        H, W = rgb.shape[:2]

        # manual transform이므로 나중에는 수정 필요
        rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        rgb = rgb_transform(rgb)

        # manual transform이므로 나중에는 수정 필요
        tensorize_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 현재는 하나의 row에 대해서만 접근이 가능함
        tf_result = []
        conf_result = []
        for index, row in df.iterrows():
            # mask는 각각의 row에 대하여 다 만들어야 함.
            sub_mask, obj_mask = self._create_mask(row, H, W)

            # word semantic meaning
            words_class_sub = row["class_sub"].split(" ")
            class_sub_vec = []
            for each_word in words_class_sub:
                class_sub_vec.append(self.glove_vectors[each_word])
            class_sub_vec = np.array(class_sub_vec).mean(axis=0)

            words_class_obj = row["class_obj"].split(" ")
            class_obj_vec = []
            for each_word in words_class_obj:
                class_obj_vec.append(self.glove_vectors[each_word])
            class_obj_vec = np.array(class_obj_vec).mean(axis=0)

            vector = torch.tensor(np.vstack([class_sub_vec, class_obj_vec]), dtype=torch.float32)
            vector = nn.functional.normalize(vector, p=2, dim=1).unsqueeze(0)
            vector = vector.to(device)

            dps = tensorize_transform(
                np.concatenate((np.expand_dims(sub_mask, axis=-1), np.expand_dims(obj_mask, axis=-1),
                                np.expand_dims(depth, axis=-1)),
                               axis=-1)
            )

            x = torch.cat((rgb, dps), dim=0)  # (3, H, W) + (3, H, W) = (6, H, W)
            x = x.unsqueeze(0)  # (H, W, 6) -> (1, 6, H, W)

            # manual transform이므로 나중에는 수정 필요
            x = torch.nn.functional.interpolate(x, size=(300, 300), mode='bilinear', align_corners=True)
            x = x.to(device)
            
            conf = torch.nn.functional.softmax(model([x, vector]), dim=1)
            pred = (torch.argmax(conf, dim=1)).cpu().numpy().astype(bool)
            conf_score, _ = conf.max(dim=1)
            conf_result.extend(conf_score.detach().cpu().numpy())
            tf_result.extend(pred)

        df['semantic'] = tf_result
        df['confidence'] = conf_result

        columns = df.columns.tolist()

        new_col = [columns[-1]] + columns[0:8] + [columns[-2]] + columns[8:-2]
        df = df[new_col]

        df = df[df['confidence'] > 0.99]  # confidence 낮은 거 걸러내기

        df.to_csv(os.path.join(self.root_dir,self.data_sub_dir[2],
                               'results', csv_file.split('\\')[-1]), index=False)

        num_true_label = df['semantic'].sum()
        num_conf_label = (df['confidence'] > 0.99).sum()
        # T, F
        num_rows = len(df)
        tot_time = time.time() - start_time
        return csv_file, num_true_label, num_rows, tot_time, num_conf_label


# model의 power가 너무 약한 것으로 생각됨 image feature, depth feature, object mask feature를 concatenate해서 binary classifier를 만들어야 겠다
class SimpleClassifier_v1(nn.Module):
    def __init__(self):
        super(SimpleClassifier_v1, self).__init__()
        # self.scalars = nn.Parameter(torch.randn(4))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tensor_10 = torch.tensor(10, dtype=torch.float32)
        # batch 인거 까먹음
        xy_diff_square = torch.pow(
            x[:, tensor_X_str.x_centre_norm_sub.value] - x[:, tensor_X_str.x_centre_norm_obj.value], 2) \
                         + torch.pow(
            x[:, tensor_X_str.y_centre_norm_sub.value] - x[:, tensor_X_str.y_centre_norm_obj.value], 2)
        depth_diff = x[:, tensor_X_str.avg_depth_sub.value] - x[:, tensor_X_str.avg_depth_obj.value]

        # 단어간의 semantic distance는 작을 수록 좋다
        # xy_diff와 log_dep_diff는 작을 수록 좋다
        # 그런데 cross_size_ratio는 최대가 1이라서 작으면 안 좋다.

        p1 = (torch.log(x[:, tensor_X_str.cross_size_ratio.value]) / torch.log(tensor_10)) * xy_diff_square
        p2 = (x[:, tensor_X_str.word_cos_similarities.value]) * xy_diff_square
        p3 = (torch.log(x[:, tensor_X_str.cross_size_ratio.value]) / torch.log(tensor_10)) * depth_diff
        p4 = (x[:, tensor_X_str.word_cos_similarities.value]) * depth_diff

        p = torch.stack([p1, p2, p3, p4], dim=1)

        p = self.relu(self.fc1(p))
        p = self.relu(self.fc2(p))
        p = self.fc3(p)

        # regression해서 최종 결과가 1에 근접하면 semantic이 유의미 하므로 label을 주어야 하는 것으로
        # 판단하면 된다.
        return p


class ImageAwareClassifier(nn.Module):
    def __init__(self):
        super(ImageAwareClassifier, self).__init__()
        _vgg16 = torchvision.models.vgg16(pretrained=True)

        # that feeds class_sub, class_obj binary masks, depth , RGB
        _first_feature_layer = [
            nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ]
        _features = list(_vgg16.features.children())[1:]

        self.features = nn.Sequential(*_first_feature_layer, *_features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        _classifier = list(_vgg16.classifier.children())[:-1]
        _last_classifier_layer = [
            nn.Linear(4096, 2)
        ]
        self.classifier = nn.Sequential(*_classifier, *_last_classifier_layer)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


# Image 공간 semantic과 word semantic aware classifier
# 다시짜기
class Image_Semantic_AwareClassifier(nn.Module):
    def __init__(self):
        super(Image_Semantic_AwareClassifier, self).__init__()
        _vgg16 = torchvision.models.vgg16(pretrained=True)

        # that feeds class_sub, class_obj binary masks, depth , RGB
        _first_feature_layer = [
            nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ]
        _features = list(_vgg16.features.children())[1:]

        self.features = nn.Sequential(*_first_feature_layer, *_features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.latent_extractor = nn.Sequential(
            *list(_vgg16.classifier.children())[:-1],
            nn.Linear(4096, 100))

        self.InS_classifier = nn.Sequential(
            nn.Linear(150, 150),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(150, 100),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x[0] = self.features(x[0])
        x[0] = self.avgpool(x[0])
        x[0] = torch.flatten(x[0], 1)
        x = torch.cat((self.latent_extractor(x[0]), torch.flatten(x[1], 1)), dim=1)
        x = self.InS_classifier(x)
        return x



# space and word semantic aware under image scene classifier


# trainer block
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (img, vec, y) in enumerate(dataloader):
        # Compute prediction error
        img, vec, y = img.to(device), vec.to(device), y.to(device)

        pred = model([img, vec])
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()  # error 발생
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(img)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# test block 이후에 deploy block을 만들어서 실제 활용을 해야한다.
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct, true_count, tot_true = 0, 0, 0, 0
    with torch.no_grad():
        for img, vec, y in dataloader:
            img, vec, y = img.to(device), vec.to(device), y.to(device)

            pred = model([img, vec])
            test_loss += loss_fn(pred, y.float()).item()
            correct += (pred.argmax(axis=1) == y.argmax(axis=1)).type(torch.float).sum().item()
            true_count += torch.count_nonzero((pred.argmax(axis=1) == y.argmax(axis=1)) * (pred.argmax(axis=1) == 1))
            # tot_true += (pred.argmax(axis=1) == 1)
    test_loss /= size
    correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f},"
          f"true_count: {true_count / (size * 0.2)} \n")

    return (100 * correct), true_count / (size * 0.2)


if __name__ == "__main__":
    dict_path = r'Z:\results\nogada\prometeus_v4'
    random_seed = 777
    dataset_path = r'C:\model_temp\dataset'
    train_size = 0.8
    lr = 1e-5
    train_model = False

    if train_model:
        torch.manual_seed(random_seed)
        model = Image_Semantic_AwareClassifier_v3()  # todo 모델이 가정이 너무 이상할 수도 있으니까 vanilla version도 짠다.
        model.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.2, 0.8]).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        full_dataset = CSV_Image_Semantic_Dataset(root_dir=dataset_path, data_sub_dir=['input_image_raw_trim-use',
                                                                                       'input_image_depth_trial2-use',
                                                                                       'prometus-use-2'])
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * train_size),
                                                                                   len(full_dataset) - int(
                                                                                       len(full_dataset) * train_size)])

        train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=1, pin_memory=False, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=1, pin_memory=False, shuffle=False)

        epochs = 100
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            acc, rtpt = test_loop(test_dataloader, model, loss_fn)
            # model checkpointer
            checkpoint = {
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            rtpt = rtpt.detach().cpu().item()
            os.makedirs(os.path.join(dict_path, str(t) + '_' + str(rtpt) + '_' + str(acc)), exist_ok=True)
            torch.save(checkpoint, os.path.join(os.path.join(dict_path, str(t) + '_' + str(rtpt) + '_' + str(acc)),
                                                f'checkpoint_{rtpt}.pt'))

        print("Done!")

    else:
        deploy = CSV_Image_Semantic_Dataset_Deployer(root_dir=dataset_path,
                                                     data_sub_dir=['input_image_raw_trim-use',
                                                                   'input_image_depth_trial2-use',
                                                                   'prometeus-deploy-trial3'],
                                                     )
        deploy_loader = DataLoader(deploy, batch_size=1, num_workers=1, pin_memory=False, shuffle=False)

        for batch in deploy_loader:
            start = time.time()
            print('Processed CSV file : ', batch[0], 'time counsumed : ' , batch[3])
            print('Processed label : ', batch[2], 'True labels : ', batch[1], 'Conf labels : ', batch[4])