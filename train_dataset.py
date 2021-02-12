#coding:utf-8

# 路径置顶
import sys
import os

sys.path.append(os.getcwd())
# 导入包
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import dlib
import cv2


# 训练数据
class TrainDataset(Dataset):
    def __init__(self, face_dir, csv_name, num_triplets, predicter_path, img_size,
                 training_triplets_path=None, transform=None):
        # 初始化
        # 读取csv文件
        self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': int})
        # 训练图片的路径
        self.face_dir = face_dir
        # 用于生成三元对的数量
        self.num_triplets = num_triplets
        self.transform = transform

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predicter_path)
        self.img_size = img_size

        # 如果有配好的三元组数据就直接用，要不就生成一下
        if os.path.exists(training_triplets_path):
            print("\nload {} triplets...".format(num_triplets))
            self.training_triplets = np.load(training_triplets_path)
            print('{} triplets loaded!'.format(num_triplets))
        else:
         
            self.training_triplets = self.generate_triplets(self, self.df, self.num_triplets,training_triplets_path)

    # 静态方法
    @staticmethod
    def generate_triplets(self, df, num_triplets,training_triplets_path):
        '''
        生成三元组数据
        每个三元组包括锚样本、正样本、负样本，其中锚样本和正样本属于同一个类（人），负样本属于另一个类（人）

        输入：原始数据的列表信息和所要的对数
        '''
        print("\nGenerating {} triplets:".format(num_triplets))

        def make_dictionary_for_face_class(df):
            # df包括：id,name,class三列，对应的图片名称、人名、类别
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                # 往字典中的这个类里加图片名称
                face_classes[label].append(df.iloc[idx, 0])
            # face_classes = {'class0': [class0_id0, class0_id1, ...], 'class1': [class1_id0, ...], ...}
            return face_classes

        triplets = []
        # 得到类别
        classes = df['class'].unique()
        print("Generating face_classes...")
        face_classes = make_dictionary_for_face_class(df)
        print("Generating npy file...")
        # 做进度条用的
        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            # FIXME 类和人名其实是一一对应的，只保留一个说不定就行
            # 随机选两个类当做正类负类
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            # 如果选出来的正类里的图片数少于2就重新选一个正类
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            # 如果选出来的正负类相等就重新选个负类
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            # FIXME 这个name其实到后来没啥用啊。。
            # 选出这个类对应的人名
            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            # 如果正类里的图片数等于2就分别取出来当做锚样本、正样本（取index）
            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            # 绕不然就随机取俩
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                # 取出来的俩一样的话，就重新拿个正样本
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            # 随机取出一个负样本
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [
                    face_classes[pos_class][ianc],  # 锚样本图片名称
                    face_classes[pos_class][ipos],  # 正样本图片名称
                    face_classes[neg_class][ineg],  # 负样本图片名称
                    pos_class,  # 正类
                    neg_class,  # 负类
                    pos_name,  # 正类人名
                    neg_name  # 负类人名
                ]
            )

        print("Saving training triplets list in datasets/ directory ...")
        np.save(training_triplets_path, triplets)
        print("Training triplets' list Saved!\n")

        # 这里返回是因为第一次生成的时候就直接用返回的而不直接读文件了
        return triplets

    def preprocess(self, image_path, detector, predictor, img_size):
        image = dlib.load_rgb_image(image_path)
        face_img, mask_img, TF = None, None, 1
        # 人脸对齐、切图
        dets = detector(image, 1)
        if len(dets) == 1:
            # 如果检测到人脸
            faces = dlib.full_object_detections()
            faces.append(predictor(image, dets[0]))
            # 裁剪出人脸，get_face_chip，人脸配准
            images = dlib.get_face_chips(image, faces, size=img_size)
            image = np.array(images[0]).astype(np.uint8)
            face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 原本下面是增加mask的代码，现在移到image_processing中去直接生成带mask的图片
        return face_img

        
    def __len__(self):
        return len(self.training_triplets)

    def add_extension(self, path):
        # 文件格式比较迷，可能有这两种情况
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __getitem__(self, idx):
        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.add_extension(os.path.join(self.face_dir, str(pos_name), str(anc_id)))
        pos_img = self.add_extension(os.path.join(self.face_dir, str(pos_name), str(pos_id)))
        neg_img = self.add_extension(os.path.join(self.face_dir, str(neg_name), str(neg_id)))

        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')

        pos_class = torch.from_numpy(np.array([pos_class]).astype('int64'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('int64'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

if __name__=="__main__":
    dataset=TrainDataset(
          face_dir="Datasets/vggface2_train_face_mask",
          csv_name='Datasets/vggface2_train_face_mask.csv',
          num_triplets=1000,
          training_triplets_path='Datasets/training_triplets_1000_mask.npy',
          transform=None,
          predicter_path='shape_predictor_68_face_landmarks.dat',
          img_size=256)

    sample=dataset[0]
