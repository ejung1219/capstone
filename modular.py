# 필요 라이브러리 정의
import os

import torchvision.models as models
import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img


import collections
from sklearn.cluster import KMeans


def GetDLModel():
    # 모델 불러와서 일부 키 값 resnet50 model에 맞게 수정
    # 아래 torch.load에서 모델을 저장한 경로만 바뀌면 됩니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    checkpoint2 = torch.load(
        '/content/drive/MyDrive/resnet50_market_xent.pth.tar', map_location=device, encoding='latin1')
    checkpoint2['state_dict']['fc.weight'] = checkpoint2['state_dict'].pop(
        'classifier.weight')
    checkpoint2['state_dict']['fc.bias'] = checkpoint2['state_dict'].pop(
        'classifier.bias')

    # 기존의 resnet50을 market1501을 학습 시킨 모델과 출력층이 동일하게 구성
    res2 = models.resnet50()
    x = res2.fc.weight
    x = torch.narrow(x, 0, 0, 751)
    y = res2.fc.bias
    y = torch.narrow(y, 0, 0, 751)

    res2.fc.weight = nn.Parameter(x)
    res2.fc.bias = nn.Parameter(y)
    res2.fc.out_features = 751

    # 모델 덮어 씌우기
    res2.load_state_dict(checkpoint2['state_dict'])
    res2.to(device)  # 가중치를 GPU 계산과 CPU 계산 중 하나로 통일

    return res2


def getImages():
    # 이미지 폴더에서 이미지 셋 리스트에 저장
    '''
    아래에서 target image는 프론트에서 받아오고, 
    나머지 이미지는 크로핑 된 결과로 넘어 옵니다.

    백에서 경로를 어떻게 사용하는 지 몰라서 일단 제 코랩에서 사용하는 방식으로 정리했습니다.
    '''
    path = '/content/drive/MyDrive/ZEPETO'
    os.chdir(path)

    images = []

    # 타겟 이미지 리스트에 삽입
    target_name = '/content/drive/MyDrive/target.png'
    images.append('/content/drive/MyDrive/target.png')

    # 이미지 파일 리스트에 삽입
    with os.scandir(path) as files:
        for file in files:
            if file.name.endswith('.png'):
                images.append(file.name)
    return images

# feature extractor class 설계


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.global_avgpool = model.avgpool
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        return

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        v = self.pool(v)
        v = self.pool(v)
        v = self.pool(v)
        v = self.pool(v)
        return v

# 이미지 특징 추출 함수


def extracting_feature(model, images, target_name, transform):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    features_ = []
    extract_feature = FeatureExtractor(model)
    for i in range(len(images)):
        if i == len(images):
            break
        if images[i] == target_name:
            path = target_name
        else:
            path = os.path.join('/content/drive/MyDrive/ZEPETO', images[i])
        img = load_img(path, target_size=(224, 224))
        img = np.array(img)
        img = transform(img)
        img = img.reshape(1, 3, 224, 224)
        img = img.to(device)
        with torch.no_grad():
            feature = extract_feature(img)
        features_.append(feature.cpu().detach().numpy().reshape(-1))
    return features_


def clusters(feat, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=40)
    kmeans.fit(feat)
    result = kmeans.labels_
    labels = np.unique(result)
    return result, labels

# 이미지 transform 구성


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def play():
    target_name = '/content/drive/MyDrive/target.png'
    MyResnet = GetDLModel()
    zepeto = getImages()
    trans = get_transform()
    features = extracting_feature(MyResnet, zepeto, target_name, trans)
    features = np.array(features)
    (res, labels) = clusters(features, 10)

    groups = {}
    for file, cluster in zip(zepeto, res):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    count = 0
    for i in range(len(labels)):
        if target_name in groups[i]:
            continue
        count += len(groups[i])
    print(count)


play()
