import cv2
import dlib
import numpy as np
import pickle
import pandas as pd
import torch
import os
import sys
import glob
import h5py
import random
from tqdm import tqdm
from torchvision.transforms import transforms
import torch.nn as nn
import argparse

from external.exp_model.resmasking import ResMasking
from external.au_model.MEFL import MEFARG

import io
import subprocess
import wave
import opensmile
import scipy.io.wavfile as wav
from python_speech_features import logfbank

from external.pose_model.pose_feature import create_model_and_load_checkpoint
from external.pose_model.pose_feature import ToFloatTensorInZeroOne
from external.pose_model.pose_feature import Resize
from torchvision import transforms
import torch
from decord import VideoReader
import torch.nn.functional as F

import h5py

sys.path.append('..')
from src.util import VAIndex, AUIndex, ExpIndex



class VideoToNumpy:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./external/haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        
    def crop_square(self, img, x, y, w, h):
        '''
        将人脸检测框裁剪为正方形
        '''
        length = max(w, h)
        center_x = x + w // 2
        center_y = y + h // 2
        x1 = max(0, center_x - length // 2)
        y1 = max(0, center_y - length // 2)
        x2 = min(img.shape[1], center_x + length // 2)
        y2 = min(img.shape[0], center_y + length // 2)

        # print(x1, y1, x2, y2)
        return img[y1:y2, x1:x2, :]

    def detect_face(self, img):
        # 使用 OpenCV 进行人脸检测
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

        # 如果 OpenCV 检测到人脸，返回人脸位置信息
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            return (x, y, w, h)

        # 如果 OpenCV 没有检测到人脸，使用 dlib 进行检测
        dets = self.detector(img, 1)

        # 如果 dlib 检测到人脸，返回人脸位置信息
        if len(dets) > 0:
            d = dets[0]
            return (d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top())

        # 如果两种方法都没有检测到人脸，返回 False
        return False
    
    def to_numpy(self, video_path):
        # 加载视频文件
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file")
            exit()

        # 定义numpy数组，用于保存裁剪出的人脸
        faces = []
        mask = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ret = self.detect_face(gray)
            if ret:
                x, y, w, h = ret
                # 裁剪出人脸并调整为正方形
                face = self.crop_square(frame, x, y, w, h)

                # 对人脸进行缩放，调整为128x128大小，且像素值设为0至1之间
                face = cv2.resize(face, (128, 128))
                face = face / 255.

                # 将人脸添加到numpy数组下一个维度中
                faces.append(face)
                mask.append(1)
            else:
                face = np.zeros((128, 128, 3), dtype=np.uint8)
                faces.append(face)
                mask.append(0)

        # 将numpy数组转换为标准格式(N,C,H,W)并保存为pkl文件
        faces = np.array(faces).transpose((0, 3, 1, 2))
        mask = np.array(mask)

        return faces, mask
    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize
        ])


class ExpressionModel(nn.Module):
    def __init__(self):
        super(ExpressionModel, self).__init__()
        self.expression_model = ResMasking()
        state = torch.load("./external/download/pretrained_ckpt", map_location="cpu")
        self.expression_model.load_state_dict(state["net"], strict=False)
        self.expression_model.eval()
        self.expression_model.cuda(0)

    def expression_extract(self, face_image):

        face_image = transform(face_image)
        face_image = face_image.cuda(0)
        face_image = torch.unsqueeze(face_image, dim=0)
        output = self.expression_model(face_image).squeeze(0).cpu().detach().numpy()

        return output


class AuModel(nn.Module):
    def __init__(self):
        super(AuModel, self).__init__()
        self.au_model = MEFARG(num_classes=12)
        self.au_model.eval()
        self.au_model.cuda(0)

    def au_extract(self, face_image):

        face_image = transform(face_image)
        face_image = face_image.cuda(0)
        face_image = torch.unsqueeze(face_image, dim=0)
        output = self.au_model(face_image).squeeze(0).cpu().detach().numpy()

        return output


class VAModel(nn.Module):
    def __init__(self):
        super(VAModel, self).__init__()
        self.va_model = torch.load("./external/download/enet_b2_8_best.pt")
        self.va_model.classifier = torch.nn.Identity()
        self.va_model.eval()
        self.va_model.cuda(0)
        
    def va_extract(self, face_image):
        
        face_image = transform(face_image)
        face_image = face_image.cuda(0)
        face_image = torch.unsqueeze(face_image, dim=0)
        output = self.va_model(face_image).squeeze(0).cpu().detach().numpy()
        
        return output


class FaceFeatureExtractor:
    def __init__(self):
        self.expression_model = ExpressionModel()
        self.au_model = AuModel()
        self.va_model = VAModel()
        self.image_size = (224, 224)
        
    def _ensure_color(self, image):
        if len(image.shape) == 2:
            return np.dstack([image] * 3)
        elif image.shape[2] == 1:
            return np.dstack([image] * 3)
        return image


    def _ensure_gray(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            pass
        return image

    def extract_features(self, cropped_faces, is_face):

        faces = cropped_faces
        mask = is_face
        expression_features = []
        au_features = []
        va_features = []
        for i in range(faces.shape[0]):
            face = np.transpose(faces[i], (1, 2, 0))
            face = (face * 255).astype(np.uint8)
            assert isinstance(face, np.ndarray)
            face = self._ensure_gray(face)
            face = self._ensure_color(face)
            face = cv2.resize(face, self.image_size)

            if mask[i] == 0:
                expression_feature = np.zeros(512)
                expression_features.append(expression_feature)
                au_feature = np.zeros(25088)
                au_features.append(au_feature)
                va_feature = np.zeros(1408)
                va_features.append(va_feature)
            else:
                expression_feature = self.expression_model.expression_extract(face)
                expression_features.append(expression_feature)
                au_feature = self.au_model.au_extract(face).flatten()
                au_features.append(au_feature)
                va_feature = self.va_model.va_extract(face)
                va_features.append(va_feature)

        return np.array(au_features), np.array(expression_features), np.array(va_features)



class AudioFeatureExtractor:
    def __init__(self):
        self.smilefunc = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.smilelld = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def extract_audio_features(self, path):
        wav_data = self.convert_to_wav(path)
        with wave.open(wav_data, 'rb') as wav_file:
            rate = wav_file.getframerate()
            sig = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

        MFCC = logfbank(sig, rate)[:750]
        GeMapfunc = self.smilefunc.process_file(path).to_numpy().flatten()
        GeMaplld = self.smilelld.process_file(path).to_numpy()
        return MFCC, GeMapfunc, GeMaplld

    def convert_to_wav(self, path):
        command = ['ffmpeg', '-i', path, '-vn', '-ar', '44100', '-ac', '2', '-ab', '192' ,'-f', 'wav','-loglevel', 'quiet', '-']
        wav_data = subprocess.check_output(command)
        return io.BytesIO(wav_data)


class PoseFeatureExtractor:
    def __init__(self):
        self.pose_model = create_model_and_load_checkpoint()
        self.transform = transforms.Compose([ToFloatTensorInZeroOne(), Resize((224, 224))])
        self.frame_interval = 5
        
    def _get_start_idx_range(self, num_frames):
        return range(0, num_frames - self.frame_interval + 1, self.frame_interval)

    def pose_extract(self, path):
        video = VideoReader(path, num_threads=1)
        feature_list = []
        
        for start_idx in self._get_start_idx_range(len(video)):
            data = video.get_batch(np.arange(start_idx, start_idx + self.frame_interval)).asnumpy()
            frame = torch.from_numpy(data)
            frame = self.transform(frame)
            input_data = frame.unsqueeze(0).cuda(0)

            with torch.no_grad():
                feature = self.pose_model.forward_features(input_data)
                feature_list.append(feature.cpu().numpy())
            
        output = np.vstack(feature_list)
            
        return output

def trans_video_path_to_csv(video_path):
    csv_path = video_path.replace('.mp4', '.csv').replace('Video_files', 'Emotion')
    return csv_path

def get_video_path(test_csv_path):
    df = pd.read_csv(test_csv_path, header=None)
    video_paths = df.values.flatten().tolist()
    return video_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data preprocessing")
    parser.add_argument('-d', "--data_type", type=str, default='test', help='train, val, or test')
    args = parser.parse_args()
    
    video_to_numpy = VideoToNumpy()
    face_extractor = FaceFeatureExtractor()
    audio_extractor = AudioFeatureExtractor()
    pose_extractor = PoseFeatureExtractor()
    expindex = ExpIndex('../data/Index/kmeans.pkl')
    va_index = VAIndex('../data/Index/VA_index.csv')
    au_index = AUIndex('../data/Index/AU_index.csv')
    print('initialized')

    """
    assume these official files in ../data:
    data
    ├── train_idx.csv
    ├── val_idx.csv
    ├── test_idx.csv
    ├── neighbour_emotion_train.npy
    ├── neighbour_emotion_val.npy
    ├── Index
        ├── AU_index.csv
        ├── kmeans.pkl
        ├── VA_index.csv
        
    and class VAIndex, AUIndex, and ExpIndex are defined in ./src/util.py.
    
    The structure of mp4 and csv files see README.md. 
    """

    # when it is train set, generate shuffled train_shuffled.h5 and train_appro_shuffled.npy to ../data
    if args.data_type == 'train':
        csv_path = '../data/train_idx.csv'
        h5_path = '../data/train_shuffled.h5'
        in_npy_path = '../data/neighbour_emotion_train.npy'
        out_npy_path = '../data/train_appro_shuffled.npy'
        video_paths = get_video_path(csv_path)
        n_sample = len(video_paths)
        
        in_npy = np.load(in_npy_path)

        shuffled_idx = list(range(n_sample))
        random.shuffle(shuffled_idx)
        
        out_npy = in_npy[shuffled_idx]
        out_npy[:] = out_npy[:, shuffled_idx]
        
        np.save(out_npy_path, out_npy)
        
        with h5py.File(h5_path, 'w') as h5out:
            str_type = h5py.special_dtype(vlen=str)
            h5out.create_dataset("s_name", shape=(n_sample, ), dtype=str_type)
            h5out.create_dataset("s_exp", shape=(n_sample, 750, 512), dtype='f4')
            h5out.create_dataset("s_AU", shape=(n_sample, 750, 25088), dtype='f4')
            h5out.create_dataset("s_VA", shape=(n_sample, 750, 1408), dtype='f4')
            h5out.create_dataset("s_pose", shape=(n_sample, 150, 1408), dtype='f4')
            h5out.create_dataset("s_MFCC", shape=(n_sample, 750, 26), dtype='f4')
            h5out.create_dataset("s_GeMapfunc", shape=(n_sample, 6373), dtype='f4')
            h5out.create_dataset("s_GeMaplld", shape=(n_sample, 194805), dtype='f4')
            h5out.create_dataset("is_face", shape=(n_sample, 750), dtype='i4')
            
            h5out.create_dataset("l_tokens", shape=(n_sample, 3, 750), dtype='i4')

            for idx, official_idx in tqdm(enumerate(shuffled_idx)):
                video_path = video_paths[official_idx]
                video_path = '../data/train/Video_files/' + video_path + '.mp4'
                cropped_faces, is_face = video_to_numpy.to_numpy(video_path)
                s_au, s_exp, s_va = face_extractor.extract_features(cropped_faces, is_face)
                s_MFCC, s_GeMapfunc, s_GeMaplld = audio_extractor.extract_audio_features(video_path)
                s_pose = pose_extractor.pose_extract(video_path)
                sample_name = video_path
                sample_idx = official_idx
                
                h5out["s_name"][idx] = sample_name
                h5out["s_exp"][idx] = s_exp
                h5out["s_AU"][idx] = s_au
                h5out["s_VA"][idx] = s_va
                h5out["s_pose"][idx] = s_pose
                h5out["s_MFCC"][idx] = s_MFCC
                h5out["s_GeMapfunc"][idx] = s_GeMapfunc
                h5out["s_GeMaplld"][idx] = s_GeMaplld.reshape(-1)[:194805]
                h5out["is_face"][idx] = is_face
                
                if official_idx < len(video_paths)/2:
                    oppo_idx = official_idx + len(video_paths)//2
                else:
                    oppo_idx = official_idx - len(video_paths)//2
                    
                csv_path = trans_video_path_to_csv(video_path[oppo_idx])
                df = pd.read_csv(csv_path)
                
                l_AU = df.iloc[:,:15].values
                l_VA = df.iloc[:,15:17].values
                l_exp = df.iloc[:,17:].values
                
                token_au_list = []
                token_va_list = []
                token_exp_list = []
                
                for combinations in l_AU:
                    token_au = au_index.get_index(tuple(combinations))
                    token_au_list.append(token_au)
                token_au = np.array(token_au_list)
                
                for combinations in l_VA:
                    token_va = va_index.get_index(tuple(combinations))
                    token_va_list.append(token_va)
                token_va = np.array(token_va_list)
                
                l_exp = l_exp.astype('float32')
                token_exp = expindex.get_index(l_exp)
                
                h5out["l_tokens"][idx, 0] = token_au
                h5out["l_tokens"][idx, 1] = token_exp
                h5out["l_tokens"][idx, 2] = token_va
                
        
    # when it is val set, generate val_sequential.h5 and val_s_gt.npy and val_l_gt.npy to ../data
    elif args.data_type == 'val':
        csv_path = '../data/val_idx.csv'
        h5_path = '../data/val_sequential.h5'
        video_paths = get_video_path(csv_path)
        n_sample = len(video_paths)
        
        with h5py.File(h5_path, 'w') as h5out:
            str_type = h5py.special_dtype(vlen=str)
            h5out.create_dataset("s_name", shape=(n_sample, ), dtype=str_type)
            h5out.create_dataset("s_exp", shape=(n_sample, 750, 512), dtype='f4')
            h5out.create_dataset("s_AU", shape=(n_sample, 750, 25088), dtype='f4')
            h5out.create_dataset("s_VA", shape=(n_sample, 750, 1408), dtype='f4')
            h5out.create_dataset("s_pose", shape=(n_sample, 150, 1408), dtype='f4')
            h5out.create_dataset("s_MFCC", shape=(n_sample, 750, 26), dtype='f4')
            h5out.create_dataset("s_GeMapfunc", shape=(n_sample, 6373), dtype='f4')
            h5out.create_dataset("s_GeMaplld", shape=(n_sample, 194805), dtype='f4')
            h5out.create_dataset("is_face", shape=(n_sample, 750), dtype='i4')

            for idx, video_path in tqdm(enumerate(video_paths)):
                video_path = '../data/val/Video_files/' + video_path + '.mp4'
                cropped_faces, is_face = video_to_numpy.to_numpy(video_path)
                s_au, s_exp, s_va = face_extractor.extract_features(cropped_faces, is_face)
                s_MFCC, s_GeMapfunc, s_GeMaplld = audio_extractor.extract_audio_features(video_path)
                s_pose = pose_extractor.pose_extract(video_path)
                sample_name = video_path
                sample_idx = idx

                h5out["s_name"][idx] = sample_name
                h5out["s_exp"][idx] = s_exp
                h5out["s_AU"][idx] = s_au
                h5out["s_VA"][idx] = s_va
                h5out["s_pose"][idx] = s_pose
                h5out["s_MFCC"][idx] = s_MFCC
                h5out["s_GeMapfunc"][idx] = s_GeMapfunc
                h5out["s_GeMaplld"][idx] = s_GeMaplld.reshape(-1)[:194805]
                h5out["is_face"][idx] = is_face
                
        with h5py.File(h5_path, 'r') as f:

            s_name_dataset = f['s_name']
            s_name_list = list(s_name_dataset)
            s_gt = np.zeros((len(s_name_list), 750, 25))
            l_gt = np.zeros((len(s_name_list), 750, 25))

            for idx, s_name in enumerate(s_name_list):
                csv_name = trans_video_path_to_csv(s_name.decode('utf-8'))
                df = pd.read_csv(csv_name)
                s_gt[idx] = df.values
                if idx < len(s_name_list)/2:
                    l_gt[idx+len(s_name_list)//2]=df.values
                else:
                    l_gt[idx-len(s_name_list)//2]=df.values

        np.save('../data/val_s_gt.npy', s_gt)
        np.save('../data/val_l_gt.npy', l_gt)
        
        
        
    # when it is test set, only generate test_sequential.h5 to ../data
    else:
        csv_path = '../data/test_idx.csv'
        h5_path = '../data/test_sequential.h5'
        video_paths = get_video_path(csv_path)
        n_sample = len(video_paths)

        with h5py.File(h5_path, 'w') as h5out:
            str_type = h5py.special_dtype(vlen=str)
            h5out.create_dataset("s_name", shape=(n_sample, ), dtype=str_type)
            h5out.create_dataset("s_exp", shape=(n_sample, 750, 512), dtype='f4')
            h5out.create_dataset("s_AU", shape=(n_sample, 750, 25088), dtype='f4')
            h5out.create_dataset("s_VA", shape=(n_sample, 750, 1408), dtype='f4')
            h5out.create_dataset("s_pose", shape=(n_sample, 150, 1408), dtype='f4')
            h5out.create_dataset("s_MFCC", shape=(n_sample, 750, 26), dtype='f4')
            h5out.create_dataset("s_GeMapfunc", shape=(n_sample, 6373), dtype='f4')
            h5out.create_dataset("s_GeMaplld", shape=(n_sample, 194805), dtype='f4')
            h5out.create_dataset("is_face", shape=(n_sample, 750), dtype='i4')

            for idx, video_path in tqdm(enumerate(video_paths)):
                video_path = '../data/test/Video_files/' + video_path + '.mp4'
                cropped_faces, is_face = video_to_numpy.to_numpy(video_path)
                s_au, s_exp, s_va = face_extractor.extract_features(cropped_faces, is_face)
                s_MFCC, s_GeMapfunc, s_GeMaplld = audio_extractor.extract_audio_features(video_path)
                s_pose = pose_extractor.pose_extract(video_path)
                sample_name = video_path
                sample_idx = idx

                h5out["s_name"][idx] = sample_name
                h5out["s_exp"][idx] = s_exp
                h5out["s_AU"][idx] = s_au
                h5out["s_VA"][idx] = s_va
                h5out["s_pose"][idx] = s_pose
                h5out["s_MFCC"][idx] = s_MFCC
                h5out["s_GeMapfunc"][idx] = s_GeMapfunc
                h5out["s_GeMaplld"][idx] = s_GeMaplld.reshape(-1)[:194805]
                h5out["is_face"][idx] = is_face


