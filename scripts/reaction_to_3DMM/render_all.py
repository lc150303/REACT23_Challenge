import os
import sys
import numpy as np
import torch
from torchvision import transforms
from skimage.io import imsave
import skvideo.io
from pathlib import Path
from tqdm import auto
from tqdm import tqdm
import argparse
import cv2
import pickle 

from PIL import Image

from utils import torch_img_to_np, _fix_image, torch_img_to_np2
from external.FaceVerse import get_faceverse
from external.PIRender import FaceGenerator

save_dir = os.path.dirname(os.path.abspath(__file__))
reference_dir = os.path.join(save_dir, 'listener_frame')
output_path=os.path.join(save_dir, "visual_output")   #the Visualization will output in this folder

def obtain_seq_index(index, num_frames, semantic_radius = 13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def transform_semantic(semantic):
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim = 0)
    return semantic.transpose(1,2)



class Render(object):
    """Computes and stores the average and current value"""

    def __init__(self, device = 'cuda:2'):
        self.faceverse, _ = get_faceverse(device=device, img_size=224)
        self.faceverse.init_coeff_tensors()
        self.id_tensor = torch.from_numpy(np.load('external/FaceVerse/reference_full.npy')).float().view(1,-1)[:,:150]
        self.pi_render = FaceGenerator().to(device)
        self.pi_render.eval()
        checkpoint = torch.load('external/PIRender/cur_model_fold.pth')
        self.pi_render.load_state_dict(checkpoint['state_dict'])

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1).to(device)
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1).to(device)

        self._reverse_transform_3dmm = transforms.Lambda(lambda e: e  + self.mean_face)
        #self._reverse_transform_3dmm = transforms.Lambda(lambda e: torch.add(e,self.mean_face))

#    def rendering(self, path, ind, listener_vectors, speaker_video_clip, listener_reference):
    def rendering(self, path, ind, listener_vectors, listener_reference):
        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()
        print("done1")
        self.faceverse.exp_tensor = listener_vectors[:, :52].view(T, -1).to('cuda:2')  # 移动到相同的GPU设备上
#        self.faceverse.exp_tensor = listener_vectors[:,:52].view(T,-1).to(listener_vectors.get_device())

        self.faceverse.rot_tensor = listener_vectors[:,52:55].view(T, -1).to('cuda:2')#to(listener_vectors.get_device())
        self.faceverse.trans_tensor = listener_vectors[:,55:].view(T, -1).to('cuda:2')#to(listener_vectors.get_device())
        self.faceverse.id_tensor = self.id_tensor.view(1,150).repeat(T,1).view(T,150).to('cuda:2')#to(listener_vectors.get_device())
        print("done2")

        with torch.no_grad():
            pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
            rendered_img_r = pred_dict['rendered_img']
            rendered_img_r = np.clip(rendered_img_r.detach().cpu().numpy(), 0, 255)
            rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        print("done3")
        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to('cuda:2')#(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        print('chw')
        print(C)
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames =np.tile(listener_reference,(listener_vectors.shape[0],1,1)).reshape(
            listener_vectors.shape[0], C,H,W)
        print("done4")
        with torch.no_grad():
            for i in range(20): #20 751=37*19+48
                print(i)
                if i != 19:
                    listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                    semantics_copy = semantics[i * duration:(i + 1) * duration]
                else:
                    listener_reference_copy = listener_reference_frames[i * duration:]
                    semantics_copy = semantics[i * duration:]
                print(listener_reference_copy.shape) #(37, 1080, 1920, 3)
                print(semantics_copy.shape)   #torch.Size([37, 58, 27])
                listener_reference_copy=torch.from_numpy(listener_reference_copy).to('cuda:2')
                print(i)
                output_dict = self.pi_render(listener_reference_copy, semantics_copy)
                print(i)
                fake_videos = output_dict['fake_image']

                fake_videos = torch_img_to_np2(fake_videos)
                output_dict_list.append(fake_videos)
        print("done5")
        listener_videos = np.concatenate(output_dict_list, axis=0)

        out = cv2.VideoWriter(os.path.join(path, ind + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25,(448,224)) #(448,224))#(672, 224))
        for i in range(rendered_img_r.shape[0]):
            img = np.zeros((224,448,3),dtype=np.uint8)
            img[0:224, 0:224] = rendered_img_r[i]
            img[0:224, 224:448]=listener_videos[i]
            out.write(img)
        out.release()


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def main():
    #folder='/home/zhf/react2023/randL_k5_3dmm'
    folder=os.path.join(save_dir, "3DMMlabel") #load 3dmmlabel 
    #folder='/home/zhf/react2023/finetune_k6_3dmm'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        listener_vectors = np.load(file_path)
        listener_vectors=np.squeeze(listener_vectors,axis=1)
        listener_vectors=np.squeeze(listener_vectors,axis=1)
        print(listener_vectors.shape)
        listener_vectors = torch.tensor(listener_vectors, device='cuda:2')  # 将NumPy数组转换为PyTorch张量并移动到GPU上
        if '152153' in file_path:
            listener_reference = pil_loader(os.path.join(reference_dir, '152153.png'))      # you should put the listner first_frame with size of 256*256 in  scripts/reaction_to_3DMM/listenner_frame
        elif 'RECOLA' in file_path:
            listener_reference = pil_loader(os.path.join(reference_dir, 'recola.png'))
        elif '001' in file_path:
            listener_reference = pil_loader(os.path.join(reference_dir, '001.png'))
        elif '019' in file_path:
            listener_reference = pil_loader(os.path.join(reference_dir, '019.png'))
        else:
            listener_reference = pil_loader(os.path.join(reference_dir, '023.png'))
        listener_reference = Transform()(listener_reference)
        listener_reference = np.array(listener_reference)

        generat=Render()
        #output_path="scripts/reaction_to_3DMM/visual_finetune_100000_offline_s1_t0.33_k6_p1"
        #output_path="scripts/reaction_to_3DMM/visual_5s-randL-gp_150000_online_s5_t0.33_k5_p1"
        parts = filename.split('.')
        name = '.'.join(parts[:-2])
        output_index=name
        generat.rendering(output_path,output_index,listener_vectors,listener_reference)
        print("done")
if __name__ == '__main__':
    main()
    