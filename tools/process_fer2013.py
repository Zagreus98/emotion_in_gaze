import torch
from gaze_estimation.models import Edsr
import cv2
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def face_ToTensor(img):
    return (ToTensor()(img) - 0.5) * 2


def increase_resolution(img, srnet):
    face = face_ToTensor(img.copy()).to(torch.device('cuda'))
    with torch.no_grad():
        img_sp = srnet(face)
    img_sp = img_sp.cpu().numpy()
    img_sp = img_sp.transpose(1, 2, 0) / 2 + 0.5
    img_sp = cv2.resize(img_sp, (224, 224), interpolation=cv2.INTER_CUBIC)

    return img_sp

def crop_face(img, mtcnn):
    h, w, _ = img.shape
    with torch.no_grad():
        boxes, probs, points = mtcnn.detect(img, landmarks=True)
        if len(points) > 1 or len(points) == 0:
            return None
        points[points < 0] = 0
        points = points[0]
        x1 = int(max(0, np.min(points[:, 0])) - 50)
        y1 = int(max(0, np.min(points[:, 1])) - 50)
        x2 = int(min(w, np.max(points[:, 0])) + 50)
        y2 = int(min(h, np.max(points[:, 1])) + 50)
        face = img[x1:x2, y1:y2]
        plt.figure()
        plt.imshow(face)
        plt.show()
    return face



device = torch.device('cuda')
srnet = Edsr()
state_dict = torch.load(
    r'C:\Users\alexo\OneDrive\Desktop\Materiale TAID new\Criminalistica\SR_LRFR\pretrained\edsr_lambda0.5.pth',
    map_location='cpu'
)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=10,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

srnet.load_state_dict(state_dict)
srnet.to(device)

fer_train_root_path = Path(r'D:\datasets\fer_2013_aligned\train')
fer_val_root_path = Path(r'D:\datasets\fer_2013_aligned\test')
emotion_dirs_train = fer_train_root_path.glob('**/*')
emotion_dirs_val = fer_val_root_path.glob('**/*')

for emotion_dir in emotion_dirs_train:
    if emotion_dir.is_dir():
        images = emotion_dir.glob('*.png')
        for img_path in tqdm(images):
            img = cv2.imread(str(img_path))[..., ::-1]
            plt.figure()
            plt.imshow(img)
            plt.show()
            img = increase_resolution(img, srnet)
            plt.figure()
            plt.imshow(img)
            plt.show()
            cv2.imwrite(str(img_path), img * 255)

for emotion_dir in emotion_dirs_val:
    if emotion_dir.is_dir():
        images = emotion_dir.glob('*.png')
        for img_path in tqdm(images):
            img = cv2.imread(str(img_path))[..., ::-1]
            img = increase_resolution(img, srnet)
            cv2.imwrite(str(img_path), img * 255)
