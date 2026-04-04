#-------------------------< Imports >--------------------------
import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split






#------------------------------------------------------< Functions >---------------------------------------------------------

def apply_video_augmentation(frames):
    augmented_frames = []
    
    for frame in frames:
        frame = frame.copy()
        
        # 1. Random Horizontal Flip (50% chance)
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)
        
        # 2. Random Brightness Adjustment (30% chance)
        if random.random() > 0.7:
            alpha = random.uniform(0.7, 1.3)
            frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)
        
        # 3. Random Rotation (-10 to +10 degrees, 30% chance)
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            frame = cv2.warpAffine(frame, M, (w, h))
        
        # 4. Random Contrast Adjustment (30% chance)
        if random.random() > 0.7:
            contrast = random.uniform(0.7, 1.3)
            frame = np.clip(frame * contrast, 0, 255).astype(np.uint8)
        
        augmented_frames.append(frame)
    
    return np.array(augmented_frames)

#--------------------------------------------------------------------------

def load_video(video_path, num_frames=16, augment=False):
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    cap.release()
    
    frames = np.array(frames)
    
    if augment:
        frames = apply_video_augmentation(frames)
    
    frames = frames.transpose(0, 3, 1, 2)
    frames = frames.astype(np.float32) / 255.0
    
    return frames

#---------------------------------------------------------------------------------

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.augment = augment
    #-----------------------------------------------
    def __len__(self):
        return len(self.video_paths)
    #-----------------------------------------------
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = load_video(video_path, self.num_frames, augment=self.augment)
        
        frames = torch.tensor(frames, dtype=torch.float32)
        
        return frames, label

#------------------------------------------------------------------------------------------------

def create_dataloaders(data_root, batch_size=8, num_frames=16, test_split=0.2, num_workers=2):
    video_paths = []
    labels = []
    
    shoplifting_dir = os.path.join(data_root, 'shop lifters')
    non_shoplifting_dir = os.path.join(data_root, 'non shop lifters')
    
    for filename in os.listdir(shoplifting_dir):
        if filename.endswith('.mp4'):
            video_paths.append(os.path.join(shoplifting_dir, filename))
            labels.append(1)
    
    for filename in os.listdir(non_shoplifting_dir):
        if filename.endswith('.mp4'):
            video_paths.append(os.path.join(non_shoplifting_dir, filename))
            labels.append(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        video_paths, labels, test_size=test_split, stratify=labels, random_state=42
    )
    
    train_dataset = VideoDataset(X_train, y_train, num_frames=num_frames, augment=True)
    test_dataset = VideoDataset(X_test, y_test, num_frames=num_frames, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


