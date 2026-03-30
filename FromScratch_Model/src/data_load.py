





#-------------------------< Imports >--------------------------
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split







#------------------------------------------------------< Functions >---------------------------------------------------------

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
    #------------------------------------------------
    def __len__(self):
        return len(self.video_paths)
    #------------------------------------------------
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video(video_path)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        frames = torch.stack(frames)
        
        return frames, label

#---------------------------------------------------------------------

def load_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
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
    frames = frames.transpose(0, 3, 1, 2)
    frames = frames.astype(np.float32) / 255.0
    
    return frames

#---------------------------------------------------------------------------

def create_dataloaders(data_root, batch_size=8, num_frames=16, val_split=0.2, num_workers=2):
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
    
    X_train, X_val, y_train, y_val = train_test_split(
        video_paths, labels, test_size=val_split, stratify=labels, random_state=42
    )
    
    train_dataset = VideoDataset(X_train, y_train, num_frames=num_frames)
    val_dataset = VideoDataset(X_val, y_val, num_frames=num_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader




