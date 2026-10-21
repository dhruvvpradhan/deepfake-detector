import os
import torch
from torch.utils.data import Dataset
from utils.face_extractor import FaceExtractor

class DeepfakeVideoDataset(Dataset):
    """
    Dataset expecting a directory structure like:
      data/
        ├── real/  (video files: 0.mp4, 1.mp4, ...)
        └── fake/  (video files: 0.mp4, 1.mp4, ...)
    """
    def __init__(self, video_paths, labels, sequence_length=20, frame_size=224, device="cpu"):
        self.video_paths = video_paths
        self.labels = labels
        self.extractor = FaceExtractor(sequence_length=sequence_length, frame_size=frame_size, device=device)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        try:
            face_sequence = self.extractor.extract_faces_from_video(path)
        except Exception as e:
            # Fallback zero sequence if file reading encounters an error
            face_sequence = torch.zeros((20, 3, 224, 224), dtype=torch.float32)

        return face_sequence, label
