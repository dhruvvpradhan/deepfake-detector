import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

class FaceExtractor:
    def __init__(self, sequence_length=20, frame_size=224, device="cuda"):
        self.seq_len = sequence_length
        self.frame_size = frame_size
        self.device = device
        # Initialize MTCNN for cropped face extraction
        self.mtcnn = MTCNN(
            image_size=frame_size,
            margin=20,
            keep_all=False,
            select_largest=True,
            post_process=False,
            device=device
        )

    def extract_faces_from_video(self, video_path):
        """
        Reads a video file, isolates the primary face using MTCNN,
        and returns a normalized sequence of RGB face crops.
        """
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(frame_rgb)
        cap.release()

        total_frames = len(raw_frames)
        if total_frames == 0:
            raise ValueError(f"Could not read any frames from: {video_path}")

        # Uniformly sample indices across the video duration
        indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int)
        sampled_frames = [Image.fromarray(raw_frames[i]) for i in indices]

        face_crops = []
        # Fallback tensor for frames where face detection fails
        placeholder = torch.zeros((3, self.frame_size, self.frame_size), dtype=torch.float32)

        for img in sampled_frames:
            face = self.mtcnn(img)
            if face is not None:
                # MTCNN returns a tensor in [0, 255] range; normalize to [0, 1]
                face_crops.append(face / 255.0)
            else:
                face_crops.append(face_crops[-1] if face_crops else placeholder)

        # Output shape: (seq_len, 3, frame_size, frame_size)
        return torch.stack(face_crops)
