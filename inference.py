import torch
from config import Config
from models.detector import DualBranchDetector
from utils.face_extractor import FaceExtractor

class DeepfakeInferenceEngine:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or Config.DEVICE
        self.extractor = FaceExtractor(
            sequence_length=Config.SEQUENCE_LENGTH,
            frame_size=Config.FRAME_SIZE,
            device=self.device
        )
        self.model = DualBranchDetector(
            sequence_length=Config.SEQUENCE_LENGTH,
            hidden_dim=Config.LSTM_HIDDEN_DIM,
            fft_dim=Config.FFT_FEAT_DIM
        ).to(self.device)

        if checkpoint_path:
            # Need to specify map_location if deploying on a machine without a GPU
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {checkpoint_path}")

        self.model.eval()

    def predict_video(self, video_path):
        """
        Runs an end-to-end inference pass on an input video file.
        Returns:
            dict: Classification label, fake probability, and status.
        """
        face_tensor = self.extractor.extract_faces_from_video(video_path)
        # Add batch dimension: (1, seq_len, 3, 224, 224)
        input_batch = face_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.model(input_batch).item()

        return {
            "prediction": "DEEPFAKE" if prob >= 0.5 else "AUTHENTIC",
            "fake_probability": round(prob, 4),
            "confidence_percentage": f"{prob * 100:.2f}%" if prob >= 0.5 else f"{(1 - prob) * 100:.2f}%"
        }

if __name__ == "__main__":
    # Example execution
    engine = DeepfakeInferenceEngine(checkpoint_path="best_deepfake_detector.pth")
    # result = engine.predict_video("sample_test_video.mp4")
    # print(result)
