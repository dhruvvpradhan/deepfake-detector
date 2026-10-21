import torch

class Config:
    # Processing parameters
    SEQUENCE_LENGTH = 20      # Number of face frames sampled per video
    FRAME_SIZE = 224          # H x W input dimensions for CNNs
    SAMPLE_RATE = 2           # Sample every Nth frame to reduce redundancy
    
    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.1
    
    # Model parameters
    CNN_BACKBONE = "resnext50_32x4d"
    LSTM_HIDDEN_DIM = 256     # Hidden size per direction (256 * 2 = 512 total)
    FFT_FEAT_DIM = 128
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
