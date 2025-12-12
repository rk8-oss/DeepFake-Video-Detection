import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import transforms
import cv2
import os
from tqdm import tqdm
from model import ECL

# Load your trained encoder (modify this based on your model definition)
def load_encoder(encoder_path):
    model = ECL()  # Your encoder model class
    checkpoint = torch.load(encoder_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model



# Extract frames from a video (modify depending on fps and how many frames you want)
def extract_frames(video_path, max_frames=32):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = preprocess_frame(frame)
        frames.append(frame)
        frame_count += 1

    cap.release()
    return torch.stack(frames)  # shape: [N, C, H, W]

# Preprocess each frame (modify this to match your encoder's input size and normalization)
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame)

# Step 1: Feature Extraction
def extract_video_features(video_paths, encoder):
    features = []
    for path in tqdm(video_paths, desc="Extracting Features"):
        frames = extract_frames(path).to('cuda')  # or 'cpu' if needed
        with torch.no_grad():
            frame_feats = encoder(frames)[0]  # shape: [N, D]
        video_feat = frame_feats.mean(dim=0)  # [D]
        features.append(video_feat.cpu())
    return torch.stack(features)

# Step 2: Binary Clustering with KMeans
def cluster_features(video_features):
    kmeans = KMeans(n_clusters=2, random_state=0)
    predicted_labels = kmeans.fit_predict(video_features.numpy())
    return predicted_labels, kmeans.cluster_centers_

# Step 3: Inter-frame Correlation
def inter_frame_correlation(frames):
    corrs = []
    for i in range(len(frames) - 1):
        frame1 = frames[i].cpu().numpy().flatten()
        frame2 = frames[i + 1].cpu().numpy().flatten()
        corr = np.corrcoef(frame1, frame2)[0, 1]
        corrs.append(corr)
    return np.mean(corrs)

# Step 4: Assign Final Labels
def assign_final_labels(video_paths, cluster_labels):
    cluster_corrs = {0: [], 1: []}
    for idx, path in enumerate(tqdm(video_paths, desc="Calculating Correlations")):
        frames = extract_frames(path)
        corr = inter_frame_correlation(frames)
        cluster_corrs[cluster_labels[idx]].append(corr)

    avg_corr_0 = np.mean(cluster_corrs[0])
    avg_corr_1 = np.mean(cluster_corrs[1])

    if avg_corr_0 > avg_corr_1:
        # Cluster 0 → Real, Cluster 1 → Fake
        final_labels = [0 if l == 0 else 1 for l in cluster_labels]
    else:
        # Cluster 1 → Real, Cluster 0 → Fake
        final_labels = [1 if l == 0 else 0 for l in cluster_labels]

    return final_labels

# Main pipeline
def stage3_pipeline(video_dir, encoder_path):
    # Load videos
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]

    # Load encoder
    encoder = load_encoder(encoder_path).to('cuda')

    # Extract features
    features = extract_video_features(video_paths, encoder)

    # Clustering
    cluster_labels, _ = cluster_features(features)

    # Final labeling
    final_labels = assign_final_labels(video_paths, cluster_labels)

    # Show results
    for path, label in zip(video_paths, final_labels):
        print(f"{os.path.basename(path)} → {'Fake' if label == 1 else 'Real'}")

    return final_labels

# Example usage
if __name__ == "__main__":
    VIDEO_DIR = "UADFVVideos"
    ENCODER_PATH = "resultSHH/ckpt_epoch_80.pth"
    stage3_pipeline(VIDEO_DIR, ENCODER_PATH)

