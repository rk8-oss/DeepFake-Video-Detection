import os
import argparse
import cv2
import numpy as np
import glob
import json
import logging
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
from model import ECL
from scipy.stats import spearmanr
from data.transform import get_transforms
from lib.test_util import get_crop
from sklearn.cluster import KMeans
from collections import Counter


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, help='The path of test data.', default="all_mix/TestVideos")
    parser.add_argument('--checkpoint_path', type=str, help='The path of trained encoder.', default="savevin/SupCon/_models/2025-04-30 15-37-16/last.pth")
    parser.add_argument('--log_path', type=str, help='Path to save the log file.', default='run.log')
    args = parser.parse_args()
    return args

def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECL()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    return model


def get_features(cropped_face, encoder):
    test_transform = get_transforms(name="val")
    input_tensor = test_transform(cropped_face)
    input_tensor = np.reshape(input_tensor, (1, *input_tensor.shape))
    input_tensor = input_tensor.cuda(non_blocking=True)
    with torch.no_grad():
        output, _ = encoder(input_tensor)
    return output.cpu().detach().numpy()


def main():
    args = args_func()
    setup_logger(args.log_path)

    logging.info("Starting process...")
    test_data_path = args.test_data_path
    test_data_list = glob.glob(os.path.join(test_data_path, '*'))
    video_extensions = ['.mp4', '.mkv', '.avi']
    test_data_list = [f for f in test_data_list if os.path.splitext(f.lower())[1] in video_extensions]
    test_data_list = test_data_list[:100]  # Only take the first 100 videos

    logging.info(f"Loaded {len(test_data_list)} videos for processing.")

    mtcnn = MTCNN(device='cuda:0').eval()
    encoder = get_model(args)
    encoder.eval()

    features_total = []
    correlation_total = []

    with tqdm(total=len(test_data_list), desc='Processing videos', unit='video') as pbar:
        for video_path in test_data_list:
            cap = cv2.VideoCapture(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            prev_descriptor = None
            correlation_sum = []
            video_features = []

            v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in range(v_len):
                ret, frame = cap.read()
                if not ret:
                    break

                cropped_face = get_crop(frame, mtcnn)
                if cropped_face is None:
                    continue

                output = get_features(cropped_face, encoder)
                video_features.append({"video_name": video_name, "features": output})

                if prev_descriptor is not None:
                    corr, _ = spearmanr(output[0], prev_descriptor[0])
                    if not np.isnan(corr):
                        correlation_sum.append(corr)

                prev_descriptor = output

            if len(correlation_sum) < 3:
                logging.warning(f"Video {video_name} skipped due to insufficient frames.")
                cap.release()
                pbar.update(1)
                continue

            avg_corr = sum(correlation_sum) / len(correlation_sum)
            correlation_total.append({"video_name": video_name, "correlation": avg_corr})
            features_total.extend(video_features)

            cap.release()
            pbar.update(1)

    if not features_total:
        logging.error("No valid features extracted. Exiting.")
        return

    features = np.array([item['features'].flatten() for item in features_total])
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)

    image_labels = {}
    for i, item in enumerate(features_total):
        video_id = item['video_name']
        image_labels.setdefault(video_id, []).append(labels[i])

    video_labels = {}
    for video_id, label_list in image_labels.items():
        common_label = Counter(label_list).most_common(1)[0][0]
        video_labels[video_id] = common_label

    video_correlation = {item["video_name"]: item["correlation"] for item in correlation_total}

    keys_0 = [k for k, v in video_labels.items() if v == 0]
    keys_1 = [k for k, v in video_labels.items() if v == 1]

    values_0 = [video_correlation[k] for k in keys_0 if k in video_correlation]
    values_1 = [video_correlation[k] for k in keys_1 if k in video_correlation]

    if not values_0 or not values_1:
        logging.error("Warning: One of the clusters has no data. Cannot determine real/fake reliably.")
        return

    avg_0 = sum(values_0) / len(values_0)
    avg_1 = sum(values_1) / len(values_1)

    if avg_0 > avg_1:
        fake_keys, real_keys = keys_0, keys_1
    else:
        fake_keys, real_keys = keys_1, keys_0

    final_labels = {}
    for key in fake_keys:
        final_labels[key] = "FAKE"
    for key in real_keys:
        final_labels[key] = "REAL"

    json_data = [{"video_name": k, "pred_label": v} for k, v in final_labels.items()]
    with open("bin_result.json", "w") as f:
        json.dump(json_data, f, indent=2)

    logging.info("Prediction saved to bin_result.json")
    logging.info("Processing completed successfully.")


if __name__ == '__main__':
    main()
    