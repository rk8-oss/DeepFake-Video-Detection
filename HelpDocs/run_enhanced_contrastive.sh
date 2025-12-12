#!/bin/bash
# filepath: /home/teaching/deepfake/Unsupervised_DF_Detection/1-Modification/run_enhanced_contrastive.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Directory paths
DATA_DIR="/home/teaching/deepfake/Unsupervised_DF_Detection/1-Modification/resized_faces/collected_images"
OUTPUT_DIR="/home/teaching/deepfake/Unsupervised_DF_Detection/1-Modification/resultSH"
PSEUDO_LABELS="/home/teaching/deepfake/Unsupervised_DF_Detection/1-Modification/ResizedVin/image_pseudo_labels.json"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/logs

# Run enhanced contrastive learner with corrected parameters
python enhanced_contrastive_learner.py \
    --data_folder ${DATA_DIR} \
    --pseudo_label_file ${PSEUDO_LABELS} \
    --batch_size 32 \
    --learning_rate 0.0003 \
    --epochs 100 \
    --weight_decay 1e-4 \
    --num_workers 8 \
    --backbone "m2tr" \
    --temp 0.07 \
    --print_freq 10 \
    --image_size 224 \
    --select_confidence_sample 80 \
    --k 128

echo "Enhanced contrastive learning training completed!"