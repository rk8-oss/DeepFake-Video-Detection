#!/usr/bin/env python3

import os
import sys
import argparse
import time
import numpy as np
import datetime
import json
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from lib.train_util import AverageMeter, save_model
from data.transform import TwoCropTransform, get_transforms
from data.dataset import CustomImageDataset
from loss import ECLoss
from model import M2TRECL

import logging
def args_func():
    parser = argparse.ArgumentParser('argument for enhanced contrastive learner training with M2TR')
    
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--select_confidence_sample', type=int, default=None, help='epoch of select confidence sample')
    parser.add_argument('--k', type=float, default=None, help='select top k confidence samples')
    parser.add_argument('--temp', type=float, default=.07, help='temperature for loss function')
    # Updated the backbone default to M2TR
    parser.add_argument('--backbone', type=str, default='M2TR')
    parser.add_argument('--data_folder', type=str, default=None, help='path to DF dataset')
    parser.add_argument('--pseudo_label_file', type=str, default=None, help='path to pseudo_label.json')
    parser.add_argument('--image_size', type=int, default=304, help='parameter for image size')
    # Added M2TR specific parameters
    parser.add_argument('--patch_size', type=int, default=16, help='patch size for M2TR')
    parser.add_argument('--dim', type=int, default=768, help='embedding dimension for M2TR')
    parser.add_argument('--depth', type=int, default=12, help='transformer depth for M2TR')
    parser.add_argument('--heads', type=int, default=12, help='number of attention heads for M2TR')
    parser.add_argument('--mlp_dim', type=int, default=3072, help='MLP dimension for M2TR')

    args = parser.parse_args()
    return args

def setup_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console Handler
#     ch = logging.StreamHandler()
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def set_loader(pseudo_label_dict, args):
    # construct data loader

    train_transform = get_transforms(name="train", image_size=args.image_size)
    train_dataset = CustomImageDataset(pseudo_label_dict, args.data_folder, TwoCropTransform(train_transform))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    val_transform = get_transforms(name="val", image_size=args.image_size)

    select_confidence_dataset = CustomImageDataset(pseudo_label_dict, args.data_folder, val_transform)

    select_confidence_loader = torch.utils.data.DataLoader(
        select_confidence_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, select_confidence_loader


def set_model(args):
    # Initialize the M2TR model
    model = M2TRECL(out_dim=128, image_size=args.image_size)
    
    # Initialize weights
    model.init_weights()
    
    criterion = ECLoss(temperature=args.temp)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    """
    Training function for one epoch
    
    Args:
        train_loader: DataLoader for training data
        model: Neural network model
        criterion: Loss function
        optimizer: Optimizer for updating weights
        epoch: Current epoch number
        scheduler: Learning rate scheduler
        args: Additional arguments
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):

        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # compute loss
        _, features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # Adam optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step() 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    # Save model checkpoint every 10 epochs
    if epoch % 10 == 0:
        save_file = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
        print(f'==> Saving checkpoint at epoch {epoch} to {save_file}')
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=save_file)
    
    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(os.path.dirname(filename), 'model_best.pth')
        shutil.copyfile(filename, best_file)


def select_confidence_sample(model, data_loader, args):
    """Select samples with high confidence for pseudo-labeling."""
    print("Selecting confidence samples at epoch {}...".format(args.select_confidence_sample))
    model.eval()
    
    # Collect features and file names
    all_features = []
    all_filenames = []
    
    with torch.no_grad():
        for images, _, filenames in data_loader:
            images = images.cuda(non_blocking=True)
            _, features = model(images)
            features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_filenames.extend(filenames)
    
    all_features = np.vstack(all_features)
    
    # Normalize features
    all_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(all_features)
    
    # Get cluster labels and distances
    labels = kmeans.labels_
    distances = kmeans.transform(all_features)  # Distance from each point to each cluster center
    
    # Get unique cluster labels (in case KMeans only found 1 cluster)
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Check if we have enough clusters
    if num_clusters < 2:
        print(f"Warning: KMeans only found {num_clusters} clusters instead of 2.")
        # Use distance-based approach rather than cluster-based
        # Sort by the distance to the single cluster center
        sorted_indices = np.argsort(distances[:, 0])
        
        # Take the top and bottom percentage as real and fake
        total_samples = len(sorted_indices)
        num_confident = int(total_samples * 0.2)  # Take 20% as confident
        
        # Bottom indices (closest to center) = confident class 0
        confident_0_indices = sorted_indices[:num_confident]
        # Top indices (furthest from center) = confident class 1
        confident_1_indices = sorted_indices[-num_confident:]
        
        # Create a dictionary with filename as key and label as value
        confidence_label_dict = {}
        
        for idx in confident_0_indices:
            confidence_label_dict[all_filenames[idx]] = "0"
        
        for idx in confident_1_indices:
            confidence_label_dict[all_filenames[idx]] = "1"
    else:
        # Original logic for when we have 2 clusters
        all_distances = np.zeros((len(all_features), num_clusters))
        for i in range(num_clusters):
            all_distances[:, i] = np.linalg.norm(all_features - kmeans.cluster_centers_[i], axis=1)
        
        # Get distances to cluster centers for samples in each cluster
        distances_cluster_0 = all_distances[np.array([i for i, label in enumerate(labels) if label == 0]), 0]
        distances_cluster_1 = all_distances[np.array([i for i, label in enumerate(labels) if label == 1]), 1]
        
        # Calculate threshold for each cluster (e.g., 20% closest points)
        threshold_0 = np.percentile(distances_cluster_0, 20) if len(distances_cluster_0) > 0 else float('inf')
        threshold_1 = np.percentile(distances_cluster_1, 20) if len(distances_cluster_1) > 0 else float('inf')
        
        # Create a dictionary with filename as key and label as value
        confidence_label_dict = {}
        
        for i, (filename, label, distance) in enumerate(zip(all_filenames, labels, all_distances)):
            if label == 0 and distance[0] <= threshold_0:
                confidence_label_dict[filename] = "0"
            elif label == 1 and distance[1] <= threshold_1:
                confidence_label_dict[filename] = "1"
    
    print(f"Selected {len(confidence_label_dict)} confident samples out of {len(all_filenames)}")
    return confidence_label_dict


def save_path(args):
    data_folder_name = os.path.basename(args.data_folder)
    model_path = './resultSHH/SupCon/{}_models'.format(data_folder_name)
    tb_path = './resultSHH/SupCon/{}_tensorboard'.format(data_folder_name)

    save_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    tb_folder = os.path.join(tb_path, save_time)
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)

    save_folder = os.path.join(model_path, save_time)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    return tb_folder, save_folder


def main():
    args = args_func()

    with open(args.pseudo_label_file, 'r') as file:
        pseudo_label_dict = json.load(file)

    # build data loader
    train_loader, select_confidence_loader = set_loader(pseudo_label_dict, args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1
    )

    tb_folder, save_folder = save_path(args)

    # Setup logger
    log_file_path = os.path.join(save_folder, 'training.log')
    logger2 = setup_logger(log_file_path)

    logger = tb_logger.Logger(logdir=tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        time1 = time.time()

        if args.select_confidence_sample and epoch % args.select_confidence_sample == 0 and epoch != args.epochs:
            # save the model from the previous cycle
            save_file = os.path.join(save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

            # select the training sample for the new cycle
            print(f"Selecting confidence samples at epoch {epoch}...")
            confidence_label_dict = select_confidence_sample(model, select_confidence_loader, args)
            train_loader, _ = set_loader(confidence_label_dict, args)

        # Then after each training epoch:
        loss = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr}")
        logger2.info(f"Current learning rate: {current_lr}")

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        logger2.info('Epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', current_lr, epoch)

    # save the last model
    save_file = os.path.join(save_folder, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)


if __name__ == '__main__':
    main()


# Example command to run training with M2TR:
# python enhanced_contrastive_learner_m2tr.py \
#   --data_folder UADFV \
#   --pseudo_label_file output/image_pseudo_labels.json \
#   --epochs 50 \
#   --select_confidence_sample 10 \
#   --k 0.5 \
#   --batch_size 16 \
#   --learning_rate 0.0001 \
#   --backbone M2TR \
#   --num_workers 8 \
#   --image_size 288 \
#   --patch_size 16 \
#   --dim 768 \
#   --depth 12 \
#   --heads 12

# # python enhanced_contrastive_learner.py   --data_folder UADFV   --pseudo_label_file output/image_pseudo_labels.json   --epochs 20   --select_confidence_sample 10   --k 0.5   --batch_size 16   --learning_rate 0.0001   --backbone Xception   --num_workers 8
# # python enhanced_contrastive_learner.py   --data_folder UADFV   --pseudo_label_file output/image_pseudo_labels.json   --epochs 50   --select_confidence_sample 10   --k 0.5   --batch_size 32   --learning_rate 0.0001   --backbone Xception   --num_workers 16

