### Important Commands to be used for the guide


## To rename file via bash
    - for f in *.jpg; do mv "$f" "${f%.jpg}_test_fake.jpg"; done 


## To copy files from the test/fake and test/real to test/merge
    - cp test/fake/*.jpg test/real/*.jpg test/merge/


## To get the number of files in curr-dir
    - find . -type f | wc -l


find /source/folder -type f -iname "*.mp4" | shuf | head -n 500 | xargs -I{} cp "{}" /destination/folder/


## To copy if too much files
    - rsync -a --include='*.png' --exclude='*' train_merged/ train-test-hybrid-optimal/


## To copy the fixed number of files
    - find /home/teaching/deepfake/Unsupervised_DF_Detection/1-Modification/D-FaceForensic++Merge/train_merged      -type f -name "*.png" | shuf | head -n 20000 | while read file; do
    cp "$file" /home/teaching/deepfake/Unsupervised_DF_Detection/1-Modification/D-FaceForensic++Merge/train_sample_20k/merged_20k
    done

 
## To get the sys info by storage size and so
    - df -h
        Filesystem      Size  Used Avail Use% Mounted on
        tmpfs            13G  9.8M   13G   1% /run
        /dev/nvme0n1p4  657G   29G  595G   5% /
        tmpfs            63G   11M   63G   1% /dev/shm
        tmpfs           5.0M  4.0K  5.0M   1% /run/lock
        efivarfs        128K   46K   78K  38% /sys/firmware/efi/efivars
        /dev/nvme0n1p1  1.9G  195M  1.6G  12% /boot
        /dev/nvme0n1p2  976M  6.1M  969M   1% /boot/efi
        /dev/sda        9.1T   28K  8.6T   1% /DATA
        /dev/nvme0n1p5  1.2T  351G  735G  33% /home
        tmpfs            13G  136K   13G   1% /run/user/1001
        tmpfs            13G   88K   13G   1% /run/user/128


## To get the sys info and so
    - du -sh ~/* | sort -h
        4.0K    /home/teaching/Desktop
        4.0K    /home/teaching/Documents
        4.0K    /home/teaching/donnn
        4.0K    /home/teaching/Music
        4.0K    /home/teaching/Public
        4.0K    /home/teaching/Templates
        4.0K    /home/teaching/Videos
        20K     /home/teaching/lag_trans.py
        28K     /home/teaching/venv
        556K    /home/teaching/Pictures
        15M     /home/teaching/lagrangain
        49M     /home/teaching/nltk_data
        103M    /home/teaching/d24136.zip
        1.2G    /home/teaching/snap
        1.5G    /home/teaching/anchal_code
        1.8G    /home/teaching/intern
        3.3G    /home/teaching/Downloads
        32G     /home/teaching/anaconda3
        54G     /home/teaching/deepfake
        58G     /home/teaching/d24136

## To get number of subfolders in a dir
    - find . -mindepth 1 -maxdepth 1 -type d | wc -l


## To get the file names in a file
    - find manipulated_sequences/DeepFakeDetection/c23/images -mindepth 1 -maxdepth 1 -type d -exec basename {} \; > video_folder_names.txt
    - find manipulated_sequences/DeepFakeDetection/c23/images -mindepth 1 -maxdepth 1 -type d -exec basename {} \; > folder_names.txt


## Remove folder and copy the image only to a new folder via renaming 
mkdir -p ../merged_images
for folder in */; do
  foldername="${folder%/}"
  for img in "$folder"*.png; do
    basename=$(basename "$img")
    newname="DFD_${foldername}_${basename}"
    mv "$img" "../merged_images/$newname"
  done
  rmdir "$folder"
done


# ------------------------------
## Copy the image from one to another folder
    - src=""
      target=""
      find "$src" -name "*.png" -print0 | xargs -0 -I{} cp "{}" "$target"


# ------------------------------
## What a PID is doing
    - ps -p <PID> -o pid,etime,cmd


# ------------------------------
## Kill the PID
    - kill -9 <PID>

# ------------------------------
# How many files of a type
    - echo "FSW_: $(ls DF_* 2>/dev/null | wc -l)"

# ------------------------------
# Remove specific images
    - rm yt_*.png


# ------------------------------
## Clear the cache by checking it

# Check size first
    - du -sh ~/.cache

# Then clean it
    - rm -rf ~/.cache/*


# -------------------------------
## Clean the conda env stuff

# See how much space is used
    - conda clean --all --dry-run

# Actually clean it
    - conda clean --all


# -------------------------------
## Clean the IDLE PID or so

# See your processes
    - ps aux | grep $USER

# Kill unwanted process by PID
    - kill -9 <PID>


# -------------------------------
## To copy some files from a folder to another
    - mkdir -p D-FF++Org/TestVids && find D-FF++Org/original_sequences/actors/c23/videos -type f | shuf -n 100 | xargs -I {} cp {} D-FF++Org/TestVids

# -----------------------------------
cd /path/to/your/folder

for file in *; do
    mv "$file" "DF_$file"
done


# -----------------------------------
## To remove random 90 vids from the current folder
    -ls *.mp4 | shuf | head -n 90 | xargs -I{} rm "{}"


# -----------------------------------
## Move files from one loc to another
    - mv /path/to/source_folder/* /path/to/destination_folder/
    - rsync --remove-source-files -av ~/deepfake/Unsupervised_DF_Detection/1-Modification/D-FF++Org/manipulated_sequences/NeuralTextures/c23/merged_images/ ~/deepfake/Unsupervised_DF_Detection/1-Modification/D-FF++Org/4kExtracted/


# -----------------------------------
## Rename the files inside folder
    - cd "Src DIR" && for folder in */; do foldername="${folder%/}"; for file in "$folder"/*; do filename=$(basename "$file"); mv "$file" "$folder/YT_${foldername}_${filename}"; done; done


# -----------------------------------

#!/bin/bash
mkdir -p merged_images

for dir in */; do
    mv "$dir"*.jpg merged_images/ 2>/dev/null
    mv "$dir"*.png merged_images/ 2>/dev/null
done

# tmux session attach
tmux attach-session -t pleasehoja





