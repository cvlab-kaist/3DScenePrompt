# sea-raft ckpt
cd ./Depth-Anything-3/third_party/RAFT
gdown --fuzzy https://drive.google.com/file/d/1a0C5FTdhjM4rKrfXiGhec7eq2YM141lu/view?usp=drive_link -O models/
cd ../../../

# sam2 ckpt
cd ./Depth-Anything-3/third_party/sam2/checkpoints
bash download_ckpts.sh
cd ../../../../