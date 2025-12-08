export PYTHONNOUSERSITE=1
unset PYTHONPATH

DATA_DIR=../dataset
BATCH=-1
TOTAL_BATCH=5


python ./demo_from_folder.py \
  --base_path $DATA_DIR/images \
  --outdir $DATA_DIR/images \
  --batch $(($BATCH)) \
  --total_batch $TOTAL_BATCH  

python ./dynamic_mask_point_tracking.py \
  --base_path $DATA_DIR/images \
  --original_video_path $DATA_DIR/original_video \
  --outdir $DATA_DIR/images \
  --batch $(($BATCH)) \
  --total_batch $TOTAL_BATCH  

python ./depth_warping_dynamic_mask_top_k_pytorch3d.py \
  --base_path $DATA_DIR/images \
  --outdir $DATA_DIR/cond_video \
  --batch $(($BATCH)) \
  --total_batch $TOTAL_BATCH  