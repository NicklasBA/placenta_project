# placenta_project
 Fetal-maternal haemorrhage project

# How to cluster
Login the cluster using:
```
ssh s183993@thinlinc.compute.dtu.dk
```
and the password provided by Peter.
GPU clusters availible:
```
ssh hyperion
``` 
see GPU status:
```
gpustat
```
Start a new session e.g.:
```
tmux new -s test
```
Attach to an existing tmux session:
```
tmux attach -t test
```
and then activate the conda environment:
```
conda activate mkenv
```
# Running SlowFast
Train a standard model:
```
python /home/s183993/placenta_project/SlowFast/tools/run_net.py \
  --cfg /home/s183993/placenta_project/Config.yaml\
  --gpu 0
python /home/s183993/placenta_project/SlowFast/tools/run_net.py \
  --cfg /home/s183993/placenta_project/video_config_3.yaml\
  --gpu 4
```
Test a model
```
python /home/s183993/placenta_project/SlowFast/tools/run_net.py \
  --cfg /home/s183993/placenta_project/Config.yaml \
  --gpu 0 \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```
Annotate images
```

```
