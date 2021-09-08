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
and then activate the conda environment:
```
conda activate mkenv
```
Then PyTorch and everything should be ready.



