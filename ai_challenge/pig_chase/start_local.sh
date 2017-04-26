#!/bin/bash
#cd ~/Freiburg/Malmo/Minecraft/
#./launchClient.sh -port 10000 &
#./launchClient.sh -port 10001 &
ps -ef | grep 'python -m visdom.server' | grep -v grep | awk '{print $2}' | xargs kill
python -m visdom.server &  
OMP_NUM_THREADS=1 python pig_chase_a3c.py \
    malmo1:10000 \
    malmo2:10000 \
    --num-processes 1
    #> stderr_local_out 2>&1
#OMP_NUM_THREADS=1 python pig_chase_a3c.py malmo1:10000 malmo2:10000
#OMP_NUM_THREADS=1 python pig_chase_a3c.py localhost:10000 localhost:10001
