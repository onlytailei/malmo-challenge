#!/bin/bash
#cd ~/Freiburg/Malmo/Minecraft/
#./launchClient.sh -port 10000 &
#./launchClient.sh -port 10001 &
ps -ef | grep 'python -m visdom.server' | grep -v grep | awk '{print $2}' | xargs kill
python -m visdom.server &  
OMP_NUM_THREADS=1 python pig_chase_a3c.py \
    malmo1:10000 \
    malmo2:10000 \
    malmo3:10000 \
    malmo4:10000 \
    malmo5:10000 \
    malmo6:10000 \
    malmo7:10000 \
    malmo8:10000 \
    malmo9:10000 \
    malmo10:10000 \
    malmo11:10000 \
    malmo12:10000 \
    malmo13:10000 \
    malmo14:10000 \
    malmo15:10000 \
    malmo16:10000 \
    malmo17:10000 \
    malmo18:10000 \
    malmo19:10000 \
    malmo20:10000 \
    malmo21:10000 \
    malmo22:10000 \
    malmo23:10000 \
    malmo24:10000 \
    malmo25:10000 \
    malmo26:10000 \
    malmo27:10000 \
    malmo28:10000 \
    malmo29:10000 \
    malmo30:10000 \
    malmo31:10000 \
    malmo32:10000 \
    --num-processes 16
   # > ./stderr/stderr_swarm_out 2>&1
#OMP_NUM_THREADS=1 python pig_chase_a3c.py malmo1:10000 malmo2:10000
#OMP_NUM_THREADS=1 python pig_chase_a3c.py localhost:10000 localhost:10001
