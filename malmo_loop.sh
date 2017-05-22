#!/bin/bash

start_time=$(date +%s)
cd ./malmo-challenge/docker/malmopy-ai-challenge
docker stack deploy --compose-file=docker-compose.yml malmo_stack

while true
do 
    right_time=$(date +%s)
    diff_time=`expr $right_time - $start_time`
    diff_=`expr $diff_time % 1500`
    
    if [  $diff_ -eq 0  ]
    then
        docker stack rm malmo_stack 
        docker stack deploy --compose-file=docker-compose.yml malmo_stack
        sleep 10
    fi 

done
