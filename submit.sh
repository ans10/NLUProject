#!/bin/bash
export JOB_NAME=$1
shift
export DL_ARGS=$@

echo $JOB_NAME
echo $DL_ARGS

qsub -N $JOB_NAME -v DL_ARGS -l nodes=1:ppn=2:gpus=1:titan,walltime=48:00:00,pmem=6GB -m ae -M myNYUID mycode.sh
