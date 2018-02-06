#!/bin/bash
export PATH=$PATH:/cfs/klemming/nobackup/x/xshi/anaconda3/bin:/sbin
source activate data
ifconfig -a | grep inet
cd ~/mlScikitTensorflow
#jupyter notebook
