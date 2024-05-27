#!/bin/bash

## #作业名称，调整区间，跑第一步

#SBATCH --job-name=LISAML

## #使用的分区，sinfo指令可查看分区信息，必选项

#SBATCH --partition=cpu-pgmf

## #使用节点数量，必选项

#SBATCH --nodes=10

## #总的任务数，也可以理解为进程数(默认为每个任务进程分配1个cpu核)，必选项

#SBATCH --ntasks=10

## #每个任务进程需要的cpu核数，针对多线程任务，默认1，可选项

#SBATCH --cpus-per-task=52

## #每个节点使用的GPU卡数量，CPU作业此项无需指定--nodelist=gpu1

## #SBATCH --gres=gpu:4

## #指定从哪个项目扣费，如果没有这条参数，则从个人账户扣费

## #SBATCH --comment=LISA_Machine_Learning

## #指定std输出和错误文件
#SBATCH --output=DNN_1_%A.out
#SBATCH --error=DNN_1_%A.err

## 邮件通知
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yiqian@hust.edu.cn

## Activate Conda Env
cd /home/ud202180035/DNN_for_GW_Parameter_Space_Estimation
./LISA_Templates_Generator_1.py
