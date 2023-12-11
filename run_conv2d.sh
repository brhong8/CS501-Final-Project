#!/bin/bash -l

metrics='flop_count_dp,flop_count_sp,flop_count_hp,gld_transactions,gst_transactions,atomic_transactions,local_load_transactions,local_store_transactions,shared_load_transactions,shared_store_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions,system_read_transactions,system_write_transactions'

mkdir -p Results

kernel=(3 7 9) # Kernel Size
stride=(1 2 3) # Stride
size=(50 150 250) # Image Size

for i in ${kernel[@]}
do
  for j in ${stride[@]}
  do
    for k in ${size[@]}
    do
      res_gpu=Results/conv2d_kernel${i}_stride${j}_size${k}.time
      res_metrics=Results/conv2d_kernel${i}_stride${j}_size${k}.metrics
      nvprof --print-gpu-trace --log-file $res_gpu python conv2d.py --kernel_size $i --stride $j --height $k --width $k
      nvprof --log-file $res_metrics --metrics $metrics python conv2d.py --kernel_size $i --stride $j --height $k --width $k
    done
  done
done
