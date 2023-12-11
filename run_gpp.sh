#!/bin/bash -l
# Adapted from compile.survey and run.survey in https://github.com/cyanguwa/nersc-roofline/tree/master

# Compile GPP code
cd nersc-roofline/GPP/Volta

labels=('fma' 'nofma')
arr=(1 2 3 4 5 6)
for label in ${labels[@]}
do
  if [ $label = 'fma' ]; then
    sed -i 's/fmad=.*/fmad=true/g' Makefile
  else
    sed -i 's/fmad=.*/fmad=false/g' Makefile
  fi;

  for i in ${arr[@]}
  do
    sed -i "s/#define nend.*/#define nend $i/g" GPUComplex.h
    sed -i "s/gppKer_gpuComplex.ex.*/gppKer_gpuComplex.ex.$label.iw$i/g" Makefile
    make clean && make
  done
done

# Run GPP Code and obtain metrics
mkdir -p Results
metrics='flop_count_dp,flop_count_sp,flop_count_hp,gld_transactions,gst_transactions,atomic_transactions,local_load_transactions,local_store_transactions,shared_load_transactions,shared_store_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions,system_read_transactions,system_write_transactions'

for label in ${labels[@]}
do
  for i in ${arr[@]}
  do
    res_gpu=Results/results.$label.$i.time
    res_metrics=Results/results.$label.$i.metrics
    nvprof --print-gpu-trace --log-file $res_gpu ./gppKer_gpuComplex.ex.$label.iw$i 512 2 32768 20 0
    nvprof --kernels "NumBandNgpown_kernel" --log-file $res_metrics --metrics $metrics ./gppKer_gpuComplex.ex.$label.iw$i 512 2 32768 20 0
  done
done

