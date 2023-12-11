import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def calculate_metrics(flop_count, dram_read, dram_write, execution_time):
    ai = flop_count / ((dram_read + dram_write) * 32)
    perf = flop_count / execution_time / 1e9
    return ai, perf

# Data for plotting
max_single = 8521.58
max_double = 356.8

dram_bw = 228.7
l2_bw = 1175.82

flop_count_spmv = 1155
dram_read_spmv  = 20
dram_write_spmv = 0
execution_time_spmv = 4.257e-6 + 2.9120e-6
l2_read_spmv  = 313 + 17
l2_write_spmv = 16 + 19

flop_count_gemm = 8.7964e+12 * 2
dram_read_gemm  = 5605157651 + 5635877729
dram_write_gemm = 33565255 + 34835825
execution_time_gemm = 2.38835
l2_read_gemm  = 8609722804 + 8611935586
l2_write_gemm = 33555728 + 34889424

flop_count_gemm2 = 3.1414e+11 * 8
dram_read_gemm2  = 212001186 * 8
dram_write_gemm2 = 2902345 * 8
execution_time_gemm2 = 371.91e-3
l2_read_gemm2  = 307384529 * 8
l2_write_gemm2 = 2785906 * 8

flop_count_gemm3 = 268697600 * 2
dram_read_gemm3  = 66722* 2
dram_write_gemm3 = 14511 * 2
execution_time_gemm3 = 99.133e-6
l2_read_gemm3  = 345976 * 2
l2_write_gemm3 = 32781 * 2

flop_count_gemm4 = 6669589 * 12
dram_read_gemm4  = 2129 * 12
dram_write_gemm4 = 660 * 12
execution_time_gemm4 = 51.453e-6
l2_read_gemm4  = 24746 * 12
l2_write_gemm4 = 1833 * 12

t = np.linspace(0, 1e3, 10000)
s = dram_bw * t
s2 = l2_bw * t

x1, y1 = calculate_metrics(flop_count_spmv, dram_read_spmv, dram_write_spmv, execution_time_spmv)
lx1, ly1 = calculate_metrics(flop_count_spmv, l2_read_spmv, l2_write_spmv, execution_time_spmv)
x2, y2 = calculate_metrics(flop_count_gemm, dram_read_gemm, dram_write_gemm, execution_time_gemm)
lx2, ly2 = calculate_metrics(flop_count_gemm, l2_read_gemm, l2_write_gemm, execution_time_gemm)
x3, y3 = calculate_metrics(flop_count_gemm2, dram_read_gemm2, dram_write_gemm2, execution_time_gemm2)
lx3, ly3 = calculate_metrics(flop_count_gemm2, l2_read_gemm2, l2_write_gemm2, execution_time_gemm2)
x4, y4 = calculate_metrics(flop_count_gemm3, dram_read_gemm3, dram_write_gemm3, execution_time_gemm3)
lx4, ly4 = calculate_metrics(flop_count_gemm3, l2_read_gemm3, l2_write_gemm3, execution_time_gemm3)

fig, ax = plt.subplots()
ax.axhline(y=max_single, color='r', ls='-', label='FMA (FP16)')
ax.axhline(y=max_single / 2, color='r', ls='--', label='No FMA (FP16)')

plt.plot(x2, y2, 'bo', label='maxwell_sgemm_128x128_nn')
plt.plot(x3, y3, 'go', label='sgemm_128x128x8_NN_vec')
plt.plot(x4, y4, 'yo', label='maxwell_sgemm_128x64_nn')
ax.loglog(t, s)
plt.legend()
plt.ylim(1e3, 1e4)
plt.xlim(10, 1e3)


ax.set(xlabel='Arithmetic Intensity (FLOPs/Byte)', ylabel='Performance (GFLOP/sec)',
       title='Roofline Model for Nvidia GeForce GTX 1080')
ax.grid()


fig.savefig("SGEMM.png")
plt.show()


