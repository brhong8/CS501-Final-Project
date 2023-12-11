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

flop_count_conv2d = 284405760 + 2580480
dram_read_conv2d  = 59270 + 326433
dram_write_conv2d = 322602 + 318801
execution_time_conv2d  = 1.0880e-6 + 1.3233e-3 + 85.326e-6
l2_read_conv2d  = 170163 + 367208 
l2_write_conv2d = 645133 + 322573

flop_count_conv2d2 = 1031965440 + 2168320
dram_read_conv2d2  = 274285 + 207395 + 6
dram_write_conv2d2 = 12 + 267170 + 271067
execution_time_conv2d2  = 1.0880e-6 + 1.3760e-6 + 4.5791e-3 + 71.531e-6
l2_read_conv2d2  = 32 + 1592720 + 308652
l2_write_conv2d2 = 28 + 542093 + 271053

flop_count_conv2d3 = 1520374464 + 1975680
dram_read_conv2d3  = 82 + 245979 + 247393
dram_write_conv2d3 = 24 + 274710 + 238227
execution_time_conv2d3  = 1.0240e-6 + 1.12e-6 + 5.5587e-3 + 64.586e-6
l2_read_conv2d3  = 147 + 2396671 + 278843
l2_write_conv2d3 = 53 + 493977 + 246989

t = np.linspace(0, 1e3, 10000)
s = dram_bw * t
s2 = l2_bw * t

x1, y1 = calculate_metrics(flop_count_conv2d, dram_read_conv2d, dram_write_conv2d, execution_time_conv2d)
x2, y2 = calculate_metrics(flop_count_conv2d2, dram_read_conv2d2, dram_write_conv2d2, execution_time_conv2d2)
x3, y3 = calculate_metrics(flop_count_conv2d3, dram_read_conv2d3, dram_write_conv2d3, execution_time_conv2d3)
lx1, ly1 = calculate_metrics(flop_count_conv2d, l2_read_conv2d, l2_write_conv2d, execution_time_conv2d)
lx2, ly2 = calculate_metrics(flop_count_conv2d2, l2_read_conv2d2, l2_write_conv2d2, execution_time_conv2d2)
lx3, ly3 = calculate_metrics(flop_count_conv2d3, l2_read_conv2d3, l2_write_conv2d3, execution_time_conv2d3)

fig, ax = plt.subplots()
ax.axhline(y=max_double, color='r', ls='-', label='FMA (FP32)')
ax.axhline(y=max_double / 2, color='r', ls='--', label='No FMA (FP32)')
plt.plot(x1, y1, 'ro', label='stride=3')
plt.plot(x2,y2, 'bo', label='stride=7')
plt.plot(x3,y3, 'go', label='stride=9')
# plt.plot(x4,y4, 'ro', label='iw4')
# plt.plot(x5,y5, 'ro', label='iw5')
# plt.plot(x6,y6, 'ro', label='iw6')
# plt.plot(x7, y7, 'bo', label='no_fma iw1')
# plt.plot(x8,y8, 'bo', label='no_fma iw2')
# plt.plot(x9,y9, 'bo', label='no_fma iw3')
# plt.plot(x10,y10, 'bo', label='no_fma iw4')
# plt.plot(x11,y11, 'bo', label='no_fma iw5')
# plt.plot(x12,y12, 'bo', label='no_fma iw6')
# plt.plot(x13, y13, 'yo', label='conv2d')
# plt.plot(lx1, ly1, 'r.', label='stride=3')
# plt.plot(lx2, ly2, 'b.', label='stride=7')
# plt.plot(lx3, ly3, 'g.', label='stride=9')
ax.loglog(t, s)
ax.loglog(t, s2)
plt.legend()
plt.ylim(1e2, 1e3)
plt.xlim(0.1, 1e2)


ax.set(xlabel='Arithmetic Intensity (FLOPs/Byte)', ylabel='Performance (GFLOP/sec)',
       title='Roofline Model for Nvidia GeForce GTX 1080')
ax.grid()


fig.savefig("conv2d.png")
plt.show()


