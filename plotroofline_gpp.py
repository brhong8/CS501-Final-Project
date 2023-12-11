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

flop_count_kpp = 1.2366e12
dram_read_kpp = 1.9427e10
dram_write_kpp = 10236633
execution_time_kpp = 5.28262

flop_count_kpp2 = 2.2809e12
dram_read_kpp2 = 1.9037e+10
dram_write_kpp2 = 18346482
execution_time_kpp2 = 9.67603

flop_count_kpp3 = 3.3252e12
dram_read_kpp3 = 1.6498e10
dram_write_kpp3 = 30542382
execution_time_kpp3 = 14.0827

flop_count_kpp4 = 4.3695e+12
dram_read_kpp4 = 1.5849e+10
dram_write_kpp4 = 36705753
execution_time_kpp4 = 18.4464

flop_count_kpp5 = 5.3038e+12
dram_read_kpp5 = 1.4866e+10
dram_write_kpp5 = 40840054
execution_time_kpp5 = 22.1823

flop_count_kpp6 = 6.3206e+12 
dram_read_kpp6 = 1.6304e+10
dram_write_kpp6 = 41035081
execution_time_kpp6 = 25.8155

flop_count_nofma_kpp = 1.2366e+12
dram_read_nofma_kpp = 1.9167e+10
dram_write_nofma_kpp = 7651291
execution_time_nofma_kpp = 7.44152

flop_count_nofma_kpp2 = 2.2534e+12
dram_read_nofma_kpp2 = 1.6160e+10
dram_write_nofma_kpp2 = 604845
execution_time_nofma_kpp2 = 12.8061

flop_count_nofma_kpp3 = 3.2702e+12
dram_read_nofma_kpp3 = 1.6742e+10
dram_write_nofma_kpp3 = 1308861
execution_time_nofma_kpp3 = 18.7507

flop_count_nofma_kpp4 = 4.2870e+12
dram_read_nofma_kpp4 = 1.5691e+10
dram_write_nofma_kpp4 = 739753
execution_time_nofma_kpp4 = 24.2948

flop_count_nofma_kpp5 = 5.3038e+12
dram_read_nofma_kpp5 = 1.5719e+10
dram_write_nofma_kpp5 = 1067119
execution_time_nofma_kpp5 = 29.7939

flop_count_nofma_kpp6 = 6.3206e+12
dram_read_nofma_kpp6 = 1.6113e+10
dram_write_nofma_kpp6 = 1778458
execution_time_nofma_kpp6 = 35.9099

flop_count_conv2d = 284405760 + 2580480
dram_read_conv2d  = 26116 + 322569
dram_write_conv2d = 322602 + 318486
execution_time_conv2d  = 1.0870e-6 + 1.3332e-3 + 85.631e-6
l2_read_conv2d  = 139459 + 363008
l2_write_conv2d = 645133 + 322573

t = np.linspace(0, 1e3, 10000)
s = dram_bw * t
s2 = l2_bw * t

x1, y1 = calculate_metrics(flop_count_kpp, dram_read_kpp, dram_write_kpp, execution_time_kpp)
x2, y2 = calculate_metrics(flop_count_kpp2, dram_read_kpp2, dram_write_kpp2, execution_time_kpp2)
x3, y3 = calculate_metrics(flop_count_kpp3, dram_read_kpp3, dram_write_kpp3, execution_time_kpp3)
x4, y4 = calculate_metrics(flop_count_kpp4, dram_read_kpp4, dram_write_kpp4, execution_time_kpp4)
x5, y5 = calculate_metrics(flop_count_kpp5, dram_read_kpp5, dram_write_kpp5, execution_time_kpp5)
x6, y6 = calculate_metrics(flop_count_kpp6, dram_read_kpp6, dram_write_kpp6, execution_time_kpp6)

x7, y7 = calculate_metrics(flop_count_nofma_kpp, dram_read_nofma_kpp, dram_write_nofma_kpp, execution_time_nofma_kpp)
x8, y8 = calculate_metrics(flop_count_nofma_kpp2, dram_read_nofma_kpp2, dram_write_nofma_kpp2, execution_time_nofma_kpp2)
x9, y9 = calculate_metrics(flop_count_nofma_kpp3, dram_read_nofma_kpp3, dram_write_nofma_kpp3, execution_time_nofma_kpp3)
x10, y10 = calculate_metrics(flop_count_nofma_kpp4, dram_read_nofma_kpp4, dram_write_nofma_kpp4, execution_time_nofma_kpp4)
x11, y11 = calculate_metrics(flop_count_nofma_kpp5, dram_read_nofma_kpp5, dram_write_nofma_kpp5, execution_time_nofma_kpp5)
x12, y12 = calculate_metrics(flop_count_nofma_kpp6, dram_read_nofma_kpp6, dram_write_nofma_kpp6, execution_time_nofma_kpp6)

x13, y13 = calculate_metrics(flop_count_conv2d, dram_read_conv2d, dram_write_conv2d, execution_time_conv2d)
lx1, ly1 = calculate_metrics(flop_count_conv2d, l2_read_conv2d, l2_write_conv2d, execution_time_conv2d)

fig, ax = plt.subplots()
ax.axhline(y=max_double, color='r', ls='-', label='FMA (FP32)')
ax.axhline(y=max_double / 2, color='r', ls='--', label='No FMA (FP32)')
plt.plot(x1, y1, 'ro', label='iw1')
plt.plot(x2,y2, 'bo', label='iw2')
plt.plot(x3,y3, 'go', label='iw3')
plt.plot(x4,y4, 'yo', label='iw4')
plt.plot(x5,y5, 'co', label='iw5')
plt.plot(x6,y6, 'mo', label='iw6')
plt.plot(x7, y7, 'r.', label='no_fma iw1')
plt.plot(x8,y8, 'b.', label='no_fma iw2')
plt.plot(x9,y9, 'g.', label='no_fma iw3')
plt.plot(x10,y10, 'y.', label='no_fma iw4')
plt.plot(x11,y11, 'c.', label='no_fma iw5')
plt.plot(x12,y12, 'm.', label='no_fma iw6')
ax.loglog(t, s)
plt.legend()
plt.ylim(1e2, 1e3)
plt.xlim(0.3, 1e2)


ax.set(xlabel='Arithmetic Intensity (FLOPs/Byte)', ylabel='Performance (GFLOP/sec)',
       title='Roofline Model for Nvidia GeForce GTX 1080')
ax.grid()


fig.savefig("gpp.png")
plt.show()


