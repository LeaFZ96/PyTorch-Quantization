import matplotlib.pyplot as plt

# int 12 - 12 - 12
y0 = [26.280, 40.530, 45.990, 54.180, 58.730, 66.770, 71.480, 72.790, 77.260, 76.630, 79.330, 79.110, 80.750, 82.070, 83.060, 83.330, 84.130, 83.700, 84.030, 84.010, 86.200, 86.090, 84.920, 87.110, 86.600, 87.220, 87.700, 87.300, 88.250, 87.240, 90.840, 90.860, 91.220, 91.400, 91.400, 91.470, 91.600, 91.660, 91.710, 91.890, 92.100, 91.860, 91.970, 92.120, 91.920, 91.890, 92.040, 92.190, 92.140, 91.940]
x0 = range(len(y0))

# int 10 - 10 - 10
y1 = [27.550, 37.510, 47.190, 54.250, 62.510, 67.780, 70.930, 74.830, 76.240, 78.450, 79.740, 80.200, 82.550, 82.760, 82.680, 83.570, 85.300, 85.020, 84.200, 85.260, 86.250, 85.110, 85.970, 86.790, 86.820, 87.540, 87.550, 87.580, 88.340, 88.870, 88.870, 91.120, 91.120, 91.330, 91.470, 91.650, 91.650, 91.640, 91.700, 91.750, 91.840, 91.680, 92.120, 91.810, 91.730, 92.000, 91.880, 92.090, 92.120, 92.090, 92.000]
x1 = range(len(y1))

# int 8 - 8 - 8
y2 = [30.980, 38.830, 47.820, 58.290, 60.160, 68.200, 71.050, 75.240, 76.380, 79.030, 78.380, 79.470, 81.680, 82.170, 82.560, 83.770, 82.880, 85.230, 85.060, 85.670, 86.130, 85.980, 86.870, 87.080, 86.660, 87.580, 87.090, 88.050, 88.170, 87.560, 90.370, 90.710, 90.970, 91.090]
x2 = range(len(y2))

# int 6 - 6 - 6
y3 = [31.170, 40.250, 47.980, 55.750, 57.180, 67.130, 72.190, 73.720, 75.530, 78.200, 77.970, 78.430, 80.880, 82.330, 82.510, 83.960, 83.600, 84.220, 85.670, 85.600, 84.790, 86.710, 86.250, 86.860, 86.920, 87.820, 86.890, 87.110, 87.980, 87.910, 90.780, 91.010, 91.170, 91.060, 91.250, 91.470, 91.320, 91.410, 91.610, 91.550, 91.220, 91.810, 91.750, 91.670, 91.640, 91.520, 91.550, 92.010, 91.850, 91.760]
x3 = range(len(y3))

# int 4 - 4 - 4
y4 = [25.170, 41.530, 46.420, 53.900, 56.800, 61.820, 65.280, 68.560, 71.940, 73.480, 75.280, 75.120, 75.890, 77.190, 77.540, 78.800, 77.840, 78.650, 79.660, 80.190, 79.700, 79.530, 80.920, 80.670, 81.480, 81.770, 82.200, 82.140, 82.150, 82.210, 85.060, 85.020, 85.430, 85.330, 85.400, 85.120, 85.550, 85.260, 85.510, 85.970, 85.730, 85.690, 85.880, 85.800, 85.560, 85.640, 85.530, 85.830, 85.360, 85.610]
x4 = range(len(y4))

# int 4 - 4 - 6
y5 = [30.430, 39.470, 46.480, 52.260, 58.200, 66.240, 70.420, 73.660, 76.270, 77.850, 79.490, 80.420, 80.440, 81.740, 81.530, 83.650, 84.700, 85.040, 83.860, 86.490, 86.230, 85.520, 86.240, 86.980, 87.060, 86.740, 88.230, 87.940, 88.540, 87.740, 90.820, 90.940, 91.170, 91.380, 91.380, 91.250, 91.590, 91.450, 91.620, 91.650, 91.540, 91.540, 91.810, 91.690, 91.360, 91.830, 91.760, 91.880, 92.030, 91.780]
x5 = range(len(y5))

# int 4 - 4 - 8
y6 = [29.690, 38.660, 48.770, 55.040, 60.740, 66.460, 70.900, 73.010, 75.930, 79.230, 78.810, 78.970, 81.380, 81.680, 82.450, 83.330, 83.910, 84.440, 85.980, 85.100, 85.990, 86.770, 86.640, 87.330, 86.670, 87.940, 88.020, 87.980, 87.250, 88.530, 90.670, 91.190, 91.080, 91.570, 91.400, 91.620, 91.530, 91.770, 91.670, 91.720, 91.860, 91.540, 91.760, 91.740, 91.720, 92.210, 91.790, 92.050, 91.960, 92.190]
x6 = range(len(y6))

# int 6 - 6 - 4
y7 = [30.630, 40.300, 47.700, 53.580, 59.200, 64.250, 67.080, 66.160, 71.330, 71.760, 74.720, 73.930, 75.710, 75.980, 78.880, 77.440, 78.750, 79.940, 78.300, 80.220, 79.910, 81.000, 81.120, 82.100, 82.070, 81.880, 81.880, 80.900, 82.820, 81.640, 85.400, 85.570, 85.430, 85.210, 85.740, 85.700, 85.600, 85.670, 85.500, 85.730, 86.090, 85.740, 85.890, 85.720, 85.890, 86.020, 85.960, 85.980, 86.100, 85.800]
x7 = range(len(y7))

# int 8 - 8 - 4
y8 = [31.750, 38.860, 44.990, 53.780, 56.820, 64.770, 67.160, 70.400, 72.360, 73.250, 74.750, 75.150, 76.460, 77.130, 77.850, 79.470, 79.060, 79.050, 79.340, 80.820, 80.930, 81.340, 80.490, 80.160, 81.330, 80.190, 81.270, 82.460, 81.140, 81.080, 84.920, 84.960, 85.490, 85.270, 85.040, 85.560, 85.430, 85.560, 85.460, 85.370, 85.360, 85.230, 85.450, 85.310, 85.730, 85.220, 85.650, 85.720, 85.650, 85.220]
x8 = range(len(y8))

y9 = [28.580, 39.460, 42.560, 51.150, 57.560, 59.360, 68.230, 71.940, 74.090, 77.910, 78.360, 74.400, 78.540, 82.370, 82.320, 81.000, 80.950, 82.730, 84.220, 83.830, 85.160, 85.770, 84.240, 84.350, 86.930, 86.070, 87.880, 86.060, 87.010, 88.950, 91.240, 91.550, 91.770, 91.800, 91.870, 91.880, 91.800, 92.270, 92.140, 92.080, 92.250, 92.420, 92.470, 92.340, 92.340, 92.400, 92.280, 92.680, 92.500, 92.670]
x9 = range(len(y9))

ticks = [(i + 1) * 10 for i in range(10)]

plt.figure(1)
plt.plot(x0, y0, label='int 12-12-12', color='c')
plt.plot(x1, y1, label='int 10-10-10', color='r')
plt.plot(x2, y2, label='int 8-8-8', color='y')
plt.plot(x3, y3, label='int 6-6-6', color='g')
plt.plot(x4, y4, label='int 4-4-4', color='b')
plt.plot(x9, y9, label='float')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.title('Same input and weight quantization')
plt.legend()
plt.grid(True)
plt.axis([-2, 50, 10, 100])
plt.yticks(ticks)

plt.figure(2)
plt.plot(x5, y5, label='int 4-4-6', color='c')
plt.plot(x6, y6, label='int 4-4-8', color='r')
plt.plot(x7, y7, label='int 6-6-4', color='y')
plt.plot(x4, y4, label='int 4-4-4', color='b')
plt.plot(x8, y8, label='int 8-8-4')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.title('Different input and weight quantization')
plt.legend()
plt.grid(True)
plt.axis([-2, 50, 10, 100])
plt.yticks(ticks)

plt.show()