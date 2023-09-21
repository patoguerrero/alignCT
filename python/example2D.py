# example of alignCT with (simulated) fan-beam (2D) data
# Patricio Guerrero
# KU Leuven
# patrico.guerrero@kuleuven.be


import alignct
import numpy
import matplotlib.pyplot as plt
from time import time


t0 = time()
print('example of alignCT with  fan-beam data')


# ----------------
# 1. simulate data
# ----------------

# simulate 2D phantom
size = 1024
foam = alignct.simulate_foam_vol(size, 1, 1, 0.2, 500)[0]
plt.figure(0); plt.imshow(foam) 

# simulated misalignment value
# h_opt: in pixels, the horizontal shift value of the detector to be estimated

h_opt = 10

# fan-beam configuration, all in mm

data, sod, sdd, rot_step, pixel = alignct.simulate_foam_fan(foam, size, size, h_opt)
voxel = 1
rows, angles = data.shape

print('data', data.shape, data.dtype)
print('time simulate', time()-t0)

plt.figure(1); plt.imshow(data) 


# -------------
# 2. shifts
# -------------

    
t2d = time()
from numpy import around as rd
print('h:', h_opt)

shift_yang = alignct.shift2D_yang(data, pixel)
print('shift yang', rd(shift_yang / pixel,3))

shift_yanglinear = alignct.shift2D_LY(data, sod, sdd, pixel, rot_step)
print('shift yanglinear', rd(shift_yanglinear / pixel,3))

shifts_fp = alignct.shifts2D_FPK(data,sod,sdd,pixel,rot_step,shift_0=0,iters=5,K=10)
shift_fp0 = shifts_fp[0]
shift_fpK = numpy.median(shifts_fp)
print('shift FP', rd(shift_fp0/pixel,3))
print('shift FPK', rd(shift_fpK/pixel,3))

shift_2d = alignct.shift2D_2DR(data, sod, sdd, pixel, rot_step)
print('shift 2DR', rd(shift_2d / pixel,3))


print('time shifts 2d', time() - t2d)
print('############')


# ------------------------
# 3. 2D FAN reconstruction
# ------------------------


trec = time()

recon2d = alignct.fanrec_astra(data, shift_fpK, sod, sdd, pixel, voxel, rot_step)

print('time recon 2D', time() - trec)

plt.figure(2); plt.imshow(recon2d)
plt.show()
