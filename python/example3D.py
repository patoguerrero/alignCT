# example of alignCT with (simulated) cone-beam (3D) data
# Patricio Guerrero
# KU Leuven
# patrico.guerrero@kuleuven.be


import alignct
import numpy
import matplotlib.pyplot as plt
from time import time
import gc
import astra


t0 = time()
print('example of alignCT with cone-beam data')

# algorithm for projected problem (on h),
# options: 'FPK' (fixed point_K), '2DR' (2D sinogram registration) 
alg = 'FPK'
print('inner algorithm for h:', alg)

# ----------------
# 1. simulate data
# ----------------

# simulate 3D phantom
size = 1024
phantom = alignct.simulate_foam_vol(size, size, 1, 0.2, 500)
plt.figure(0); plt.imshow(phantom[0]); plt.title('phantom, slice 0')

## simulated misalignment values
# h_opt: in pixels, the horizontal shift value of the detector to be estimated
# eta_opt: in degrees, the rotation of the detector (\eta) to be estimated 

h_opt = 10
eta_opt = 1
print('h* (pix):', h_opt)
print('eta* (deg):', eta_opt)

voxel = 1
data, sod, sdd, rot_step, pixel = alignct.simulate_foam(phantom, size, size, size, shx=h_opt, shy=0, tilt=eta_opt)
rows, angles, cols = data.shape


print('data', data.shape, data.dtype)
print('time simulate', time()-t0)
print('############')


plt.figure(1); plt.imshow(data[:,0]); plt.title('projection 0')



# -------------
# 2. shifts 2D (h) (initial values)
# -------------



t2d = time()
from numpy import around as rd

shift_2d = alignct.shift2D_2DR(data, sod, sdd, pixel, rot_step)
print('initial shift', rd(shift_2d / pixel,3))

print('time initial 2d shift', time() - t2d)
print('############')

    

# -----------
# 3. eta estimation
# -----------



ttilt = time()
tilt_VP = alignct.tilt_VP(data, sod, sdd, pixel, rot_step, shift_2d, alg)
print('eta VP (deg)', tilt_VP * 180 / numpy.pi)
print('time VP', time() - ttilt)
print('############')

   
    
    
# -----------
# 4. final h estimation (from obtained eta)
# -----------




tshift3d = time()

sampling = alignct.cone_sampling(data, sod, sdd, pixel, rot_step)
interp2d, _ = alignct.cone_interpolator2D_tilt(data, sampling, 'linear', 0) 
interp3d = alignct.cone_interpolator3D_tilt(data, sampling, 'linear')

if alg == 'FPK':
    shift_h, _ = alignct.shift3D_FPK(data,interp3d,sod,sdd,sampling,pixel,tilt_VP,shift_2d*sod/sdd)
    shift_h = numpy.median(shift_h)*sdd/sod

if alg == '2DR':
    shift_h, _ = alignct.shift3D_2DR(data, interp3d, sod, sod/sdd, sampling, pixel, tilt_VP)
    shift_h *= sdd/sod

print('final shift h', rd(shift_h / pixel,3))

print('time shift h', time() - tshift3d)
print('############')


#plt.show(); exit()




# -----------------
# 5. astra geometry
# -----------------




tgeo = time()

slices = cols +0

det_x = -shift_h
det_y = 0
eta = tilt_VP * 180 / numpy.pi  
print('x(mm), eta(deg) ', det_x, eta)


vectors = alignct.vectors_astra(pixel, voxel, sod, sdd, rot_step, angles, det_x, det_y, eta, 0, 0)
proj_id, vol_id = alignct.geometry_astra(data, vectors, rows, slices)

print('time geom', time() - tgeo)



# ------------
# 4. astra fdk
# ------------



tfdk = time()

gc.collect()

x = numpy.linspace(-1, 1, rows)
xx, yy = numpy.meshgrid(x, x)
mask = xx*xx + yy*yy < 0.95

rec = alignct.reconstruct_astra(proj_id, vol_id) * mask 
astra.data3d.delete(proj_id)
astra.data3d.delete(vol_id)

rec = numpy.maximum(rec, 0)


print('rec', rec.shape, rec.dtype)
print('time fdk', time() - tfdk)


xx = rec[slices//2] 
yy = rec[:, cols//2]
zz = rec[:, :, cols//2]  


plt.figure(10); plt.imshow(xx); plt.title('FDK slice y=0')
plt.figure(11); plt.imshow(yy); plt.title('FDK x=0')
plt.figure(12); plt.imshow(zz); plt.title('FKD z=0')


plt.show()
