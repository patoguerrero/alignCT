# alignCT
# Patricio Guerrero
# KU Leuven
# patrico.guerrero@kuleuven.be


import numpy
from scipy.interpolate import RegularGridInterpolator as RGI, RectBivariateSpline as RBS
from skimage.registration import phase_cross_correlation as pcc
import astra
from numpy.fft import fftshift as SHFT
import matplotlib.pyplot as plt


def simulate_foam_vol(size, size_y, seed, rmax, nballs):

    # simulate foam phantom with foam_ct_phantom
    
    import foam_ct_phantom as pht

    pht.FoamPhantom.generate('phantom.h5', seed, nballs, 10, rmax, 2)   #p_1
    
    phantom = pht.FoamPhantom('phantom.h5')
    geo = pht.VolumeGeometry(size, size, size_y, 3/size)
    phantom.generate_volume('geo.h5', geo)
    vol = pht.load_volume('geo.h5')

    return vol


def simulate_foam_fan(phantom, size, size_angles, shx):

    # simulate misalignrs fan-feam projections of a given phantom
    
    magn = 1
    src_ori = size * 2  # remember: src_ori > size 
    ori_det = src_ori * (magn-1)
    angles = numpy.flip(numpy.linspace(0, 2*numpy.pi, size_angles, endpoint = False))

    fan_geom = astra.create_proj_geom('fanflat', magn, size, angles, src_ori, ori_det)
    vol_geom = astra.create_vol_geom(size, size)
    proj_id = astra.creators.create_projector('line_fanflat', fan_geom, vol_geom)
    _, projs = astra.create_sino(phantom, proj_id)


    # misaligment
    projs = numpy.roll(projs, shx, axis = 1) 
    
    return projs, src_ori, src_ori+ori_det, (angles[0]-angles[1])*180/numpy.pi, magn




def vectors_astra(pixel, voxel, sod, sdd, rot_step, angles, det_x, det_y, eta, theta, phi):

    # eta theta phi in degrees
    
    pixel /= voxel 
    angles_seq = (numpy.arange(angles) * rot_step) * numpy.pi / 180 
    sod /= voxel
    sdd /= voxel
    odd = sdd - sod  
    mgn_i = sod / sdd
    
    det_x /= voxel
    det_y /= voxel 

    eta *= numpy.pi / 180 
    theta *= numpy.pi / 180 
    phi *= numpy.pi / 180

    sangles = numpy.sin(angles_seq)
    cangles = numpy.cos(angles_seq)

    # rotation matrices
    
    def ss(a):
        return numpy.sin(a)
    def cc(a):
        return numpy.cos(a)
    def rot_eta(a):
        return numpy.array([[1,0,0],[0,cc(a),-ss(a)],[0,ss(a),cc(a)]])
    def rot_theta(a):
        return numpy.array([[cc(a),0,-ss(a)],[0,1,0],[ss(a),0,cc(a)]])
    def rot_phi(a):
        return numpy.array([[cc(a),-ss(a),0],[ss(a),cc(a),0],[0,0,1]])

    u_shift = sangles * det_x
    v_shift = cangles * det_x

    rot_det = rot_theta(theta) @ rot_phi(phi) @ rot_eta(eta)

    det_u = rot_det @ numpy.array([0, pixel, 0])
    det_v = rot_det @ numpy.array([0, 0, pixel]) 

    vectors = numpy.zeros((angles, 12))
    # (source, detector center, det01, det10)
    
    vectors[:,0] = cangles * sod     
    vectors[:,1] = -sangles * sod     
    vectors[:,2] = 0
    vectors[:,3] = -cangles * odd + u_shift    
    vectors[:,4] = sangles * odd + v_shift
    vectors[:,5] = det_y
    vectors[:,6] = cangles * det_u[0] + sangles * det_u[1]
    vectors[:,7] = -sangles * det_u[0] + cangles * det_u[1]
    vectors[:,8] = det_u[2]
    vectors[:,9] = cangles * det_v[0] + sangles * det_v[1]
    vectors[:,10] = -sangles * det_v[0] + cangles * det_v[1]
    vectors[:,11] = det_v[2]

    return vectors


def geometry_astra(projs, vectors, det_len, slices):

    proj_geom = astra.create_proj_geom('cone_vec', det_len, det_len, vectors);
    proj_id = astra.data3d.create('-proj3d', proj_geom, projs)
    vol_geom = astra.create_vol_geom(det_len, det_len, slices)  
    vol_id = astra.data3d.create('-vol', vol_geom)

    return proj_id, vol_id


def reconstruct_astra(proj_id, vol_id):

    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = vol_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    astra.algorithm.delete(alg_id)

    return astra.data3d.get(vol_id) 



def fanrec_astra(g, shift_x, sod, sdd, pixel, voxel, rot_step):

    # fanbeam astra gives noisy recons
    
    if len(g.shape) == 3:
        g = g[g.shape[0]//2 - 1]*0 + g[g.shape[0]//2]

    angles, pixels = g.shape
    pixel /= voxel 
    sod /= voxel
    sdd /= voxel
    odd = sdd - sod
    magn_i = sod / sdd
    beta = numpy.arange(angles) * rot_step * numpy.pi / 180
        
 
    #s_M = (pixels * pixel * magn_i - 1) * 0.5
    #gamma_M = numpy.arctan(s_M / sod)
    #shift_x *= magn_i
    

    slice_geom = astra.create_vol_geom(pixels, pixels)
    slice_id = astra.data2d.create('-vol', slice_geom)
    #sino_geom = astra.create_proj_geom('fanflat', pixel, pixels, beta, sod, sdd-sod)


    sangles = numpy.sin(beta)
    cangles = numpy.cos(beta)
    det_x = -shift_x / voxel

    u_shift = sangles * det_x
    v_shift = cangles * det_x


    vectors = numpy.zeros((angles, 6))
    # (source, detector center, det01)

    vectors[:,0] = cangles * sod     
    vectors[:,1] = -sangles * sod     
    vectors[:,2] = -cangles * odd + u_shift    
    vectors[:,3] = sangles * odd + v_shift
    vectors[:,4] = sangles * pixel
    vectors[:,5] = cangles * pixel
    
    
    sino_geom = astra.create_proj_geom('fanflat_vec', pixels, vectors)
    sino_id = astra.data2d.create('-sino', sino_geom, g)

    x = numpy.linspace(-1, 1, pixels)
    xx, yy = numpy.meshgrid(x, x)
    mask = xx*xx + yy*yy < 0.95
    
    cfg2d = astra.astra_dict('FBP_CUDA')
    cfg2d['ReconstructionDataId'] = slice_id
    cfg2d['ProjectionDataId'] = sino_id
    cfg2d['option'] = {}
    cfg2d['option']['PixelSuperSampling'] = 4
    alg_id = astra.algorithm.create(cfg2d)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(slice_id) * mask
    rec = numpy.maximum(rec,0) 

    # Clean up
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(slice_id)
    astra.data2d.delete(sino_id)

    return rec








#               -----------------------------------
#                                         shifts 2D
#               -----------------------------------








def shift2D_FP(g, sod, sdd, pixel, rot_step, b0, shift_0 = 0, iters = 5):

    # x shift of the center of rotation in mm
    # based on fanbeam symmetry relantionship
    # g can be fan or conebeam

    if len(g.shape) == 3:
        g = g[g.shape[0]//2 - 1]*0 + g[g.shape[0]//2]

    angles, columns = g.shape
    
    magn_i = sod / sdd
    s_M = (columns * pixel * magn_i - 1) * 0.5
    beta = numpy.arange(angles) * rot_step * numpy.pi / 180
    s = numpy.linspace(-s_M, s_M, columns)

    interpolator = RBS(beta, s, g, kx = 1, ky = 1)    
    
    upsampling = 1 / 0.01  # for image registration, 1 / pixels
    pad = columns // 2 * int(upsampling-1)
    f1 = numpy.fft.rfft(g[b0]).conjugate()

    twopi = 2 * numpy.pi
    for i in numpy.arange(iters):

        beta_s = 2*numpy.arctan2(s-shift_0*magn_i, sod) + numpy.pi + beta[b0]
        mask = beta_s > twopi
        beta_s[mask] -= twopi
        f2 = interpolator(beta_s, -s+2*shift_0*magn_i, grid = False)
        
        xcorr = numpy.fft.irfft(numpy.pad((numpy.fft.rfft(f2) * f1),(0,pad)))
        shift_0 -= (numpy.argmax(SHFT(xcorr)) - columns*upsampling/2) / upsampling*pixel*0.5
        #diff = temp - shift_0


    return shift_0  # this is in mm


def shifts2D_FPK(g, sod, sdd, pixel, rot_step, shift_0 = 0, iters = 5, K = 10):

    angles, _  = g.shape
    a_samp = numpy.arange(K) * angles//K
    shift_fp = numpy.zeros(K)
    for a in numpy.arange(K):
        shift_fp[a] += shift2D_FP(g,sod,sdd,pixel,rot_step,b0=a_samp[a],shift_0=0,iters=5)

    return shift_fp

def shift2D_2DR(g, sod, sdd, pixel, rot_step):

    # x shift of the center of rotation in mm
    # based on fanbeam symmetry relantionship
    # g can be fan or conebeam

    if len(g.shape) == 3:
        g = g[g.shape[0]//2 - 1]*0 + g[g.shape[0]//2]

    angles, columns = g.shape
    
    magn_i = sod / sdd
    s_M = (columns * pixel * magn_i - 1) * 0.5
    beta = numpy.arange(angles) * rot_step * numpy.pi / 180
    s = numpy.linspace(-s_M, s_M, columns)# - shift_0 * magn_i

    interpolator = RBS(beta, s, g, kx = 1, ky = 1)
  
    upsampling = 1 / 0.01  # for image registration, 1 / pixels

    twopi = 2 * numpy.pi
    f2 = numpy.zeros((angles, columns))
    sh_0 = 2*numpy.arctan2(s, sod) + numpy.pi
        
    for bb in numpy.arange(angles):
        beta_s = sh_0 + beta[bb]
        mask = beta_s > twopi
        beta_s[mask] -= twopi
        f2[bb,:] = interpolator(beta_s, -s , grid = False)


    shifts = pcc(g, f2, upsample_factor = upsampling, normalization = None)[0] * pixel * 0.5       

    return shifts[1]  # this is in mm






def shift2D_LY(g, sod, sdd, pixel, rot_step):

    # yang method with linear interpolation
    # yang is actualy nearest neighbor 


    if len(g.shape) == 3:
        g = g[g.shape[0]//2 - 1]*0 + g[g.shape[0]//2]

    angles, columns = g.shape
    
    magn_i = sod / sdd
    s_M = (columns * pixel * magn_i - 1) * 0.5
    beta = numpy.arange(angles) * rot_step * numpy.pi / 180
    s = numpy.linspace(-s_M, s_M, columns)# - shift_0 * magn_i

    interpolator = RBS(beta, s, g, kx = 1, ky = 1)
    
    
    upsampling = 1 / 0.01  # for image registration, 1 / pixels
    pad = columns // 2 * int(upsampling-1)

    f1 = numpy.fft.rfft(g.sum(axis = 0)).conjugate()

    twopi = 2 * numpy.pi

    f2 = numpy.zeros(columns)
    sh_0 = 2*numpy.arctan2(s, sod) + numpy.pi
      
    for bb in numpy.arange(angles):
        beta_s = sh_0 + beta[bb]
        mask = beta_s > twopi
        beta_s[mask] -= twopi
        f2 += interpolator(beta_s, -s , grid = False)
            
    xcorr = numpy.fft.irfft(numpy.pad((numpy.fft.rfft(f2) * f1),(0,pad)))
    
    return -(numpy.argmax(SHFT(xcorr)) - columns*upsampling/2) / upsampling*pixel*0.5








def shift2D_yang(g, pixel):

    if len(g.shape) == 3:
        g = g[g.shape[0]//2 - 1] + g[g.shape[0]//2]

    f1 = g.sum(axis = 0)
    f2 = numpy.flip(f1)
    
    return pcc(f1, f2, upsample_factor = 1/0.01, normalization = None)[0][0] * pixel * 0.5










#           ----------------------------  
#           ---------------------------- end shifts 2D











def loss_fan(g, sod, sdd, pixel, rot_step, shift_x, fig):


    if len(g.shape) == 3:
        g = g[g.shape[0]//2 - 1]*0 + g[g.shape[0]//2]

    angles, pixels = g.shape
    
    #f1 = fan[0]
       
    magn_i = sod / sdd
        
    s_M = (pixels * pixel * magn_i - 1) * 0.5 
    s = numpy.linspace(-s_M, s_M, pixels) 
    sh = shift_x * magn_i
    beta = numpy.arange(angles) * rot_step * numpy.pi / 180

    interpolator = RBS(beta, s, g, kx = 1, ky = 1)
    #f2 = interpolator(2*numpy.arctan2(s-sh, sod) + numpy.pi, -s+2*sh, grid = False)

    twopi = 2 * numpy.pi
    f2 = numpy.zeros((angles, pixels))
    sh_0 = 2*numpy.arctan2(s-sh, sod) + numpy.pi
        
    for bb in numpy.arange(angles):
        beta_s = sh_0 + beta[bb]
        mask = beta_s > twopi
        beta_s[mask] -= twopi
        f2[bb,:] = interpolator(beta_s, -s+2*sh , grid = False)


    f = g - f2
    f_norm = 100 * (f*f).sum() / (g*g).sum()  # / pixels / angles 

    #plt.figure(fig); plt.plot(g[0]); plt.plot(f2[0])
    #print('shift, norm', numpy.around(shift_x), f_norm)

    return f_norm #f, f_norm





def loss_cone_tilt(interp3d, interp2d, sod, sdd, s, shx, shy, tilt, b0):

    magn_i = sod / sdd
    stilt = numpy.sin(tilt)
    ctilt = numpy.cos(tilt)

    #shx = shift_x * magn_i
    #shy = shift_y * magn_i
  
    f1 = interp2d((-(s) * stilt, (s) * ctilt ))

    beta_s = 2*numpy.arctan2(s-shx, sod) + numpy.pi + b0
    mask = beta_s > 2*numpy.pi
    beta_s[mask] -= 2*numpy.pi
    f2 = interp3d(((s-2*shx) * stilt, beta_s, (-s+2*shx) * ctilt))  

    # this is per slice, no interpolation
    #interpolator = interpolate.RectBivariateSpline(beta, s-shx, cone[shift_y], kx = 3, ky = 3)
    #f1 = interpolator(0, s-shx, grid = False)
    #f2 = interpolator(2*numpy.arctan2(s-shx, sod) + numpy.pi, -s+shx, grid = False)

    f = f1 - f2
    norm = (f*f).sum() * (s[1]-s[0])  #* 1E6 / (f1*f1).sum()

    #print(shift_x, norm)
    #plt.plot(f1)
    #plt.plot(f2)
    #plt.show(); exit()
    
    '''upsampling = 10
    ls = len(s)
    pad = ls // 2 * int(upsampling-1)
    f1 = numpy.fft.rfft(f1).conjugate()
    xcorr = numpy.fft.irfft(numpy.pad((numpy.fft.rfft(f2) * f1),(0,pad)))
    shift = (numpy.argmax(numpy.fft.fftshift(xcorr)) - ls*upsampling/2) #/ upsampling #* pixel
    shift *= shift'''
    
    #print('shift', shift, shift_x, shift_y, tilt)
    #print('norm', norm, shift_x, shift_y, tilt) ; exit()
              
    return norm  #f, norm




def cone_sampling(cone, sod, sdd, pixel, rot_step):
    
    cols, angles, pixels = cone.shape
    magn_i = sod / sdd

    cs = cols // 4  #cropped columns
    c0 = cols // 2 - cs // 2 
    
    s_M = (pixels * pixel * magn_i - 1) * 0.5
    s_c = (cs * pixel * magn_i - 1) * 0.5
    beta = numpy.arange(angles) * rot_step * numpy.pi / 180

    s0 = numpy.linspace(-s_M, s_M, pixels)
    sc = numpy.linspace(-s_c, s_c, cs)

    grid3d = (sc, beta, s0)
    grid2d = (sc, s0)

    return (cols, angles, pixels, magn_i, c0, cs, beta, s0, sc, grid2d, grid3d)



    
def cone_interpolator2D_tilt(cone, sampling, order, beta_0):

    # order: 'cubic' or 'linear'
    cols, angles, pixels, magn_i, c0, cs, beta, s0, sc, grid2d, _ = sampling    
    beta_0_val = beta[beta_0]
    if cone.shape[0] == pixels : cone = cone[c0:c0+cs]

    return RGI(grid2d, cone[:,beta_0], method=order, bounds_error=False, fill_value=0), beta_0_val



def cone_interpolator3D_tilt(cone, sampling, order):

    # order: 'cubic' or 'linear'
    cols, angles, pixels, magn_i, c0, cs, beta, s0, sc, _, grid3d = sampling
    if cone.shape[0] == pixels : cone = cone[c0:c0+cs]

    return RGI(grid3d, cone, method=order, bounds_error=False, fill_value=None)
    





def shift_x_tiltedfan(interp3d, interp2d, sod, sdd, s, pixel, tilt, b0, shx = 0, iters = 5):

    # x shift of the center of rotation in mm
    # based on fanbeam symmetry relantionship
    # on a tilted fan


    magn_i = sod / sdd
    stilt = numpy.sin(tilt)
    ctilt = numpy.cos(tilt)
    columns = len(s)

    upsampling = 1 / 0.1  # for image registration, 1 / pixels
    pad = columns // 2 * int(upsampling-1)

    f1 = interp2d((-s * stilt, s * ctilt ))
    f1 = numpy.fft.rfft( f1 ).conjugate()
    k1 = pixel * magn_i * 0.5 / upsampling
    k2 = columns * upsampling * 0.5
    nb0 = numpy.pi + b0
    twopi = 2*numpy.pi


    for i in numpy.arange(iters):

        #print(i, shx)
        beta_s = 2*numpy.arctan2(s-shx, sod) + nb0
        mask = beta_s > twopi
        beta_s[mask] -= twopi
        f2 = interp3d(((s-2*shx) * stilt, beta_s, (-s+2*shx) * ctilt))       
        xcorr = numpy.fft.irfft(numpy.pad((numpy.fft.rfft(f2) * f1),(0,pad)))
        shx -= (numpy.argmax(numpy.fft.fftshift(xcorr)) - k2) * k1

    return shx  # scaled to the origin (in mm)



def shift_x_fpK(data, interp3d, sod, sdd, sampling, pixel, tilt, shx0 = 0, iters = 5):

    s = sampling[7]
    angles = sampling[1]
    angs = 10
    a_samp = numpy.arange(angs) * angles//angs
    shx = numpy.zeros(angs)
    for a in numpy.arange(angs):
        interp2d, b0 = cone_interpolator2D_tilt(data, sampling, 'linear', a_samp[a])
        shx[a] = shift_x_tiltedfan(interp3d,interp2d,sod,sdd,s,pixel,tilt,b0,shx0,iters=iters)
    return shx



def shift_x_2d3d(data, interp3d, sod, magn_i, sampling, pixel, tilt):


    s = sampling[7]
    angles = sampling[1]
    cols = sampling[0]
    beta = sampling[6]
    stilt = numpy.sin(tilt)
    ctilt = numpy.cos(tilt)
    stilt_s = s * stilt
    ctilt_s = s * ctilt

    upsampling = 1 / 0.1
    sh_0 = 2*numpy.arctan2(s, sod) + numpy.pi

    twopi = 2 * numpy.pi
    f1 = numpy.zeros((angles, cols))
    f2 = numpy.zeros((angles, cols))
        
    for b in numpy.arange(angles):

        interp2d, b0 = cone_interpolator2D_tilt(data, sampling, 'linear', b)
        f1[b] = interp2d((-stilt_s, ctilt_s))

        beta_s = sh_0 + b0
        mask = beta_s > twopi
        beta_s[mask] -= twopi
        
        f2[b] = interp3d((stilt_s, beta_s, -ctilt_s))
                
    #plt.figure(2); plt.imshow(f2); plt.show(); exit()
    #plt.figure(12); plt.plot(g[0]); plt.plot(f2[0])

    shifts = pcc(f1, f2, upsample_factor = upsampling, normalization = None)[0] * pixel * 0.5 * magn_i      

    return shifts[1], f1  # this is in mm





def onlyloss(data, gmax, sod, pixel, rot_step, sampling, interp3d, tilt, shiftx, sdd):

     
    cols, angles, pixels, magn_i, c0, cs, beta, s, sc, grid2d, _ = sampling
    
    twopi = 2*numpy.pi
    shiftx *= magn_i


    shx = shift_x_fpK(data, interp3d, sod, sdd, sampling, pixel, tilt, shiftx, iters = 5)

    #shx0 = shx[0]
    #shx = numpy.median(shx)
    
    shx, f1s = shift_x_2d3d(data, interp3d, sod, magn_i, sampling, pixel, tilt)

    #print(shx, ss)
    
    fs = numpy.zeros((angles, cols))

    shx_stilt = ((s-2*shx) * numpy.sin(tilt))
    shx_ctilt = ((s-2*shx) * numpy.cos(tilt))
    
    beta0 = 2*numpy.arctan2(s-shx, sod) + numpy.pi

    norm = 0 
    for b in numpy.arange(angles): # on beta, 

        f1 = f1s[b]
        b0 = beta[b]
        
        beta_s = beta0 + b0
        mask = beta_s > twopi
        beta_s[mask] -= twopi

        f2 = interp3d((shx_stilt, beta_s, -shx_ctilt))  
        fs[b] = f1 - f2

        norm += (fs[b]*fs[b]).sum()

    norm *= rot_step / 180 * numpy.pi # * gmax

    #print('tilt, norm', numpy.around(tilt[0],5), numpy.around(norm,5))
    return norm



def tilt_GD(data, sod, sdd, pixel, rot_step, shiftx):



    sampling = cone_sampling(data, sod, sdd, pixel, rot_step)    

    c0 = sampling[4]
    cs = sampling[5]
    data = data[c0:c0+cs].astype('float')
    gmax = data.max()
    data /= gmax
    
    interp3d = cone_interpolator3D_tilt(data, sampling, 'linear')
    
    n_GD = 5
    n_LS = 20
    eta = 0    
    c = 0.01
    gamma = 1e-4  # radians
    bound = 0.02 # radians

    for k in numpy.arange(n_GD):  # GD
        
        lossGD = onlyloss(data, gmax, sod, pixel, rot_step, sampling, interp3d, eta, shiftx, sdd)
        lossFD = onlyloss(data, gmax, sod, pixel, rot_step, sampling, interp3d, eta-1e-3, shiftx, sdd)
        lossLS = lossGD + 0
        grad = (lossGD-lossFD)*1e3
        #print(grad, eta)
     
        if abs(grad) * gamma >= bound : gamma = bound / abs(grad)  # bound constraint
        
        etaGD = eta + 0
        for l in numpy.arange(n_LS):  # backtracking LS

            gg = grad * gamma
            if abs(gg) < 1e-4 :
                print('lossLS', lossLS, 'grad', grad)
                return eta
            
            eta -= gg
            lossLS = onlyloss(data, gmax, sod, pixel, rot_step, sampling, interp3d, eta, shiftx, sdd)
            #print(eta*180/numpy.pi, lossLS, lossGD - c*gg**2)

            if lossLS <= lossGD - c*gg*grad :
                print('break', l)
                break

            if l+1 == n_LS :
                print('lossGD', lossGD, 'grad', grad)
                return eta
            
            eta = etaGD + 0
            gamma *= 0.5

        #print('break', n_LS)
        diff = abs(etaGD - eta)
        print('diff', diff)
        if diff < 1e-4: break 

    print('lossLS', lossLS, 'grad', grad)

    return eta # radians

