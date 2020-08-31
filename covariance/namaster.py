import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pyccl as ccl
import sys
import sacc
import pymaster as nmt
sys.path.insert(0, '/global/u1/z/zhzhuoqi/tjpcov/TJPCov')
from tjpcov import wigner_transform, bin_cov, parse

d2r=np.pi/180

#this is the cosmology I use. 
cosmo = ccl.Cosmology(Omega_k=0.0, Omega_g=0.0, w0=-1.0, wa=0.0, T_CMB=2.7,
                      Neff=0, m_nu=0.0, transfer_function='bbks', mass_function='tinker',
                      Omega_b=0.045, Omega_c=0.21, h=0.71, sigma8=0.80, n_s=0.964)

# this sacc data has a number-count tracer and a weak-lensing tracer, both correspond
# to 0.8 < z < 1.0; and it has a cmb-lensing tracer
twopoint_data = sacc.Sacc.load_fits('covariance.fits')

#use the same ell and ell_bins as namaster
nside = 1024
ell = np.linspace(0, int(nside*3-1), int(nside*3))
ell_bins = np.linspace(0, int(nside*3-2), int(nside*3./32))

#setup theta array and theta binning for wigner transform 
th_min=1/60 # in degrees
th_max=300./60
n_th_bins=20
th_bins=np.logspace(np.log10(2.5/60.),np.log10(250./60.),n_th_bins+1)

th=np.logspace(np.log10(th_min),np.log10(th_max),3000) #covariance is oversampled at th values and then binned.
th2=np.linspace(1,th_max*1.02,3000) #binned covariance can be sensitive to the th values. Make sue you check convergence for your application
th=np.unique(np.sort(np.append(th,th2)))
thb=0.5*(th_bins[1:]+th_bins[:-1])

# Wigner Transform setup... 
WT_kwargs={'l': ell,'theta': th*d2r,'s1_s2':[(0,2),(2,0),(0,0)]}

# input basic info about the data
fsky=445./(4*np.pi*((180.**2)/(np.pi**2)))
Ngal = twopoint_data.metadata['gc4']
Ngal=Ngal*3600/d2r**2
sigma_e=twopoint_data.metadata['wl4']

#get noises
nc_noise = 1./Ngal
wl_noise = sigma_e**2/Ngal
cmb_noise_smth = np.loadtxt('full_noise.txt')[0:nside*3]

# generate ccl tracers for number-count and cmb-lensing from sacc data
nc_tracer = twopoint_data.get_tracer('gc4')
nc_z = nc_tracer.z
nc_nz = nc_tracer.nz

bias = twopoint_data.metadata['bias4']*np.ones(len(nc_z))
dNdz_nc = nc_nz
dNdz_nc/=(dNdz_nc*np.gradient(nc_z)).sum()
dNdz_nc*=Ngal

wl_tracer = twopoint_data.get_tracer('wl4')
wl_z = wl_tracer.z
wl_nz = wl_tracer.nz

dNdz_wl = wl_nz
dNdz_wl/=(dNdz_wl*np.gradient(wl_z)).sum()
dNdz_wl*=Ngal

nc_trcr = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(nc_z,dNdz_nc), bias=(nc_z,bias))
cmb_trcr = ccl.CMBLensingTracer(cosmo,1100.)
wl_trcr = ccl.WeakLensingTracer(cosmo, dndz=(wl_z,dNdz_wl))



#namaster setup
# a) Read and apodize mask
m = hp.read_map('../mask.fits')
#m = m*0+1
m = hp.pixelfunc.ud_grade(m, nside)
mask = nmt.mask_apodization(m, 1., apotype="Smooth")

cmb_map = hp.read_map('../cmb_masked.fits', verbose=False)
cmb_map = hp.pixelfunc.ud_grade(cmb_map, nside)
nc_map = hp.read_map('../data/nc_80-100.fits', verbose=False)
nc_map = hp.pixelfunc.ud_grade(nc_map, nside)
g1_map = hp.read_map('../data/g1_80-100.fits', verbose=False)
g1_map = hp.pixelfunc.ud_grade(g1_map, nside)
g2_map = hp.read_map('../data/g2_80-100.fits', verbose=False)
g2_map = hp.pixelfunc.ud_grade(g2_map, nside)
print('finihsed loading')

fN = nmt.NmtField(mask, [nc_map])
fG = nmt.NmtField(mask, [g1_map, g2_map])
fCMB = nmt.NmtField(mask, [cmb_map])

# generate bandpowers and weights based on ell 
bpws = np.digitize(ell, ell_bins)-1
bpws[-1] = -1


weights = []
for i in range(0, max(bpws)+2):
    count = np.count_nonzero(bpws == i)
    for j in range (0, count):
        weights.append(1./count)
weights.append(1)
weights = np.array(weights)

b = nmt.NmtBin(nside=nside, ells=ell, bpws=bpws, weights=weights)
#b = nmt.NmtBin.from_nside_linear(nside, 32)

print("Workspace")
w_nk = nmt.NmtWorkspace()
w_nk.compute_coupling_matrix(fN, fCMB, b)
w_gk = nmt.NmtWorkspace()
w_gk.compute_coupling_matrix(fG, fCMB, b)

print("Covariance")
# First we generate a NmtCovarianceWorkspace object to precompute
# and store the necessary coupling coefficients
cw = nmt.NmtCovarianceWorkspace()
# This is the time-consuming operation
# Note that you only need to do this once,
# regardless of spin
cw.compute_coupling_coefficients(fN, fN, fN, fN)



#note that C_ell here has to be longer than the ell array for cw
ell_cl = np.linspace(0, int(nside*3-1), int(nside*3))
cl_nn = ccl.angular_cl(cosmo, nc_trcr, nc_trcr, ell_cl)
cl_nk = ccl.angular_cl(cosmo, nc_trcr, cmb_trcr, ell_cl)
cl_kn = ccl.angular_cl(cosmo, cmb_trcr, nc_trcr, ell_cl)
cl_kk = ccl.angular_cl(cosmo, cmb_trcr, cmb_trcr, ell_cl)

# add noise
cl_nn = cl_nn + nc_noise
cl_kk = cl_kk + cmb_noise_smth


covar_00_00 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_nn],  # TT    cla1b1
                                      [cl_nk],  # TE, TB cla1b2   
                                      [cl_nk],  # ET, BT cla2b1
                                      [cl_kk],  # EE, EB, BE, BB cla2b2    
                                                       # "2" is a spin-2 field and has two components
                                      wa=w_nk, wb=w_nk, coupled=True)
print(np.shape(covar_00_00))
covar_00_00 = covar_00_00.reshape([len(ell), 1, len(ell), 1])
covar_TT_TT = covar_00_00[:, 0, :, 0]

np.savetxt('cl_NKNK44_nmt.txt',covar_TT_TT)




cl_gg = ccl.angular_cl(cosmo, wl_trcr, wl_trcr, ell_cl)
cl_gk = ccl.angular_cl(cosmo, wl_trcr, cmb_trcr, ell_cl)
cl_kg = ccl.angular_cl(cosmo, cmb_trcr, wl_trcr, ell_cl)
cl_kk = ccl.angular_cl(cosmo, cmb_trcr, cmb_trcr, ell_cl)

# add noise
cl_gg_ee = cl_gg + wl_noise
cl_gg_eb = 0*cl_gg
cl_gg_bb = 0*cl_gg + wl_noise
cl_gk_te = cl_gk
cl_gk_tb = 0*cl_gk
cl_kk_tt = cl_kk + cmb_noise_smth



covar_00_00 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  # Spins of the 4 fields
                                      [cl_gg + wl_noise],  # TT    cla1b1
                                      [cl_gk],  # TE, TB cla1b2   
                                      [cl_kg],  # ET, BT cla2b1
                                      [cl_kk + cmb_noise_smth],  # EE, EB, BE, BB cla2b2    
                                                       # "2" is a spin-2 field and has two components
                                      wa=w_nk, wb=w_nk, coupled=True)
print(np.shape(covar_00_00))
covar_00_00 = covar_00_00.reshape([len(ell), 1, len(ell), 1])
covar_TT_TT = covar_00_00[:, 0, :, 0]

np.savetxt('cl_GKGK44_nmt.txt',covar_TT_TT)

