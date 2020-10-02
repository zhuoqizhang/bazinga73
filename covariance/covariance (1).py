import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pyccl as ccl
import sacc
import sys
import pymaster as nmt
import scipy
sys.path.insert(0, '/global/u1/z/zhzhuoqi/tjpcov/TJPCov')
from tjpcov import wigner_transform, bin_cov, parse

d2r=np.pi/180


#this is the cosmology I use. 
cosmo = ccl.Cosmology(Omega_k=0.0, Omega_g=0.0, w0=-1.0, wa=0.0, T_CMB=2.7,
                      Neff=0, m_nu=0.0, transfer_function='bbks', mass_function='tinker',
                      Omega_b=0.045, Omega_c=0.21, h=0.71, sigma8=0.80, n_s=0.964)


twopoint_data = sacc.Sacc.load_fits('/global/u1/z/zhzhuoqi/5x2/cov/cov/covariance.fits')


# set up ell for tjp and nmt
nside = 1024
fsky = 445./(4*np.pi*((180.**2)/(np.pi**2)))
ell = np.concatenate((
        np.linspace(2, 49, 48),
        np.logspace(np.log10(50), np.log10(6e4), 200)))
mask_tjp = ell>nside*3
ell_tjp = ell[mask_tjp]
ell_nmt = np.linspace(0, 3*nside-1, 3*nside)
ell1 = ell[np.invert(mask_tjp)]


#get cmb noise
cmb_noise = np.loadtxt('full_noise.txt')
cmb_noise_nmt = np.interp(ell_nmt, np.arange(len(cmb_noise)), cmb_noise)
cmb_noise_tjp = np.interp(ell_tjp, np.arange(len(cmb_noise)), cmb_noise)


#The spin based factor to decide the wigner transform. Based on spin of tracers. Sometimes we may use s1_s2 to denote these factors
WT_factors={}
WT_factors['ck','wl']=(0,2)
WT_factors['wl','ck']=(2,0)
WT_factors['gc','ck']=(0,0)
WT_factors['ck','gc']=(0,0)



# Wigner Transform setup... 
#setup theta array and theta binning for wigner transform 
th_min=1/60 # in degrees
th_max=300./60
n_th_bins=20
th_bins=np.logspace(np.log10(2.5/60.),np.log10(250./60.),n_th_bins+1)

th=np.logspace(np.log10(th_min),np.log10(th_max),3000) #covariance is oversampled at th values and then binned.
th2=np.linspace(1,th_max*1.02,3000) 
th=np.unique(np.sort(np.append(th,th2)))
thb=0.5*(th_bins[1:]+th_bins[:-1])

WT_kwargs={'l': ell,'theta': th*d2r,'s1_s2':[(0,2),(2,0),(0,0)]}
WT=wigner_transform(**WT_kwargs)


#namaster setup...
print("Workspace")
w = nmt.NmtWorkspace()
w.read_from('w1024.fits')

print("Covariance")
cw = nmt.NmtCovarianceWorkspace()
cw.read_from('cw1024.fits')


def get_tracer_info(two_point_data={}):
    ccl_tracers={}
    nmt_noise={}
    tjp_noise={}
    for tracer in twopoint_data.tracers:
        
        if tracer != 'ck':
            tracer_dat=twopoint_data.get_tracer(tracer)
            z= tracer_dat.z
            Ngal = twopoint_data.metadata[tracer+'_Ngal'] #arc_min^2
            Ngal=Ngal*3600/d2r**2
            
            dNdz = tracer_dat.nz
            
        
            if 'wl' in tracer or 'wl' in tracer:  
                ccl_tracers[tracer]=ccl.WeakLensingTracer(cosmo, dndz=(z, dNdz)) #CCL automatically normalizes dNdz
                sigma_e= twopoint_data.metadata[tracer+'_sigma_e']
                nmt_noise[tracer]=sigma_e**2/Ngal
                tjp_noise[tracer]=sigma_e**2/Ngal
                
            elif 'gc' in tracer:
                bias = twopoint_data.metadata[tracer+'_bias']
                bias = bias*np.ones(len(z)) #Galaxy bias (constant with scale and z)
                nmt_noise[tracer]=1./Ngal
                tjp_noise[tracer]=1./Ngal
                ccl_tracers[tracer]=ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z,dNdz), bias=(z,bias))
                      
        else: 
            ccl_tracers[tracer]=ccl.CMBLensingTracer(cosmo,1100.)
            nmt_noise[tracer]=cmb_noise_nmt
            tjp_noise[tracer]=cmb_noise_tjp
    return ccl_tracers,nmt_noise,tjp_noise


def get_cov_WT_spin(tracer_comb=None):

    tracers=[]
    for i in tracer_comb:
        if 'ck' in i:
            tracers+=['ck']
        if 'wl' in i:
            tracers+=['wl']
        if 'gc' in i:
            tracers += ['gc'] 
    return WT_factors[tuple(tracers)]


def get_tjp_cov(tracer_comb1=None,tracer_comb2=None,ccl_tracers=None,tracer_Noise=None):  
    
    cl={}
    cl[13] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[0]], ell_tjp)
    cl[24] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[1]], ell_tjp)
    cl[14] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[1]], ell_tjp)
    cl[23] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[0]], ell_tjp)
    
    SN={}
    SN[13]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[0]  else 0
    SN[24]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[1]  else 0
    SN[14]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[1]  else 0
    SN[23]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[0]  else 0

    coupling_mat={}
    coupling_mat[1324]=np.eye(len(ell_tjp)) #placeholder
    coupling_mat[1423]=np.eye(len(ell_tjp)) #placeholder
    
    cov={}
    cov[1324]=np.outer(cl[13]+SN[13],cl[24]+SN[24])*coupling_mat[1324]
    cov[1423]=np.outer(cl[14]+SN[14],cl[23]+SN[23])*coupling_mat[1423]
    
    tjp_cov = cov[1423]+cov[1324]

    return tjp_cov


def get_nmt_cov(tracer_comb1=None,tracer_comb2=None,ccl_tracers=None,tracer_Noise=None):  

    cl={}
    cl[13] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[0]], ell_nmt)
    cl[24] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[1]], ell_nmt)
    cl[14] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[1]], ell_nmt)
    cl[23] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[0]], ell_nmt)
    
    SN={}
    SN[13]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[0]  else 0
    SN[24]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[1]  else 0
    SN[14]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[1]  else 0
    SN[23]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[0]  else 0
    
    nmt_cov = nmt.gaussian_covariance(cw, 0, 0, 0, 0, 
                                      [cl[13]+SN[13]],  
                                      [cl[14]+SN[14]],    
                                      [cl[23]+SN[23]],  
                                      [cl[24]+SN[24]],  
                                      wa=w, wb=w, coupled=True)
    
    nmt_cov *= 62.9/(fsky**2)
    norm_nmt = (2*ell_nmt+1)*np.gradient(ell_nmt)*fsky
    nmt_cov *= np.sqrt(np.outer(norm_nmt, norm_nmt))
    
    f = scipy.interpolate.interp2d(ell_nmt, ell_nmt, nmt_cov, kind='cubic')
    nmt_cov = f(ell1,ell1)
    

    return nmt_cov


def covariance(tracer_comb1=None,tracer_comb2=None,ccl_tracers=None,nmt_noise=None,tjp_noise=None):
    
    nmt_cov = get_nmt_cov(tracer_comb1,tracer_comb2,ccl_tracers,nmt_noise)
    tjp_cov = get_tjp_cov(tracer_comb1,tracer_comb2,ccl_tracers,tjp_noise)
    
    
    
    cov = np.zeros((len(ell),len(ell)))
    cov[:len(ell1),:len(ell1)] = nmt_cov
    cov[len(ell1):,len(ell1):] = tjp_cov
    
    norm=np.pi*4*fsky
    cov /= norm
    
    s1_s2_1=get_cov_WT_spin(tracer_comb=tracer_comb1)
    s1_s2_2=get_cov_WT_spin(tracer_comb=tracer_comb2)
    
    th, cov = WT.projected_covariance2(l_cl=ell,s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2, cl_cov=cov) 
    thb, cov = bin_cov(r=th/d2r,r_bins=th_bins,cov=cov) 
    return cov


def get_all_cov(two_point_data={}):
    
    #FIXME: Only input needed should be two_point_data, which is the sacc data file. Other parameters should be included within sacc and read from there.
    ccl_tracers,nmt_noise,tjp_noise=get_tracer_info(two_point_data=two_point_data)
    tracer_combs=two_point_data.get_tracer_combinations()# we will loop over all these
    N2pt=len(tracer_combs)

    Nell_bins=len(th_bins)-1
    cov_full=np.zeros((Nell_bins*N2pt,Nell_bins*N2pt))
    
    for i in np.arange(N2pt):
        print("{}/{}".format(i+1, N2pt))
        tracer_comb1=tracer_combs[i]
        indx_i=i*Nell_bins
        for j in np.arange(i,N2pt):
            tracer_comb2=tracer_combs[j]
            indx_j=j*Nell_bins
#covariance(tracer_comb1=None,tracer_comb2=None,ccl_tracers=None,nmt_noise=None,tjp_noise=None)
            cov_ij=covariance(tracer_comb1=tracer_comb1, tracer_comb2=tracer_comb2, ccl_tracers=ccl_tracers,
                                         nmt_noise=nmt_noise, tjp_noise=tjp_noise)

            cov_full[indx_i:indx_i+Nell_bins,indx_j:indx_j+Nell_bins]=cov_ij
            cov_full[indx_j:indx_j+Nell_bins,indx_i:indx_i+Nell_bins]=cov_ij.T
    return cov_full

print('starting cov')
cov = get_all_cov(twopoint_data)
np.savetxt('full_cov.txt',cov)

