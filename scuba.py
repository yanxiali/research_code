import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.constants import c,h,k_B
c = c.value
h = h.value
kb = k_B.value


def region(ra, dec, name, r=2, color='red'):

    n = len(ra)
    text = np.empty([2+n*2], dtype=object)

    text[0] = ('global color='+color+' dashlist=8 3 width=1 font="helvetica 10 normal roman"'+
             ' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
    text[1] = 'fk5\n'

    for i in range(n):
        text[i*2+2] = 'circle('+str(ra[i])+','+str(dec[i])+','+str(r)+'")\n'
        text[1+i*2+2] = '# text('+str(ra[i])+','+str(dec[i]+(r+3.)/3600.)+')'+' text={'+str(i+1)+'}\n'

    target = open(name, 'w')
    target.writelines(text)
    target.close()


def doubleGauss(x, c1, sigma1, sigma2):

    c2 = c1 - 1.0
  
    return c1 * np.exp(-0.5*x**2/sigma1**2) - c2 * np.exp(-0.5*x**2/sigma2**2)


class scuba850(object):
  
    def __init__(self, image):
        """load the fits file"""
        data=fits.open(image)
        s1=data[0].data[0,:,:]  # flux  
        s2=data[1].data[0,:,:]  # variance
        s2 = np.sqrt(s2)        # noise (rms)
        s3 = s1/s2              # signal-to-noise ratio

        # save the header and WCS information
        header=data[0].header
        self.header = header
        self.WCS = wcs.WCS(header)
        
        # fill all the NANs with 0 (flux,snr) or 1000 (noise)
        NANs = np.isnan(s1)
        s1[NANs] = 0
        s2[NANs] = 1000
        s3[NANs] = 0

        self.flux = s1
        self.noise = s2
        self.snr = s3

    def mkrms(self, name='s8_rms.fits'):
        """write the rms map to a fits file"""
        fits.writeto(name, self.noise, header=self.header,overwrite=True)
        print 'central noise value (mJy/beam):', self.noise.min()
    

    def mkpsf(self, name='psf850.fits', level=3.0, thresh=7.0):
        """ make the PSF """
        print 'name:',name
        print 'level:',level    # define the area for source detection: noise <= level * noise.min()
        print 'thresh:',thresh  # detection threshold (snr)

        snr = self.snr.copy()
        noise = self.noise.copy()
        flux = self.flux.copy()

        # set snr = 0 for the outer area where source detection is not performed
        snr[noise > level*noise.min()]=0  

        stack = np.zeros((141,141),float)
        peak = peak_local_max(snr, threshold_abs=thresh) # find local maxima with snr >= threshold

        # stack these detected sources
        for i in range(len(peak)):
            stack += flux[peak[i,0]-70:peak[i,0]+70+1,peak[i,1]-70:peak[i,1]+70+1]
            
        stack = stack/stack.max() # normalize the stacked image
        print '# of sources:',len(peak)

        # compute the radial profile of the stacked image
        profile=np.zeros(71,float)
        radial=np.zeros(71,float)
        dist=np.zeros((141,141),float)

        for i in range(141):
            for j in range(141):
                dist[i,j]=np.sqrt( (i-70.)**2.0 + (j-70.)**2.0 )

        for i in range(71):
            filter = (dist >= -0.5+i) & (dist < 0.5+i)
            profile[i]=np.mean(stack[filter])
            radial[i]=np.mean(dist[filter])

        # fit a double-Gaussion model to the radial profile
        p_best, cov = curve_fit(doubleGauss, radial, profile, bounds=(0, [10., 15., 15.]))

        # make a plot
        fig = plt.figure()
        plt.scatter(radial,profile)
        plt.plot(radial, doubleGauss(radial, *p_best), 'r-')
        plt.xlabel('arcsec')
        plt.ylabel('normalized flux')
        plt.grid(True)
        plt.show(block=False)

        print('Best-fit parameters:')
        print(p_best)

        # use the best-fit double-Gaussion model to generate the PSF
        psf = np.zeros((141,141),float)

        for i in range(141):
            for j in range(141):
                psf[i,j]=doubleGauss(dist[i,j], *p_best)

        # write the stacked image and the PSF to fits files        
        fits.writeto('stack850.fits',stack,overwrite=True)        
        fits.writeto(name,psf,overwrite=True)

        self.psf = psf
        return psf

    
    def extract(self,name,level=3.0,thresh=4.0,blend=7.25,r_psf=50,psf=None):
        """run source extraction, producing a csv file, a region file, and residul images"""

        print 'name:',name
        print 'level:',level    # define the area for source detection: noise <= level * noise.min() 
        print 'thresh:',thresh  # detection threshold (snr)
        print 'blend:',blend    # ignore a detection if it's within this distance from a previous detection
        print 'r_psf:',r_psf    # the radius of the PSF we use
        
        if psf is None:
            print 'psf is from mkpsf'
            psf = self.psf
        else:
            print 'psf:',psf
            psf = fits.open(psf)[0].data  # read the PSF from an existing fits file

        snr = self.snr.copy()
        flux = self.flux.copy()
        noise = self.noise.copy()
        
        # set snr = 0 for the outer area where source detection is not performed
        # a slightly larger area is used here; will later remove any sources that are outside of the border we define
        snr[noise > (level+0.1)*noise.min()]=0

        # remove the outer radii of the PSF (defined by r_psf)
        ncol = psf.shape[1]
        center = (ncol-1)/2
        psf=psf[center-r_psf : center+r_psf+1,center-r_psf : center+r_psf+1]
        psfdist=np.zeros((r_psf*2+1,r_psf*2+1),float)

        for i in range(r_psf*2+1):
            for j in range(r_psf*2+1):
                psfdist[i,j]=np.sqrt( (i-r_psf)**2.0 + (j-r_psf)**2.0 )    

        psf[psfdist > r_psf]=0

        """
        source extraction:

        find the source with the highest snr, subtract a scaled PSF from the image at the source position,
        repeat the same process until there are no more sources with snr >= threshold in the image
       
        during the iterations, record the coordinate, flux, noise value, and snr of each detection

        """
        ra = np.array([])
        dec = np.array([])
        x = np.array([])
        y = np.array([])
        noiselevel = np.array([])
        flux_out = np.array([])
        snr_out = np.array([])

        i=0

        while snr.max() >= thresh:

            index = np.where(snr == snr.max())
            row = index[0][0]
            col = index[1][0]

            # compute the minimum among the distances between the current detection and previous detections
            if i > 0:
                mindist = np.sqrt( (col-x)**2.0 + (row-y)**2.0 ).min()
            else: mindist = 1000

            # ignore a detection if it's within "blend" from a previous detection
            if mindist <= blend:  
                snr[row,col]=0
            else:

                # record the flux, noise, and snr
                flux_detected = flux[row,col]
                flux_out = np.append(flux_out, round(flux_detected,6) )
                snr_out = np.append(snr_out, round(snr[row,col],6) )
                noiselevel = np.append(noiselevel, noise[row,col])

                # record the coordinate
                ra_temp,dec_temp,junk = self.WCS.wcs_pix2world(col,row,0,0)
                ra  = np.append(ra ,round(ra_temp,6) )
                dec  = np.append(dec ,round(dec_temp,6) )
                x = np.append(x, col)
                y = np.append(y, row)
               
                # subtract a scaled PSF from the image at the source position
                flux[row-r_psf :row+r_psf+1, col-r_psf :col+r_psf+1] -= flux_detected*psf
                snr[row-r_psf :row+r_psf+1, col-r_psf :col+r_psf+1] -= flux_detected * psf/noise[row-r_psf :row+r_psf+1, col-r_psf :col+r_psf+1]

                i=i+1

        # write the residual images to fits files        
        fits.writeto(name+'-snr-resid.fits',snr,header=self.header,overwrite=True)
        fits.writeto(name+'-resid.fits',flux,header=self.header,overwrite=True)

        # remove any detected sources that are outside of the border we define
        filter = noiselevel <= level*noise.min()
        ra=ra[filter]
        dec=dec[filter]
        flux_out=flux_out[filter]
        snr_out=snr_out[filter]
        
        # sort all the recorded parameters by the flux
        order = np.argsort(flux_out)
        ra=np.flipud(ra[order])
        dec=np.flipud(dec[order])
        snr_out = np.round(np.flipud(snr_out[order]),3)
        flux_out = np.round(np.flipud(flux_out[order]),3)
        err_out = np.round(flux_out/snr_out,3)

        # write the result to a csv file
        data = {'ID':range(1,len(ra)+1), 'ra':ra, 'dec':dec, 'flux(mJy/beam)':flux_out, 'error(mJy/beam)':err_out, 'S/N':snr_out}
        df=pd.DataFrame(data,columns=['ID', 'ra', 'dec','flux(mJy/beam)','error(mJy/beam)','S/N'])
        df.to_csv(name+'-source.log',index=False)

        # produce a region file for the detected sources
        region(ra,dec,name+'.reg',r=blend,color='red')

        print '# of sources detected above '+str(thresh)+' sigma:', len(ra)
        print 'central noise value (mJy/beam):', noise.min()
        return df

    

class scuba450(object):
  
    def __init__(self, image):
        """load the fits file"""
        data=fits.open(image)
        s1=data[0].data[0,:,:]   
        s2=data[1].data[0,:,:]  
        s2 = np.sqrt(s2)        
        s3 = s1/s2              

        header=data[0].header
        self.header = header
        self.WCS = wcs.WCS(header)
  
        NANs = np.isnan(s1)
        s1[NANs] = 0
        s2[NANs] = 1000
        s3[NANs] = 0

        self.flux = s1
        self.noise = s2
        self.snr = s3

    def mkrms(self, name='s4_rms.fits'):
        """write the rms map to a fits file"""
        fits.writeto(name, self.noise, header=self.header,overwrite=True)
        print 'central noise value (mJy/beam):', self.noise.min()
    

    def mkpsf(self, name='psf450.fits', level=3.0, thresh=4.0):
        """ make the PSF """
        print 'name:',name
        print 'level:',level
        print 'thresh:',thresh

        snr = self.snr.copy()
        noise = self.noise.copy()
        flux = self.flux.copy()
        
        snr[noise > level*noise.min()]=0

        stack = np.zeros((81,81),float)
        peak = peak_local_max(snr, threshold_abs=thresh)

        for i in range(len(peak)):
            stack += flux[peak[i,0]-40:peak[i,0]+40+1,peak[i,1]-40:peak[i,1]+40+1]

        stack = stack/stack.max()
        print '# of sources:',len(peak)

        profile=np.zeros(41,float)
        radial=np.zeros(41,float)
        dist=np.zeros((81,81),float)

        for i in range(81):
            for j in range(81):
                dist[i,j]=np.sqrt( (i-40.)**2.0 + (j-40.)**2.0 )

        for i in range(41):
            filter = (dist >= -0.5+i) & (dist < 0.5+i)
            profile[i]=np.mean(stack[filter])
            radial[i]=np.mean(dist[filter])


        p_best, cov = curve_fit(doubleGauss, radial, profile, bounds=(0, [10., 15., 15.]))

        fig = plt.figure()
        plt.scatter(radial,profile)
        plt.plot(radial, doubleGauss(radial, *p_best), 'r-')
        plt.xlabel('arcsec')
        plt.ylabel('normalized flux')
        plt.grid(True)
        plt.show(block=False)

        print('Best-fit parameters:')
        print(p_best)

        psf = np.zeros((81,81),float)

        for i in range(81):
            for j in range(81):
                psf[i,j]=doubleGauss(dist[i,j], *p_best)

        fits.writeto('stack450.fits',stack,overwrite=True)             
        fits.writeto(name,psf,overwrite=True)

        self.psf = psf
        return psf

    def extract(self,name,level=3.0,thresh=4.0,blend=3.75,r_psf=30,psf=None):
        """run source extraction, producing a csv file, a region file, and residul images"""

        print 'name:',name
        print 'level:',level
        print 'thresh:',thresh
        print 'blend:',blend
        print 'r_psf:',r_psf
        
        if psf is None:
            print 'psf is from mkpsf'
            psf = self.psf
        else:
            print 'psf:',psf
            psf = fits.open(psf)[0].data

        snr = self.snr.copy()
        flux = self.flux.copy()
        noise = self.noise.copy()

        snr[noise > (level+0.1)*noise.min()]=0

        ncol = psf.shape[1]
        center = (ncol-1)/2
        psf=psf[center-r_psf : center+r_psf+1,center-r_psf : center+r_psf+1]
        psfdist=np.zeros((r_psf*2+1,r_psf*2+1),float)

        for i in range(r_psf*2+1):
            for j in range(r_psf*2+1):
                psfdist[i,j]=np.sqrt( (i-r_psf)**2.0 + (j-r_psf)**2.0 )    

        psf[psfdist > r_psf]=0

        ra = np.array([])
        dec = np.array([])
        x = np.array([])
        y = np.array([])
        noiselevel = np.array([])
        flux_out = np.array([])
        snr_out = np.array([])

        i=0

        while snr.max() >= thresh:

            index = np.where(snr == snr.max())
            row = index[0][0]
            col = index[1][0]

            if i > 0:
                mindist = np.sqrt( (col-x)**2.0 + (row-y)**2.0 ).min()
            else: mindist = 1000

            if mindist <= blend:
                snr[row,col]=0
            else:
          
                flux_detected = flux[row,col]
                flux_out = np.append(flux_out, round(flux_detected,6) )
                snr_out = np.append(snr_out, round(snr[row,col],6) )
                noiselevel = np.append(noiselevel, noise[row,col])

                ra_temp,dec_temp,junk = self.WCS.wcs_pix2world(col,row,0,0)
                ra  = np.append(ra ,round(ra_temp,6) )
                dec  = np.append(dec ,round(dec_temp,6) )
                x = np.append(x, col)
                y = np.append(y, row)
       
                flux[row-r_psf :row+r_psf+1, col-r_psf :col+r_psf+1] -= flux_detected*psf
                snr[row-r_psf :row+r_psf+1, col-r_psf :col+r_psf+1] -= flux_detected * psf/noise[row-r_psf :row+r_psf+1, col-r_psf :col+r_psf+1]

                i=i+1
            
        fits.writeto(name+'-snr-resid.fits',snr,header=self.header,overwrite=True)
        fits.writeto(name+'-resid.fits',flux,header=self.header,overwrite=True)

        
        filter = noiselevel <= level*noise.min()
        ra=ra[filter]
        dec=dec[filter]
        flux_out=flux_out[filter]
        snr_out=snr_out[filter]
        
        order = np.argsort(flux_out)
        ra=np.flipud(ra[order])
        dec=np.flipud(dec[order])
        snr_out = np.round(np.flipud(snr_out[order]),3)
        flux_out = np.round(np.flipud(flux_out[order]),3)
        err_out = np.round(flux_out/snr_out,3)

        data = {'ID':range(1,len(ra)+1), 'ra':ra, 'dec':dec, 'flux(mJy/beam)':flux_out, 'error(mJy/beam)':err_out, 'S/N':snr_out}
        df=pd.DataFrame(data,columns=['ID', 'ra', 'dec','flux(mJy/beam)','error(mJy/beam)','S/N'])
        df.to_csv(name+'-source.log',index=False)

        region(ra,dec,name+'.reg',r=blend,color='green')

        print '# of sources detected above '+str(thresh)+' sigma:', len(ra)
        print 'central noise value (mJy/beam):', noise.min()
        return df      


def greybody(x, z, T, beta, norm, simp=False):

    x = np.array(x)  
    nu0 = 3e+12
    scale = 1e-35  
    lambda_obs = x*1e-6
    lambda_rest = lambda_obs/(1.0+z)
    nu=c/lambda_rest
  
    if simp:
        S = scale*norm*(nu**beta)*nu**3 / (np.exp(h*nu/(kb*T))-1.0)
    else: S = scale*norm*(1.0-np.exp(-(nu/nu0)**beta))*nu**3 / (np.exp(h*nu/(kb*T))-1.0)  

    return S      


def crossmatch(ra,dec,ra_cat,dec_cat,r):

    n = len(ra)
    index = np.zeros(n)-1
    ra_match = np.zeros(n)-1
    dec_match= np.zeros(n)-1

    for i in range(n):
        offset = np.sqrt( (ra[i]-ra_cat)**2 + (dec[i]-dec_cat)**2 )*3600.
        if offset.min() <= r:
            where = np.where(offset == offset.min())[0][0]
            index[i] = where
            ra_match[i] = ra_cat[where]
            dec_match[i] = dec_cat[where]

    return index,ra_match,dec_match    


























































  
