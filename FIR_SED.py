# The analysis of the FIR SED of the GOODS-N sample

# import modules:
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
import os.path
from scipy.stats import norm

# import packages:
from Sys_Tool.Script_command import *
import Figure_Tool.Single_Frame_Figure
from Figure_Tool.Figure_command import HandlerXoffset
from Computation_Tool.Gal_SED_models import Grey_PL, MCMC_GreyPL_Finish
from Astrophysics_Calc_Tool.Astro_Param import c_light, pc2cm, Wien_m_K


#####################################
# Assistant Parameters and Functions
#####################################

# cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# fitting parameters
niter = 10000
prob_scale = 50
cal_snr_cut = None
plt_snr_cut = 2
# ----

# Luminosity integration parameters
wave_range_3_1100 = np.array([3,1100]) * 1.0e-6  # unit: m
wave_range_40_120 = np.array([40,120]) * 1.0e-6  # unit: m
wave_range_40_1000 = np.array([40,1000]) * 1.0e-6  # unit: m
wave_range_8_1000 = np.array([8,1000]) * 1.0e-6  # unit: m
freq_int_step = 5.0e10  # unit: Hz
# ----


# figure settings:
figsize=(12,12)
tick_fontsize=30
tick_labelsize=30
label_fontsize=30
legend_fontsize=22
# --

# limits
wavelength_lim = [5, 2000]
# --

# labels
wavelength_label = '$\lambda$ ($\mu$m)'
flux_label = r'F$_{\nu}$ (mJy)'
# --

# marker
marker_Spitzer = '^'
marker_Herschel = 'o'
marker_SCUBA = 's'
markerfacecolor = 'none'
markeredgecolor_Sptizer = 'blue'
markeredgecolor_Herschel = 'magenta'
markeredgecolor_SCUBA = 'red'
markeredgecolor_bad = 'grey'
markeredgewidth = 4
markersize = 30
errbar_width = 2
errbar_capsize = 3
errbar_alpha = 0.7
label_SHS = 'Spitzer / Herschel / SCUBA-2'
label_model_total = 'Modified Blackbody Model'
label_model_grey = 'Greybody Component'
label_model_pl = 'Powerlaw Component'
# --

# line
model_grey_color = 'red'
model_grey_linewidth = 3
model_pl_color = 'blue'
model_pl_linewidth = 3
model_total_color = 'k'
model_total_linewidth = 5
# --



def fit_sed_obj(Main_Path, FIRSED_Content):
    ''' fit the SED of individual object'''
    ### load preliminary fitting settings
    Fit_set_path = Main_Path + '1.firsed_script/Fit_Setting.dat'
    Fit_set_Content = np.genfromtxt(Fit_set_path, names=True, dtype=None)
    # load info
    Nbb0 = Fit_set_Content['Nbb0']
    Nbb_step = Fit_set_Content['Nbb_step']
    Temp0 = Fit_set_Content['Temp0']
    Temp_step = Fit_set_Content['Temp_step']
    beta0 = Fit_set_Content['beta0']
    beta_step = Fit_set_Content['beta_step']
    alpha0 = Fit_set_Content['alpha0']
    alpha_step = Fit_set_Content['alpha_step']
    ### wavelength array used for fitting
    wave_array = np.array([24,100,160,250,350,450,850]) * 1.0e-6  # unit: m
    ### load photometry info
    Obj_Index = FIRSED_Content.field('Obj_Index')
    # FIR flux
    flux_24 = FIRSED_Content.field('F_24')
    flux_24_err = FIRSED_Content.field('F_24_err')
    flux_100 = FIRSED_Content.field('F_100')
    flux_100_err = FIRSED_Content.field('F_100_err')
    flux_160 = FIRSED_Content.field('F_160')
    flux_160_err = FIRSED_Content.field('F_160_err')
    flux_250 = FIRSED_Content.field('F_250')
    flux_250_err = FIRSED_Content.field('F_250_err')
    flux_350 = FIRSED_Content.field('F_350')
    flux_350_err = FIRSED_Content.field('F_350_err')
    flux_450 = FIRSED_Content.field('F_450')
    flux_450_err = FIRSED_Content.field('F_450_err')
    flux_850 = FIRSED_Content.field('F_850')
    flux_850_err = FIRSED_Content.field('F_850_err')
    ### fitting temporary file
    Fit_Temp = Main_Path + '3.FIR_SED/' + '1.Fit_SED/' + 'TEMP_FIRSED_DIR_Fit.dat'
    ### fitting
    for index in range(1,len(Obj_Index)):
        print "-"*15
        print "- Fitting Object # %d -"%(index+1)
        print "-"*15
        # set the starting parameters and steps
        start_params = [ Nbb0[index], Temp0[index], beta0[index], alpha0[index] ]
        steps = [ Nbb_step[index], Temp_step[index], beta_step[index], alpha_step[index] ]
        # fitting
        meas = np.array([ flux_24[index], flux_100[index], flux_160[index], flux_250[index], flux_350[index], flux_450[index], flux_850[index] ]) * 1.0e-3
        meas_err = np.array([ flux_24_err[index], flux_100_err[index], flux_160_err[index], flux_250_err[index], flux_350_err[index], flux_450_err[index], flux_850_err[index] ]) * 1.0e-3
        final, final_err, num_fit = MCMC_GreyPL_Finish(wave_array, meas, meas_err, start_params, steps, niter, snr_cut=cal_snr_cut, prob_scale=prob_scale, weight="snr")
        print final
        # find the dust temperature
        dust_wave_array = np.arange(wavelength_lim[0], wavelength_lim[-1], 0.1) * 1.0e-6
        model = Grey_PL(dust_wave_array, final[0], final[1], final[2], final[3])
        dust_temp_wave = dust_wave_array[np.where(model == np.max(model))]
        Dust_Temp = Wien_m_K / dust_temp_wave
        # save answers to temp file
        temp_append = open(Fit_Temp, "a")
        temp_append.write("%-25d%-25d%-25f%-25f%-25f%-25f%-25f%-25f%-25f%-25f%-25f"%(index+1,num_fit,final[0],final_err[0],final[1],final_err[1],final[2],final_err[2],final[3],final_err[3],Dust_Temp) + "\n")
        temp_append.close()
        # plot the fitting results
        plt_sed_obj(Main_Path, index, FIRSED_Content, final[0], final[1], final[2], final[3])
    ### reload the answers and make FITS columns
    temp_Content = np.genfromtxt(Fit_Temp,names=True,dtype=None)
    # make columns
    col_Num_Fit = fits.Column(name='Num_Fit', format='I', array=temp_Content['Num_Fit'])
    col_Nbb = fits.Column(name='Nbb', format='D', array=temp_Content['Nbb'])
    col_Nbb_err = fits.Column(name='Nbb_err', format='D', array=temp_Content['Nbb_err'])
    col_Temp = fits.Column(name='Temp', format='D', array=temp_Content['Temp'])
    col_Temp_err = fits.Column(name='Temp_err', format='D', array=temp_Content['Temp_err'])
    col_beta = fits.Column(name='beta', format='D', array=temp_Content['beta'])
    col_beta_err = fits.Column(name='beta_err', format='D', array=temp_Content['beta_err'])
    col_alpha = fits.Column(name='alpha', format='D', array=temp_Content['alpha'])
    col_alpha_err = fits.Column(name='alpha_err', format='D', array=temp_Content['alpha_err'])
    col_Dust_Temp = fits.Column(name='Dust_Temp', format='D', array=temp_Content['Dust_Temp'])
    # combine columns
    cols = fits.ColDefs([col_Num_Fit, col_Nbb, col_Nbb_err, col_Temp, col_Temp_err, col_beta, col_beta_err, col_alpha, col_alpha_err, col_Dust_Temp])
    return cols
# --




def plt_sed_obj(Main_Path, index, FIRSED_Content, Nbb, Temp, beta, alpha):
    ''' plot the SED of the object'''
    print "# Plotting the SED for objects %d..."%(index+1)
    ### output figure path
    figure_path = Main_Path + '3.FIR_SED/2.Plt_SED/' + 'Obj_%d_FIRSED.pdf'%(index+1)
    ### wavelength array
    wavelength_Spitzer = np.array([24])
    wavelength_Herschel = np.array([100,160,250,350])
    wavelength_SCUBA = np.array([450,850])
    wave_plt_array = np.linspace(wavelength_lim[0], wavelength_lim[-1],10000)
    wave_cal_array = wave_plt_array * 1.0e-6
    ### FIR flux
    flux_24 = FIRSED_Content.field('F_24')[index]
    flux_24_err = FIRSED_Content.field('F_24_err')[index]
    flux_100 = FIRSED_Content.field('F_100')[index]
    flux_100_err = FIRSED_Content.field('F_100_err')[index]
    flux_160 = FIRSED_Content.field('F_160')[index]
    flux_160_err = FIRSED_Content.field('F_160_err')[index]
    flux_250 = FIRSED_Content.field('F_250')[index]
    flux_250_err = FIRSED_Content.field('F_250_err')[index]
    flux_350 = FIRSED_Content.field('F_350')[index]
    flux_350_err = FIRSED_Content.field('F_350_err')[index]
    flux_450 = FIRSED_Content.field('F_450')[index]
    flux_450_err = FIRSED_Content.field('F_450_err')[index]
    flux_850 = FIRSED_Content.field('F_850')[index]
    flux_850_err = FIRSED_Content.field('F_850_err')[index]
    ### flux array
    flux_Spitzer = np.array([flux_24])
    flux_Herschel = np.array([flux_100, flux_160, flux_250, flux_350])
    flux_SCUBA = np.array([flux_450, flux_850])
    ### flux uncertainty
    flux_Spitzer_err = np.array([flux_24_err])
    flux_Herschel_err = np.array([flux_100_err, flux_160_err, flux_250_err, flux_350_err])
    flux_SCUBA_err = np.array([flux_450_err, flux_850_err])
    ### model
    model_total, model_grey, model_pl = Grey_PL(wave_cal_array, Nbb, Temp, beta, alpha, sep_model=True)
    ### start plotting...
    # build figure frame
    figure, ax = Figure_Tool.Single_Frame_Figure.Single_Frame_Birth(figsize, tick_fontsize, tick_labelsize)
    ### plot SED
    ## Spitzer
    plt_Spitzer, = plt.plot(wavelength_Spitzer, flux_Spitzer, linestyle='none', marker=marker_Spitzer, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor_Sptizer, markeredgewidth=markeredgewidth)
    plt.errorbar(wavelength_Spitzer, flux_Spitzer, yerr=flux_Spitzer_err, linestyle='none', ecolor=markeredgecolor_Sptizer, linewidth=errbar_width, capsize=errbar_capsize, alpha=errbar_alpha)
    # data with S/N <= snr_cut
    id_bad_Spitzer = np.where((flux_Spitzer / flux_Spitzer_err) <= plt_snr_cut)
    plt.plot(wavelength_Spitzer[id_bad_Spitzer], flux_Spitzer[id_bad_Spitzer], linestyle='none', marker=marker_Spitzer, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor_bad, markeredgewidth=markeredgewidth)
    plt.errorbar(wavelength_Spitzer[id_bad_Spitzer], flux_Spitzer[id_bad_Spitzer], yerr=flux_Spitzer_err[id_bad_Spitzer], linestyle='none', ecolor=markeredgecolor_bad, linewidth=errbar_width, capsize=errbar_capsize, alpha=errbar_alpha)
    ## Herschel
    plt_Herschel, = plt.plot(wavelength_Herschel, flux_Herschel, linestyle='none', marker=marker_Herschel, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor_Herschel, markeredgewidth=markeredgewidth)
    plt.errorbar(wavelength_Herschel, flux_Herschel, yerr=flux_Herschel_err, linestyle='none', ecolor=markeredgecolor_Herschel, linewidth=errbar_width, capsize=errbar_capsize, alpha=errbar_alpha)
    # data with S/N <= snr_cut
    id_bad_Herschel = np.where((flux_Herschel / flux_Herschel_err) <= plt_snr_cut)
    plt.plot(wavelength_Herschel[id_bad_Herschel], flux_Herschel[id_bad_Herschel], linestyle='none', marker=marker_Herschel, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor_bad, markeredgewidth=markeredgewidth)
    plt.errorbar(wavelength_Herschel[id_bad_Herschel], flux_Herschel[id_bad_Herschel], yerr=flux_Herschel_err[id_bad_Herschel], linestyle='none', ecolor=markeredgecolor_bad, linewidth=errbar_width, capsize=errbar_capsize, alpha=errbar_alpha)
    # SCUBA
    plt_SCUBA, = plt.plot(wavelength_SCUBA, flux_SCUBA, linestyle='none', marker=marker_SCUBA, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor_SCUBA, markeredgewidth=markeredgewidth)
    plt.errorbar(wavelength_SCUBA, flux_SCUBA, yerr=flux_SCUBA_err, linestyle='none', ecolor=markeredgecolor_SCUBA, linewidth=errbar_width, capsize=errbar_capsize, alpha=errbar_alpha)
    # data with S/N <= snr_cut
    id_bad_SCUBA = np.where((flux_SCUBA / flux_SCUBA_err) <= plt_snr_cut)
    plt.plot(wavelength_SCUBA[id_bad_SCUBA], flux_SCUBA[id_bad_SCUBA], linestyle='none', marker=marker_SCUBA, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor_bad, markeredgewidth=markeredgewidth)
    plt.errorbar(wavelength_SCUBA[id_bad_SCUBA], flux_SCUBA[id_bad_SCUBA], yerr=flux_SCUBA_err[id_bad_SCUBA], linestyle='none', ecolor=markeredgecolor_bad, linewidth=errbar_width, capsize=errbar_capsize, alpha=errbar_alpha)
    ### plot model
    plt_model_grey, = plt.plot(wave_plt_array, model_grey * 1.0e3, color=model_grey_color, linestyle='-', linewidth=model_grey_linewidth)
    plt_model_pl, = plt.plot(wave_plt_array, model_pl * 1.0e3, color=model_pl_color, linestyle='-', linewidth=model_pl_linewidth)
    plt_model_total, = plt.plot(wave_plt_array, model_total * 1.0e3, color=model_total_color, linestyle='-', linewidth=model_total_linewidth)
    # scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    ## legend
    lgd = plt.legend([(plt_Spitzer,plt_Herschel,plt_SCUBA), plt_model_total, plt_model_grey, plt_model_pl], [label_SHS, label_model_total, label_model_grey, label_model_pl], handler_map={plt_Spitzer:HandlerXoffset(x_offset=20), plt_Herschel:HandlerXoffset(x_offset=0), plt_SCUBA:HandlerXoffset(x_offset=-20)}, loc='upper left', ncol=1, numpoints=1, markerscale=0.5, frameon=False, fontsize=legend_fontsize)
    ## label
    figure, ax = Figure_Tool.Single_Frame_Figure.Single_Frame_Labels(figure, ax, wavelength_label, flux_label, label_fontsize)
    ## limits
    flux_lim = [np.min(np.hstack((flux_Spitzer-flux_Spitzer_err,flux_SCUBA-flux_SCUBA_err)))-1.0e-2, np.max(np.hstack((flux_Herschel+flux_Herschel_err,flux_SCUBA+flux_SCUBA_err)))+5]
    figure, ax = Figure_Tool.Single_Frame_Figure.Single_Frame_Limits(figure, ax, wavelength_lim, flux_lim)
    ## saving
    figure, ax = Figure_Tool.Single_Frame_Figure.Single_Frame_Funeral(figure, ax, figure_path)
# --



def cal_fir_lum(wave_range, Nbb, Temp, beta, alpha, redshift):
    ''' calculate the integrated FIR luminosity given the wavelength range and the SED fitting parameters

        the redshift is also needed for the luminosity distance

        '''
    ### convert to frequency
    freq_range = (c_light / wave_range)[::-1]
    ## model values
    freq_array = np.arange(freq_range[0], freq_range[-1]+freq_int_step, freq_int_step)
    model = Grey_PL(c_light / freq_array, Nbb, Temp, beta, alpha)
    # integration
    flux = ( np.sum(model) - model[0]/2 - model[-1]/2 ) * freq_int_step * 1.0e-3 * 1.0e-23  # unit erg s-1 cm-2
    # FIR luminosity
    lum_dist = cosmo.luminosity_distance(redshift).value * 1.0e6 * pc2cm
    fir_lum = 4 * np.pi * lum_dist**2 * flux
    return fir_lum
# ----





########################
# Official Functions
########################
def startFIR_SED(Main_Path):
    ''' create the directory for FIR_SED'''
    FIR_SED_path = Main_Path + '3.FIR_SED/'
    find_dir(FIR_SED_path)
    print "# GOT ACCESS TO *FIR_SED* ROOT DIRECTORY...\n"
# --



def Fit_ModBB(Main_Path):
    ''' fit each SED with the modified blackbody model, suggested by Equation (9) in Casey+2014(PHYREV.1402.1456v1)
        
        The fitting process adopts a MCMC fashion

        '''
    print "# Fitting individual SED using MCMC..."
    ### output path
    # fit
    Fit_path  = Main_Path + '3.FIR_SED/' + '1.Fit_SED/'
    find_dir(Fit_path)
    Fit_DIR = Fit_path + 'FIRSED_DIR_Fit.fits'
    Fit_Temp = Fit_path + 'TEMP_FIRSED_DIR_Fit.dat'
    # plot
    Plt_path = Main_Path + '3.FIR_SED/' + '2.Plt_SED/'
    find_dir(Plt_path)
    ### input directory
    FIRSED_DIR = Main_Path + '2.FIR_Data/' + 'FIRSED_DIR.fits'
    # load info
    FIRSED_Load = fits.open(FIRSED_DIR)
    FIRSED_Content = FIRSED_Load[1].data
    FIRSED_Columns = FIRSED_Content.columns
    ### open the temperory file to save fitting results; create one if the file doesn't exist
    if os.path.isfile(Fit_Temp) == False:
        temp_data = open(Fit_Temp, "w")
        temp_data.write("%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s"%("Obj_index","Num_Fit","Nbb","Nbb_err","Temp","Temp_err","beta","beta_err","alpha","alpha_err","Dust_Temp") + "\n")
        temp_data.close()
    ### fitting
    fit_cols = fit_sed_obj(Main_Path, FIRSED_Content)
    ### merge the fitting results into the candidates list
    HDU = fits.BinTableHDU.from_columns(FIRSED_Columns + fit_cols)
    HDU.writeto(Fit_DIR, clobber=True)
# ----



def Lum_FIR(Main_Path):
    ''' calcualte the FIR luminosity and the "q" parameter of each object according to the fitted SED and their redshift z
        
        "FIR" luminosity:
        1. obtain the integration flux F_IR by integrating the SED from the FIR-wavelength range
        2. calculate luminosity distance from redshift
        3. report FIR luminosity
        4. calculate Radio luminosity from the radio power
        
        "q" parameter:
        1. calculate the flux density at 60um and 100um (in unit of Jy)
        2. calculate the flux density at 1.4GHz (in unit of W m-2 Hz-1)
        3. calculate the FIR and q parameter using the equation 5 in Yun+2001(ApJ,554,803)
        
        '''
    print "# Calculating FIR luminosity..."
    ### output path
    Lum_path = Main_Path + '3.FIR_SED/' + '3.FIRLum_SED/'
    find_dir(Lum_path)
    Lum_DIR = Lum_path + 'FIRSED_DIR_Lum.fits'
    ### input directory
    Fit_DIR = Main_Path + '3.FIR_SED/1.Fit_SED/' + 'FIRSED_DIR_Fit.fits'
    # load info
    Fit_Load = fits.open(Fit_DIR)
    Fit_Content = Fit_Load[1].data
    Fit_Columns = Fit_Content.columns
    Obj_Index = Fit_Content.field('Obj_Index')
    Rad_Pow = Fit_Content.field('Rad_Pow')
    ### computing...
    # create arrays to save answers
    Lum_FIR_3_1100 = np.ones_like(Obj_Index)
    Lum_FIR_40_120 = np.ones_like(Obj_Index)
    Lum_FIR_40_1000 = np.ones_like(Obj_Index)
    Lum_FIR_8_1000 = np.ones_like(Obj_Index)
    Flux_60 = np.ones_like(Obj_Index)
    Flux_100 = np.ones_like(Obj_Index)
    Flux_1_4_GHz = np.ones_like(Obj_Index)
    q_param = np.ones_like(Obj_Index)
    for index in range(0,len(Obj_Index)):
        print "-"*15
        print "- Fitting Object # %d -"%(index+1)
        print "-"*15
        ## import fitting parameters
        Nbb = Fit_Content.field('Nbb')[index]
        Temp = Fit_Content.field('Temp')[index]
        beta = Fit_Content.field('beta')[index]
        alpha = Fit_Content.field('alpha')[index]
        ## load redshift
        redshift = Fit_Content.field('redshift')[index]
        # FIR luminosity
        Lum_FIR_3_1100[index] = cal_fir_lum(wave_range_3_1100, Nbb, Temp, beta, alpha, redshift)
        Lum_FIR_40_120[index] = cal_fir_lum(wave_range_40_120, Nbb, Temp, beta, alpha, redshift)
        Lum_FIR_40_1000[index] = cal_fir_lum(wave_range_40_1000, Nbb, Temp, beta, alpha, redshift)
        Lum_FIR_8_1000[index] = cal_fir_lum(wave_range_8_1000, Nbb, Temp, beta, alpha, redshift)
        ## flux density at 60um and 100um in unit of Jy
        Flux_60[index] = Grey_PL(6.0e-5 * (1+redshift), Nbb, Temp, beta, alpha) * (1+redshift)**(-0.2)  # K-correction
        Flux_100[index] = Grey_PL(1.0e-4 * (1+redshift), Nbb, Temp, beta, alpha) * (1+redshift)**(-0.2)  # K-correction
        # luminosity distance in unit of cm
        lum_dist = cosmo.luminosity_distance(redshift).value  * 1.0e6 * pc2cm
        # flux density at 1.4 GHz in unit of W m-2 Hz-1
        Flux_1_4_GHz[index] = Rad_Pow[index] / (4 * np.pi * lum_dist**2 * (1+redshift)**(-0.2)) * 1.0e-3
        # q parameter
        q_param[index] = np.log10(1.26e-14 * (2.58 * Flux_60[index] + Flux_100[index]) / (3.75e12)) - np.log10(Flux_1_4_GHz[index])
    ### save columns
    col_Lum_FIR_3_1100 = fits.Column(name='Lum_FIR_3_1100', format='D', array=Lum_FIR_3_1100)
    col_Lum_FIR_40_120 = fits.Column(name='Lum_FIR_40_120', format='D', array=Lum_FIR_40_120)
    col_Lum_FIR_40_1000 = fits.Column(name='Lum_FIR_40_1000', format='D', array=Lum_FIR_40_1000)
    col_Lum_FIR_8_1000 = fits.Column(name='Lum_FIR_8_1000', format='D', array=Lum_FIR_8_1000)
    col_Flux_60 = fits.Column(name='Flux_60', format='D', array=Flux_60)
    col_Flux_100 = fits.Column(name='Flux_100', format='D', array=Flux_100)
    col_Flux_1_4_GHz = fits.Column(name='Flux_1_4_GHz', format='D', array=Flux_1_4_GHz)
    col_q_param = fits.Column(name='q_param', format='D', array=q_param)
    # define column
    cols = fits.ColDefs([col_Lum_FIR_3_1100, col_Lum_FIR_40_120, col_Lum_FIR_40_1000, col_Lum_FIR_8_1000, col_Flux_60, col_Flux_100, col_Flux_1_4_GHz, col_q_param])
    ### merge the calculated luminosity into the candidates list
    HDU = fits.BinTableHDU.from_columns(Fit_Columns + cols)
    HDU.writeto(Lum_DIR, clobber=True)
# --





