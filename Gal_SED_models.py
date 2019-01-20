# There are models describing the galaxy SEDs

# import modules:
import numpy as np
import time
from scipy.interpolate import interp1d
from scipy.stats import norm


# import packages
from Astrophysics_Calc_Tool.Astro_Param import c_light, h_planck, kB, wave_unitoptdepth




def find_ln_prob_Gaussian(meas, meas_err, model, weight=None):
    ''' given an array of measurements (yi) and the corresponding errors (sigma_i), and the predictions of the model (y_bar), calculate the combined log probability
        
                                                   (y_i - y_bar)^2      1
        ln P = Sum { weight_i x [ - ln sigma_i - ------------------- - --- ln (2 pi) }
                                                     2 sigma_i^2        2

        if weight is "rms", then multiply the ln_prob of each component i with the normalized rms


        '''
    if (len(meas)!=len(meas_err)) or (len(meas)!=len(model)):
        print "# ERROR: meas/meas_err/model must have the same dimension! Please check..."
        return False
    elif (np.isnan(meas_err).any()) or (np.nanmin(meas_err)<=0):
        print "# ERROR: meas_err must have non-Nan and positive values! Please check..."
        return False
    else:
        if weight is None:
            weight_array = np.ones_like(meas)
        elif weight=="snr":
            snr = 1.0 * meas / meas_err
            weight_array = snr / np.sum(snr)
        ln_prob = np.sum( weight_array * (- np.log(meas_err) - 1.0 * (meas - model)**2 / (2 * meas_err**2) - 0.5 * np.log(2*np.pi) ) )
        return ln_prob
# ----


def wave_c_Casey(alpha, Temp):
    ''' this is used to approximate the lambda_c in the modified blackbody model
        
        from Table 1 of Casey+2012(MNRAS,425,3094)
        '''
    b1 = 26.68
    b2 = 6.246
    b3 = 1.905e-4
    b4 = 7.243e-5
    L_func = ( (b1 + b2 * alpha)**(-2.) + (b3 + b4 * alpha) * Temp )**(-1.0)
    wave_c = 3. / 4 * L_func * 1.0e-6  # convert to the unit of m
    return wave_c
# ----


def Grey_PL(wavelength, Nbb, Temp, beta, alpha, sep_model=False):
    ''' this is the modified blackbody model for the galaxy SED mentioned by Equation (9) in Casey+2014(PHYREV.1402.1456v1)
        
        the model is actually the combination of the greybody and a powerlaw form. We adopt some approximation of the free parameters suggested by Casey+2012(MNRAS,425,3094), see their Table 1
        
        input
        - wavelength (unit: m)
        - Nbb, beta, alpha, Temp are free parameters
        - sep_model 
            if True, then the Greybody and PL parts of the total function will also be returned, respectively.

        output
        - flux density  [S(lambda)]
            in unit of Jy
        
        Note:
        - Nbb is in the log10 scale
        - original equation of Npl in Casey+2012(MNRAS,425,3094) Table 1 is wrong!!! should be
        
                   (1 - exp[-(wave_0 / wave_c)^beta]) c^3 wave_c^(-(alpha + 3))
        Npl = Nbb --------------------------------------------------------------
                                   exp[hc / wave_c kB T] - 1

        '''
    ### Casey+2012 parameters
    wave_c = wave_c_Casey(alpha, Temp)
    Nbb_lin = 10**(Nbb)
    Npl = Nbb_lin * (1 - np.exp(- (wave_unitoptdepth / wave_c)**beta )) * c_light**3 * wave_c**(-(3.+alpha)) / (np.exp(h_planck * c_light / (wave_c * kB * Temp)) - 1)
    ### final expression
    S_lambda_grey = Nbb_lin * (1 - np.exp(- (wave_unitoptdepth / wavelength)**beta)) * (c_light / wavelength)**3 / ( np.exp(h_planck * c_light / (wavelength * kB * Temp)) - 1)
    S_lambda_pl = Npl * (wavelength)**(1.0*alpha) * np.exp(- (wavelength / wave_c)**2)
    S_lambda = S_lambda_grey + S_lambda_pl
    if sep_model==False:
        return S_lambda
    elif sep_model==True:
        return S_lambda, S_lambda_grey, S_lambda_pl
    else:
        print "# ERROR: sep_model should be either True or False! Please check..."
        return False
# ----



def MCMC_GreyPL(wave_array, meas, meas_err, start_params, steps, niter, snr_cut=None, weight=None):
    ''' a MCMC fitting process for the Grey_PL model
        
        input
        - wave_array
            an array of wavelength (unit: m)
        - meas, meas_err
            measurements and uncertainties at the corresponding wavelength
        - start_params
            initial values of the parameters
        - steps
            step values for iteration
        - niter
            number of iterations
        - snr_cut
            if not None, then exclude the data from fitting which has S/N lower than snr_cut
            (default = None)
        
        Note:
            we assume the parameters are {Nbb, Temp, beta, alpha}
            - Nbb is in log scale -
        

        output
        - out_steps:
            parameter values at each step
        - out_prob:
            probability at each step
        - accept_rate:
            accept rate of the parameters during the iteration
        
        '''
    ### apply snr_cut:
    snr = 1.0 * meas / meas_err
    if snr_cut is not None:
        id_gd = np.where(snr > snr_cut)
        wave_array_gd = wave_array[id_gd]
        meas_gd = meas[id_gd]
        meas_err_gd = meas_err[id_gd]
        num_fit = len(id_gd[0])
    else:
        wave_array_gd = wave_array * 1.0
        meas_gd = meas * 1.0
        meas_err_gd = meas_err * 1.0
        num_fit = len(meas)
    if len(meas_gd)==0:
        print "# All data have S/N lower than %.2f!!! No fitting is available!"%(snr_cut)
        return False
    ### create arrays to save answers
    out_steps = np.ones((niter, len(start_params)))
    out_prob = np.ones(niter)
    ### basic parameters for fitting
    # accept rate calculation
    acc_num = 0.
    rej_num = - (niter - 1.)
    # make steps
    acc_flag = 0  # a flag indicating if a new step could be made
    U_check_step = 0.0                                  # a uniformly distributed parameter in [0,1]; used to check if the step could be made
    U_min = 0
    U_max = 1
    G_make_step = np.ones_like(start_params)            # a normaly distributed parameter with \mu=0 and \sigma=1; used to make a step from the previous paramters
    G_mu = 0.
    G_std = 1.
    next_step = np.ones_like(start_params)
    next_prob = 0.
    # code running progress
    prog_num = 0.0
    ### initial step
    # save initial choice
    out_steps[0] = start_params
    # calculate initial probability
    model_initial = Grey_PL(wave_array_gd, start_params[0], start_params[1], start_params[2], start_params[3])
    out_prob[0] = find_ln_prob_Gaussian(meas_gd, meas_err_gd, model_initial, weight=weight)
    ### start MCMC
    for index_mc in range(1, niter):
        start_time = time.time()
        acc_flag = 0
        while (acc_flag == 0):
            rej_num = rej_num + 1
            G_make_step = np.random.normal(G_mu, G_std, size=len(start_params))
            next_step = out_steps[index_mc-1] + G_make_step * steps
            model_next = Grey_PL(wave_array_gd, next_step[0], next_step[1], next_step[2], next_step[3])
            prob_next = find_ln_prob_Gaussian(meas_gd, meas_err_gd, model_next, weight=weight)
            if (prob_next > out_prob[index_mc-1]):
                acc_flag = 1
            else:
                U_check_step = np.random.uniform(U_min, U_max)
                if U_check_step < np.exp(prob_next - out_prob[index_mc-1]):
                    acc_flag = 1
        # accept new steps
        acc_num = acc_num + 1
        out_steps[index_mc] = next_step
        out_prob[index_mc] = prob_next
        # record the progress
        prog_num = prog_num + 1.0
        prog_per = prog_num / (niter - 1.0) * 100.
        end_time = time.time()    # record the time at NEARLY the end
        total_time = (end_time - start_time) * (niter - 1.0) / 60.     # predict the total time (unit: min)
        remain_time = total_time * (1 - prog_per / 100)     # predict the remaining time (unit: min)
        print "current progress: %.6f%%; predicted total time: %f minutes; predicted remaining time: %f minutes" % (prog_per, total_time, remain_time), '\r',
    ### accept rate
    accept_rate = 1.0 * acc_num / (acc_num + rej_num)
    print "\n accept rate is %f"%(accept_rate)
    ### output
    return out_steps, out_prob, accept_rate, num_fit
# ----



def MCMC_GreyPL_Finish(wave_array, meas, meas_err, start_params, steps, niter, snr_cut=None, prob_scale=0, weight=None):
    ''' perform the MCMC fitting, using "MCMC_GreyPL" 
        and produce the final results with uncertainties
        
        Given the output steps and probabilities, we need to find the best answer for each parameter, respectively:
        for each parameter,
            1. fit the output steps using a Gaussian distribution => find the mean and std (err)
            2. report results
            
        prob_scale is used to scale the probability to avoid extremely negative values

        '''
    ### find the output steps and probability
    out_steps, out_prob, accept_rate, num_fit = MCMC_GreyPL(wave_array, meas, meas_err, start_params, steps, niter, snr_cut=snr_cut, weight=weight)
    ### create arrays to save answers
    final = np.ones_like(start_params)
    final_err = np.ones_like(start_params)
    ### find the best answer for each parameter
    for index_param in range(0,len(start_params)):
        mean, std = norm.fit(out_steps[:,index_param])
        final[index_param] = mean
        final_err[index_param] = std
    ### output
    return final, final_err, num_fit
# ----




























