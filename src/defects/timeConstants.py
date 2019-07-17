#! python3

import os
import numpy as np
import semiconductor.material as mat
import scipy.constants as C

tm = mat.ThermalVelocity(author='Green_1990', temp=300)
_ni = mat.IntrinsicCarrierDensity()


def escape_time_constants(Ed, sigma_e, sigma_h, ni, temp, ne0, nh0, bk_tau):
    '''
    returns the escape time constants for a single level defect.
    These values are returned in a dictionary

    Parameters
    ----------
    Ed:
        activation energy level
    sigma_e:
        capture cross section of electrons
    sigma_h:
        capture cross section of holes
    ni:
        intrinsic carrier density
    temp:
        temperature
    ne0:
        number of electrons in the dark
    nh0:
        number of holes in the dark
    bk_tau:
        a background lifetime

    Returns
    -------
    dictionary:
        The keys of which are
         * DLTS: the time constant expected from DLTS
         * DLTS-Mo: The time constant expected from minority carrier DLTS
         * PC: the time constant expected from transient photoconductance
         * HH-simp: the time constant expected from transient photoconductance according to the simplified hornbeck and hanes model.
    '''

    return {'electron': get_electron_timeconstant(Ed, sigma_e, ni, temp),
            'hole': get_hole_timeconstant(Ed, sigma_h, ni, temp),
            'recombination': get_rec_transient(Ed, sigma_e, sigma_h, ne0, nh0, temp),
            'PC': get_PC_transient(Ed, sigma_e, sigma_h, ni, ne0, nh0, temp),
            'HH-simp': HornbeckHanes_simplified(Ed, sigma_e, sigma_h, ni, ne0, nh0,  temp, bk_tau)}


def get_electron_timeconstant(Ed, sigma_e,  ni, temp=300):
    '''
    Get the time for the emission of an electron

    Parameters
    ----------

    Ed :
        Energy level of the defect from the intrinsic level
    sigma_e :
        The capture cross section of electrons
    ni :
        The intrinsic carrier density
    temp :
        The temperature of the material
    '''

    vth_e, vth_h = tm.update(temp=temp)

    Vt = C.k * temp / C.e

    ne1 = ni * np.exp(Ed / Vt)

    esc_thran = 1. / (sigma_e * vth_e * ne1)

    return esc_thran


def get_hole_timeconstant(Ed, sigma_h, ni, temp=300):
    '''
    Get the lifetime from the emission of an hole

    Parameters
    ----------

    Ed :
        Energy level of the defect from the intrinsic level
    sigma_h :
        The capture cross section of holes
    ni :
        The intrinsic carrier density
    temp :
        The temperature of the material
    '''

    vth_e, vth_h = tm.update(temp=temp)

    Vt = C.k * temp / C.e

    nh1 = ni * np.exp(-Ed / Vt)

    esc_thran = 1. / (sigma_h * vth_h * nh1)

    return esc_thran


def get_DLTS_transient(Ed, sigma_e, sigma_h, ni, temp=300):
    '''
    Calculates the time constant from DLTS. This assumes the doping type
    is on the opposite side of the band half than the defect.

    Parameters
    ----------

    Ed :
        Energy level of the defect from the intrinsic level
    sigma_e :
        The capture cross section of electrons
    sigma_h :
        The capture cross section of holes
    ni :
        The intrinsic carrier density
    temp :
        The temperature of the material
    '''

    vth_e, vth_h = tm.update(temp=temp)

    Vt = C.k * temp / C.e
    #print(vth_e, vth_h, 'thermal val {0:.2e} {1:.4f} {2:.2e}'.format(ni, Vt, sigma_h))

    nh1 = ni * np.exp(-Ed / Vt)
    ne1 = ni * np.exp(Ed / Vt)

    if Ed < 0:
        #print('the Ed is', Ed)
        esc_thran = 1. / (sigma_h * vth_h * nh1)
    elif Ed > 0:
        esc_thran = 1. / (sigma_e * vth_e * ne1)

    return esc_thran


def get_minority_DLTS_transient(Ed, sigma_e, sigma_h, ni, temp=300):
    '''
    Calculates the time constant from DLTS minority carrier measurement.
    This assumes the doping type is on the opposite side of the band half
    than the defect.

    Parameters
    ----------

    Ed :
        Energy level of the defect from the intrinsic level
    sigma_e :
        The capture cross section of electrons
    sigma_h :
        The capture cross section of holes
    ni :
        The intrinsic carrier density
    temp :
        The temperature of the material
    '''

    vth_e, vth_h = tm.update(temp=temp)

    Vt = C.k * temp / C.e

    nh1 = ni * np.exp(-Ed / Vt)
    ne1 = ni * np.exp(Ed / Vt)

    if Ed > 0:
        esc_thran = 1. / (sigma_h * vth_h * nh1)
    elif Ed < 0:
        esc_thran = 1. / (sigma_e * vth_e * ne1)

    return esc_thran


def get_rec_transient(Ed, sigma_e, sigma_h, ne0, nh0, temp=300):
    '''
    Calculates the time constant from recombiation
    This represents the capture of the majority carrier, as its concentration
    is significantly larger and does not change.

    Parameters
    ----------

    Ed :
        Energy level of the defect from the intrinsic level
    sigma_e :
        The capture cross section of electrons
    sigma_h :
        The capture cross section of holes
    ne0 :
        The concentration of electrons in the dark
    nh0 :
        The concentration of holes in the dark
    temp :
        The temperature of the material
    '''
    vth_e, vth_h = tm.update(temp=temp)

    Vt = C.k * temp / C.e

    if nh0 > ne0:
        nh_thing = nh0
        ne_thing = 0
    else:
        ne_thing = ne0
        nh_thing = 0

    #print('222ne0 {0:.2e}, nho {1:.2e}'.format(ne0, nh0))
    esc_thran = 1. / (
        sigma_h * vth_h * (nh_thing) +
        sigma_e * vth_e * (ne_thing))

    return esc_thran


def get_PC_transient(Ed, sigma_e, sigma_h, ni, ne0, nh0, temp=300):
    '''
    Calculates the time constant from PC

    Parameters
    ----------

    Ed :
        Energy level of the defect from the intrinsic level
    sigma_e :
        The capture cross section of electrons
    sigma_h :
        The capture cross section of holes
    ni :
        The intrinsic carrier density
    ne0 :
        The concentration of electrons in the dark
    nh0 :
        The concentration of holes in the dark
    temp :
        The temperature of the material

    '''
    vth_e, vth_h = tm.update(temp=temp)

    Vt = C.k * temp / C.e

    nh1 = ni * np.exp(-Ed / Vt)
    ne1 = ni * np.exp(Ed / Vt)

    if nh0 > ne0:
        ne_thing = ne1
        nh_thing = nh1 + nh0
    else:
        ne_thing = ne1 + ne0
        nh_thing = nh1

    esc_thran = 1. / (
        sigma_h * vth_h * (nh_thing) +
        sigma_e * vth_e * (ne_thing))

    return esc_thran


def HornbeckHanes_simplified(Ed, sigma_e, sigma_h, Nd, ni, ne0, nh0, temp, tau_bkg):
    '''
    Calculates the time constant from hornbeck and Hanes Simplified model.
    This is equation 13 from 10.1103/PhysRev.97.311

    Parameters
    ----------

    Ed :
        Energy level of the defect from the intrinsic level
    sigma_e :
        The capture cross section of electrons
    sigma_h :
        The capture cross section of holes
    ni :
        The intrinsic carrier density
    ne0 :
        The concentration of electrons in the dark
    nh0 :
        The concentration of holes in the dark
    temp :
        The temperature of the material
    tau_bkg:
        The effective recombination lifetime of the semiconductor
    '''
    vth_e, vth_h = tm.update(temp=temp)
    Vt = C.k * temp / C.e

    nh1 = ni * np.exp(-Ed / Vt)
    ne1 = ni * np.exp(Ed / Vt)

    Ef = np.log(ni / nh0) * Vt

    if nh0 > ne0:
        ne_thing = ne1
        nh_thing = nh1 + nh0

        Ndh = Nd / (np.exp(-(Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)

    else:
        ne_thing = ne1 + ne0
        nh_thing = nh1

        Nde = Nd / (np.exp((Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)

    tau_d = 1. / (sigma_e * vth_e * (ne_thing) +
                  sigma_h * vth_h * (nh_thing))

    HH_s = tau_d + tau_bkg * tau_d / tau_t
    return HH_s


def HH_full(Ed, sigma_e, sigma_h, Nd, ni, ne0, nh0, temp, tau_bkg):
    '''
    Man there is a lot of defined vairables in this paper. this current;y doesn't work

    Parameters
    ----------

    tau_r:
        recombiation in the bulk of the semiconductor
    tau_g:
        generation of minority carriers from the trap. sigma_e vth_e n_i exp(+-Ed)z`
    tau_dr:
        recombination in the defect
    simga:
        capture cross section of the minority
    vth_m: thermal velocity of the minority carrier
    N: change in occupation from the dark.
    y: 1 under illumination 0 in the dark. Trap occupation.

    '''

    Vt = C.k * temp / C.e
    vth_e, vth_h = tm.update(temp=temp)

    HH_s = HornbeckHanes_simplified(
        Ed, sigma_e, sigma_h, Nd, ni, ne0, nh0, temp, tau_bkg)

    if nh0 > ne0:
        tau_b = 1 / (sigma_h * vth_h * nh0)

    else:
        tau_b = 1 / (sigma_e * vth_e * ne0)

    itau_HH_f = 1 / tau_b + 1 / HH_s

    return 1 / itau_HH_f


def HH_s_plot(ax, Ed, sigma_e, sigma_h, Nd, ni, nh0, ne0, temp, tau_BGK):

    vth_e, vth_h = tm.update(temp=temp)
    Vt = C.k * temp / C.e

    yi = np.linspace(0, 1)
    Ef = np.log(ni / nh0) * Vt

    tau_r = tau_BGK

    if nh0 > ne0:

        tau_g = 1 / (sigma_e * ni * vth_e * np.exp(Ed / Vt))
        Ndh = Nd / (np.exp(-(Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)
        ax.set_xlabel('1-Nde/Ndh')
    else:

        tau_g = 1 / (sigma_h * ni * vth_h * np.exp(-Ed / Vt))
        Nde = Nd / (np.exp((Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)
        ax.set_xlabel('1-Ndh/Nde')

    ax.plot(yi, tau_g + tau_r * tau_g / tau_t * yi, ':', label='HH_s')
    ax.plot(yi, tau_g + tau_r * tau_g / tau_t * yi, ':', label='HH_s')

    ax.set_ylabel('Tau (s)')


def PC_mutli_nU(Ed, sigma_e, sigma_h, Nd, ni, nh0, ne0, temp, tau_BGK):
    '''
    A testing function that assume 3 states and that the middle state has a very low rate of change.
    '''
    Vt = C.k * temp / C.e
    vth_e, vth_h = tm.update(temp=temp)

    eh = sigma_h * vth_h * ni * np.exp(-Ed / Vt)
    ee = sigma_e * vth_e * ni * np.exp(Ed / Vt)

    ch = nh0 * sigma_h * vth_h
    ce = ne0 * sigma_e * vth_e

    a1 = eh[0] + ce[0]
    a2 = eh[1] + ce[1]
    b2 = ee[0] + ch[0]
    b3 = ee[1] + ch[1]

    if b3 / (a2 + b2) > 0.1:
        print('issue b4')
        if(ee[1] > ch[1]):
            print('electron emission is to big')
            print(ee[1], Ed[1])
        else:
            print('capture is to big')

    elif a1 / (a2 + b2) > 0.1:
        print('issue a1')
        if(eh[0] > ce[0]):
            print('emission is to big')
        else:
            print('capture of electrons is to big')

    # else:
        # print(b3 / (a2 + b2), a1 / (a2 + b2))
    if a2 > b2:
        print(a2 / b2, 'should be larger than 1')
        if eh[1] > ce[1]:
            print('emission of hole from the upper level', eh[1])
        else:
            print('capture of electron from the upper level')
    else:
        if ee[0] > ch[0]:
            print('capture of electron from the lower level')
        else:
            print('emission of hole from the lower level')

    # print('changed')
    tau1 = (a1 * a2 + b2 * b3) / (a2 + b2)
    tau2 = a1
    return 1. / tau1, 1. / tau2


def HH_f_plot(ax, Ed, sigma_e, sigma_h, Nd, ni, nh0, ne0, temp, tau_BGK):

    vth_e, vth_h = tm.update(temp=temp)
    Vt = C.k * temp / C.e

    yi = np.linspace(0, 1)
    Ef = np.log(ni / nh0) * Vt

    tau_r = tau_BGK

    if nh0 > ne0:

        tau_g = 1 / (sigma_e * ni * vth_e * np.exp(Ed / Vt))
        Ndh = Nd / (np.exp(-(Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)
        ax.set_xlabel('1-Nde/Ndh')
        tau_b = 1 / (sigma_h * vth_h * nh0)

    else:

        tau_g = 1 / (sigma_h * ni * vth_h * np.exp(-Ed / Vt))
        Nde = Nd / (np.exp((Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)
        ax.set_xlabel('1-Ndh/Nde')
        tau_b = 1 / (sigma_e * vth_e * ne0)

    HH_s = tau_g + tau_r * tau_g / tau_t * yi
    itau_HH_f = 1 / tau_b + 1 / HH_s

    ax.plot(yi, 1 / itau_HH_f, ':', label='HH_f')

    ax.set_ylabel('Tau (s)')


def HH_f_plot2(ax, Ed, sigma_e, sigma_h, Nd, ni, nh0, ne0, temp, tau_BGK):

    vth_e, vth_h = tm.update(temp=temp)
    Vt = C.k * temp / C.e

    yi = np.linspace(0, 1)
    Ef = np.log(ni / nh0) * Vt

    tau_r = tau_BGK

    if nh0 > ne0:

        tau_g = 1 / (sigma_e * ni * vth_e * np.exp(Ed / Vt))
        Ndh = Nd / (np.exp(-(Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)
        ax.set_xlabel('1/(1-Ndh/Nde)')
        tau_b = 1 / (sigma_h * vth_h * nh0)

    else:

        tau_g = 1 / (sigma_h * ni * vth_h * np.exp(-Ed / Vt))
        Nde = Nd / (np.exp((Ed - Ef) / Vt) + 1)
        tau_t = 1. / (sigma_e * vth_e * Nd)
        ax.set_xlabel('1/(1-Ndh/Nde)')
        tau_b = 1 / (sigma_e * vth_e * ne0)

    HH_s = tau_g + tau_r * tau_g / tau_t * yi
    itau_HH_f = 1 / tau_b + 1 / HH_s

    ax.plot(1 / yi, itau_HH_f, ':', label='HH_f')

    ax.set_ylabel('1/Tau (1/s)')


if __name__ == "__main__":
    import doctest
    doctest.testmod()
