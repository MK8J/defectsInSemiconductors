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


def __get_HH_full(tau_r, tau_dr, sigma_m, vth_m, nte, nte0, n_i, Ed, temp):
    '''
    Man there is a lot of defined vairables in this paper. this current;y doesn't work

    Parameters

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

    dt = abs(nte[:] - nte[-1])
    N = abs(nte[0] - nte[-1])
    y = abs(dt) / N

    tau_g = 1 / (sigma_m * vth_m * n_i * np.exp(Ed / Vt))

    print('tau_g: {0:.2e} tau_r {1:.2e} Nsigmavth_m {2:.2e}'.format(
        tau_g, tau_r, N * sigma_m * vth_m))
    # tau_ge = sigma_e * vth_e * n_i np.exp(Ed / Vt)
    # tau_gh = sigma_h * vth_h * n_i np.exp(-Ed / Vt)

    itau_m = 1 / tau_dr + 1 / \
        (tau_g + tau_r * tau_g * N * sigma_m * vth_m * (1 - y))

    # itau_e = 1 / t_bulk + 1 / \
    # (tau_ge + tau_r * tau_g * N * sigma_e * vth_e * (1 - y))

    # itau_h = 1 / t_bulk + 1 / \
    # (tau_gh + tau_r * tau_g * N * sigma_h * vth_h * (1 - y))

    return 1 / itau_m


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


def HornbeckHanes_simplified(Ed, sigma_e, sigma_h, ni, ne0, nh0, temp, tau_bkg):
    '''
    Calculates the time constant from hornbeck and Hanes Simplified model

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

    if nh0 > ne0:
        ne_thing = ne1
        nh_thing = nh1 + nh0
    else:
        ne_thing = ne1 + ne0
        nh_thing = nh1

    Nd = max(nh0, ne0)

    tau_d = 1. / (sigma_e * vth_e * (ne_thing) +
                  sigma_h * vth_h * (nh_thing))

    tau_t = 1. / (sigma_e * vth_e * Nd)

    HH_s = tau_d + tau_bkg * tau_d / tau_t
    return HH_s


if __name__ == "__main__":
    import doctest
    doctest.testmod()
