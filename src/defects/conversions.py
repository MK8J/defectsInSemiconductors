#!python3

import numpy as np
import semiconductor.material as mat


def defectParams_from_rates(smp, e_e, e_h, c_e, c_h):
    '''
    from rates calculates the defect paramters

    Parameters
    ----------

    smp : (sample class)
        Uses this to get temperature and sample material concentrations, e.g. intrinsic carrier density, density of conduction band states and density of valance band states, thermal velocity.
    e_e : (float, units 1/s)
        The emission rate of electrons
    e_h : (float, units 1/s)
        The emission rate of holes
    c_e : (float, units 1/s)
        The capture rate of electrons
    c_h : (float, units 1/s)
        The capture rate of electrons

    Returns
    -------

    sigma_e: (float, units cm^2)
        the capture cross section of electrons

    sigma_e: (float, units cm^2)
        the capture cross section of holes

    Ed: (float, units eV)
        the energy level of the defect relative to ni
    '''
    sigma_e, sigma_h, Ed = _defectParams_from_rates(e_e=e_e,
                                                    e_h=e_h,
                                                    c_e=c_e,
                                                    c_h=c_h,
                                                    ni=smp.ni)

    Ed *= smp.Vt
    sigma_e /= smp.vth_e
    sigma_h /= smp.vth_h

    return sigma_e, sigma_h, Ed


def _defectParams_from_rates(e_e, e_h, c_e, c_h, ni, Nc=None, Nv=None):
    '''
    Calculates the defect paramters from the provided rates and coefficients

    Only three need to be provided to calculate the three rates, pass None to the other rate.

    Note c_e and c_h are the capture coefficients:
                    $$\sigma * v_{th}$$
    Parameters
    ----------

    e_e: (float, units 1 / s)
        The emission rate of electrons
    e_h: (float, units 1 / s)
        The emission rate of holes
    c_e: (float, units 1 / s)
        The capture rate of electrons
    c_h: (float, units 1 / s)
        The capture rate of electrons
    ni: (float, units cm ^ -3)
        The intrinisic carrier density
    Nc: (float, units cm ^ -3)
        The density of states in the conduction band
    Nv: (float, units cm ^ -3)
        The density of states in the valance band

    Return
    ------

    sigma_e: (float, units cm ^ 2 / s)
        the product of the capture cross section and thermal velocity

    sigma_e: (float, units cm ^ 2 / s)
        the product of the capture cross section and thermal velocity

    Ed: (float)
        the energy level normalised to the thermal voltage
    '''

    if e_e is None:
        e_e = ni**2 * c_e * c_h / e_h

    elif e_h is None:
        e_h = ni**2 * c_e * c_h / e_e
    elif c_h is None:
        c_h = e_e * e_h / c_e / ni**2
    elif c_e is None:
        c_e = e_e * e_h / c_h / ni**2

    assert np.isclose(c_e * c_h / e_h / e_e, 1 / ni**2), c_e * c_h / e_h / e_e

    sigma_e = c_e
    sigma_h = c_h

    if Nc is not None:
        Ed = np.log(e_e / c_e / Nc)
    elif Nv is not None:
        Ed = -np.log(e_h / c_h / Nv)
    else:
        Ed = -np.log(e_h / c_h / ni)

    return sigma_e, sigma_h, Ed
