
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import odeint
from . import defects


def transient_decay(s, nxc, t_stepf=500, t_stepNo=1000, auto=True, nxc_min=1e8, G_ss=0):
    '''
    calculates a transient decay from an inital steady state condition.

    Parameters
    ----------
    s : (sample class)
        the sample class
    nxc : (float)
        The steady state number of excess carrier from which the decay states
    t_stepf : (numebr, optional)
        The step size of the numerical solution compared to the lowest of the SRH and radiative Lifetime
    t_stepNo : (number, optional)
        The number of time steps to take
    auto : (bool)
        If this is true, the program will attempt to solve the case until the nxc_min is reached. This is attempbed by appending sequantial simulations with each new simulations the time step (t_stepf) of the last simulation increased by a multiple.
    nxc_min : (float, default=1e8)
        The excess carrier density at which the transient simulatulation should true and stop. This only has an impact if auto is set to true. the is caculated as the larger of the excess holes or excess electrons.
    G_ss : (float, default = 0)
        Allows for a transient decay to a fixed generation rate.

    Returns
    -------
    ne : (array like)
        concentration of electrons
    nh : (array like)
        concentration of holes
    nd : (array like)
        concentration of defect/defect states
    t : (array like)
        time.
    '''
    assert type(auto) == bool

    # get steady state
    ne0, nh0, nd0 = s.steady_state_concentraions(nxc=nxc)

    def solve(ne=None, nh=None, ncs=None):
        '''
        This is just a little solver that prevents duplication if auto is used.
        '''

        # these are here for the auto function
        # if something is passed used them

        if all(isinstance(dft, defects.MultiLevel) for dft in s.defectlist):
            if ne is None or nh is None:
                ne = ne0
                nh = nh0
                ncs = nd0

            _ne, _nh, _t, _nd = trans_multilevel(
                s, ne=ne, nh=nh, ncs=ncs,
                G_ss=G_ss, t_stepf=t_stepf, t_stepNo=t_stepNo)

        elif all(isinstance(dft, defects.SingleLevel) for dft in s.defectlist):

            if ne is None or nh is None:
                ne = ne0
                nh = nh0
                ncs = [dft.Nd for dft in s.defectlist]

            _ne, _nh, _t, _nd = trans(
                s, ne=ne, nh=nh, nte=ncs,
                G_ss=G_ss, t_stepf=t_stepf, t_stepNo=t_stepNo)
        else:
            print('somethign went wrong', s.defectlist)
            print(all(isinstance(dft, defects.MultiLevel)
                      for dft in s.defectlist))
            print([isinstance(dft, defects.MultiLevel)
                   for dft in s.defectlist])
            pass

        return _ne, _nh, _t, _nd

    # this is to make it easier to solve
    # as you don't have to guess the step size
    # nor the number of points.
    if auto:
        t_stepf = t_stepf

        # try running it
        ne, nh, t, nd = solve()

        # see if we made it
        nef = max(ne[-1] - s.ne0, nh[-1] - s.nh0)

        # if it not enough, go deeper
        # this appends simulations of longer step size onto the current one
        while nef > nxc_min:

            _ne, _nh, _t, _nd = solve(ne[-1], nh[-1], nd[-1])

            ne = np.concatenate((ne, _ne[1:]), axis=0)
            nh = np.concatenate((nh, _nh[1:]), axis=0)
            nd = np.concatenate((nd, _nd[1:]), axis=0)
            t = np.concatenate((t, _t[1:] + t[-1]), axis=0)

            t_stepf *= 5
            nef = max(ne[-1] - s.ne0, nh[-1] - s.nh0)

            if t_stepf > 1e11:
                print('could not reach minimum excess carrier density', nef, nxc_min)
                nxc_min = 1e20

    elif not auto:

        ne, nh, t, nd = solve()

    return ne, nh, nd, t


def steadyState_carriers(s, nxc, ne, nh,  output=None):
    '''
    Calculate the steady state lifetime of a sample for the provide of carrier concentration. This is much faster than steadyState_excesscarriers as it does not ensure charge neutrality.

    Parameters
    ----------
    s : (class)
        the sample class
    nxc : (array like)
        The excess carrier density at which the recombiation is to be evaluated
    ne : (array like)
        The electron density at which the recombiation is to be evaluated
    nh : (array like)
        The hole density at which the recombiation is to be evaluated
    output : (bool default False)
        determines the output. If True provides the carrier densities as well

    Returns
    -------
    gen : (numpy array, in cm^-3)
        generation rate required to obtain the excess carrier density
    tau : (numpy array, in seconds)
        the minoirty carrier lifetime
    ne : (numpy array, in cm^-3, optional)
        the number of free electrons
    nh : (numpy array, in cm^-3, optional)
        the number of free holes

    Examples
    --------

    define a defect

    >>> defect = dict(Ed=[0, -0.35], sigma_e=[3e-14, 1e-16], sigma_h=[3e-15, 1e-15], charge=[[0, -1], [0, 1]], Nd=1e12)

    define the sample properties

    >>> from defects.sample import Sample
    >>> from defects.defects import MultiLevel
    >>> from defects.solvers import steadyState_carriers

    >>> s = Sample()
    >>> s.tau_rad = 1 # a constant bulk lifetime in seconds
    >>> s.Nacc = 0 # number acceptors in cm^-3
    >>> s.Ndon = 1e16 # number donors in cm^-3
    >>> s.temp = 300 # the sample temperature

    This  can be also used with the single level defect class, but here we are just showing the multi level defect

    >>> s.defectlist = MultiLevel(Ed=defect['Ed'], sigma_e=defect['sigma_e'], sigma_h=defect['sigma_h'], Nd=defect['Nd'], charge=defect['charge'])

    >>> nxc = 1e13
    >>> gen,tau = steadyState_carriers(s, nxc, 1e16, 1e13, output=False)
    >>> print('gen = {0:.2e}cm^-3 \t tau = {1:.2e}s'.format(gen[0], tau[0]))
    gen = 5.06e+17cm^-3        tau = 1.97e-05s

    '''
    if isinstance(nxc, float):
        nxc = np.array([nxc])

    if isinstance(ne, float):
        ne = np.array([ne])

    if isinstance(nh, float):
        nh = np.array([nh])

    qEfe = np.log(ne / s.ni) * s.Vt
    qEfh = -np.log(nh / s.ni) * s.Vt

    rec = np.zeros(nxc.shape)
    for i in s.defectlist:
        rec += i.recombination_SS(qEfe, qEfh, s.temp, s.ni)

    rec += nxc / s.tau_rad

    if hasattr(s.defectlist[0], 'recombination_SS_ateachlevel'):
        lr = s.defectlist[0].recombination_SS_ateachlevel(
            qEfe, qEfh, s.temp, s.ni)

    if output:
        return rec, nxc / rec, ne, nh
    else:
        return rec, nxc / rec


def steadyState_excesscarriers(s, nxc, plot_carriers=True,  plot_lifetime=True, output=None):
    '''
    Calculates the steady state lifetime of a sample, give a specific defect given the concentration of excess carriers. This ensures at each step that neutrality is ensured.

    Parameters
    ----------
    s : (class)
        the sample class
    nxc : (array like)
        The excess carrier density to be evaluated
    plot_carriers : (bool default  True)
        determines if the function automatically plots the carriers with time
    plot_lifetime : (bool default  True)
        determines if the function automatically plots the lifetime as a function of excess carriers
    output : (bool default False)
        determines the output. If True provides the carrier densities as well

    Returns
    -------
    gen : (numpy array, in cm^-3)
        generation rate required to obtain the excess carrier density
    tau : (numpy array, in seconds)
        the minoirty carrier lifetime
    ne : (numpy array, in cm^-3, optional)
        the number of free electrons
    nh : (numpy array, in cm^-3, optional)
        the number of free holes
    nd : (numpy array, in cm^-3, optional)
        the number of defect states

    Examples
    --------

    >>> from defects.sample import Sample
    >>> from defects.defects import MultiLevel
    >>> from defects.solvers import steadyState_excesscarriers

    define a defect

    >>> defect = dict(Ed=[0, -0.35], sigma_e=[3e-14, 1e-16], sigma_h=[3e-15, 1e-15], charge=[[0, -1], [0, 1]], Nd=1e12)

    define the sample properties

    >>> s = Sample()
    >>> s.tau_rad = 1 # a constant bulk lifetime in seconds
    >>> s.Nacc = 0 # number acceptors in cm^-3
    >>> s.Ndon = 1e16 # number donors in cm^-3
    >>> s.temp = 300 # the sample temperature

    This  can be also used with the single level defect class, but here we are just showing the multi level defect

    >>> s.defectlist = MultiLevel(Ed=defect['Ed'], sigma_e=defect['sigma_e'], sigma_h=defect['sigma_h'], Nd=defect['Nd'], charge=defect['charge'])


    >>> nxc = 1e13
    >>> gen,tau = steadyState_excesscarriers(s, nxc, plot_carriers=False,   plot_lifetime=False)
    >>> print('gen = {0:.2e}cm^-3 \t tau = {1:.2e}s'.format(gen[0], tau[0]))
    gen = 5.06e+17cm^-3        tau = 1.97e-05s
    >>> plt.cla()
    '''

    if not isinstance(nxc, np.ndarray):
        nxc = np.array([nxc])

    nd = None
    ne0, nh0 = s.equlibrium_concentrations()
    ne0, nh0, nd0 = s.steady_state_concentraions([0])
    ne, nh, nd = s.steady_state_concentraions(nxc=nxc)

    # define the generation
    nd = nd.reshape((ne.shape[0], nd.shape[0] // ne.shape[0]))

    if plot_carriers:
        plt.figure('a')
        plt.plot(nh - nh0, ne - ne0, label='electonrs')
        plt.plot(nh - nh0, nh - nh0, '--', label='holes')
        plt.plot(nh - nh0, nd, '--', label='defects')
        # plt.plot(nh - nh0, nd, ':', label='defect!')
        plt.loglog()
        plt.xlabel('Number of excess carriers')
        plt.ylabel('Number of carriers')
        plt.legend(loc=0)

    qEfe = np.log(ne / s.ni) * s.Vt
    qEfh = -np.log(nh / s.ni) * s.Vt

    rec = np.zeros(nxc.shape)
    for i in s.defectlist:
        rec += i.recombination_SS(qEfe, qEfh, s.temp, s.ni)

    rec += nxc / s.tau_rad

    if hasattr(s.defectlist[0], 'recombination_SS_ateachlevel'):
        lr = s.defectlist[0].recombination_SS_ateachlevel(
            qEfe, qEfh, s.temp, s.ni)

    if plot_lifetime:
        plt.figure('life')
#         plt.plot(nxc, (nh - nh0) / rec, 'r.-')
#         plt.plot(nxc, (ne - ne0) / rec, 'g.-')
        plt.plot(nxc, (nxc) / rec, 'b--')
        # plt.plot(nh - nh0, ((nh - nh0) / lr.T).T, '--')
        # plt.plot(nh - nh0, (nh - nh0) / rec)

        plt.xlabel('excess carrier density (cm^-3)')
        plt.loglog()

    if output:
        return rec, nxc / rec, ne, nh, nd
    else:
        return rec, nxc / rec


def squareWavePulse(s, t_stepf=500, t_stepNo=1000, Gss=1e20, plot_carriers=True,   plot_lifetime=True):
    '''
    This is a function to determine what happens to a semiconductors in 0D when
    illumination with a square wave of light.
    This runs a square wave pulse, cals the components and cals lifeimte

    This is basiccally a wrapper around trans_multilevel(), to make this specific simulations easier.

    Parameters
    ----------
    s : class
        the sample class
    t_stepf : (float, defualt 500, uniltess)
        The time step taking in ht numerical simulations. It is the ratio to the minoirty carrier Lifetime
    t_stepNo : (float, default=1000, unitless)
        The number of time steps taken
    Gss : (float, defualy=1e20, photons)
        the illumination intensity.
    plot_carriers : (bool, default=True)
        creats a plot of the carriers with time
    plot_lifeimte : (bool, default=True)
        creates a plot of lifetime

    Returns
    -------
    ne : (1D, array like)
        number of electrons for each time step
    nh : (1D, array like)
        number of holes for each time step
    nd : (nD, array like)
        occupation of defects in the posisble states. The n comes from the number of states in a defect.
    t : (array like)
        the time

    '''
    # put in it the simulation
    # s._defectlist = []
    # s.defectlist = MultiLevel(Ed=defect['Ed'],
    #                                 sigma_e=defect['sigma_e'],
    #                                 sigma_h=defect['sigma_h'],
    #                                 charge=defect['charge'],
    #                                 Nd=defect['Nd'])

    # get the dark carrier density
    ne0, nh0 = s.equlibrium_concentrations()
    # print('ne0 {0:.2e}, nho {1:.2e}'.format(ne0, nh0))

    ncs = s.defectlist[0].charge_state_concentration(s.Ef, s.temp)

    # solve under light
    ne, nh, t, other = trans_multilevel(
        s, ne=ne0, nh=nh0, ncs=ncs, G_ss=Gss,
        t_stepf=t_stepf, t_stepNo=t_stepNo)

    if plot_carriers:
        plt.figure('carriers')
        p1, = plt.plot(t, ne)
        p2, = plt.plot(t, nh)
        p = plt.plot(t, other, '--')

    # solve for dark
    t0 = t[-1]
    nef = 1e20
    t_stepf = 1000

    min_val = max(min(ne0, nh0) / 100, 1e6)
    min_val = 1e8

    while nef - min(ne0, nh0) > min_val:
        _ne, _nh, _t, _other = trans_multilevel(
            s, ne=ne[-1], nh=nh[-1], ncs=other[-1, :],
            G_ss=0, t_stepf=t_stepf, t_stepNo=t_stepNo)

        nef = min(_ne[-1], _nh[-1])
        t_stepf *= 5
        t_stepNo *= 5

    ne, nh, t, other = _ne, _nh, _t, _other

    if plot_carriers:
        plt.figure('carriers')
        plt.plot(t + t0, ne, '-', c=p1._color)
        plt.plot(t + t0, nh, '-', c=p2._color)

        for c, p0 in enumerate(p):
            plt.plot(t + t0, other[:, c], '--', c=p0._color)

    if plot_carriers:
        plt.figure('carriers')
        plt.plot(t[-1] + t0, ne0, 'o', c=p1._color)
        plt.plot(t[-1] + t0, nh0, 'o', c=p2._color)
        plt.loglog()

        for c, p0 in enumerate(p):
            plt.plot(t[-1] + t0, ncs[c], 'o', c=p0._color)
        plt.xlabel('time (s)')
        plt.ylabel('carrier density (cm^-3)')
    if plot_lifetime:

        # cal lifetime for the decay
        # dn_pc = (ne + nh - ne[-1] - nh[-1]) / 2
        dn_pc_n0 = (ne + nh - ne0 - nh0) / 2
        # cal PL
        pl = (ne * nh - ne0 * nh0)
        dn_pl = (-s.doping + np.sqrt(s.doping**2 + 4 * pl)) / 2

        print('Error in finial value e',
              (ne0 - ne[-1]) * 100 / ne0, ' h', (nh0 - nh[-1]) * 100 / nh0)

        plt.figure('lifetime')
        plt.plot(ne - ne0, -(ne - ne0) /
                 np.gradient(ne - ne0, t), label='Electrons')
        plt.plot(nh - nh0, -(nh - nh0) /
                 #                  np.gradient(nh - nh0, t), label='holes')
                 #         plt.plot(nh - nh0, -(nh) /
                 np.gradient(nh, t), label='holes')
        plt.legend(loc=0)
        plt.loglog()
        # skip = dn_pc_n0.shape[0] // 1000
        # p1 = plt.plot(dn_pc_n0[::skip], (dn_pc_n0)[::skip] /
        #               (0 - np.gradient(dn_pc_n0, t))[::skip], '.', label='PC')
        # p1 = plt.plot(dn_pl[::skip], (dn_pl)[::skip] /
        #               (0 - np.gradient(dn_pl, t))[::skip], ':', label='PL')
        plt.xlabel('excess carrier density (cm^-3)')
        plt.ylabel('lifetime (s)')
        # plt.loglog()
        # plt.ylim(top=1e-3)
        # plt.xlim(left=1e5)
        # plt.legend(loc=0)

    return ne, nh, other, t


def trans(sample, ne, nh, nte, G_ss=1e22, t_stepf=2000, t_stepNo=10000):
    '''
    A function that calculates the carrier density with time, under transient conditions.
    You need to provide the carrier density at which the simulations starts, and the
    illumination intensity under which the carriers being subjected to.

    This function allows easy passing of the samples class.

    Parameters
    ----------
    sample: (class)
        An instance of the sample class
    ne: (float)
        the inital number of free electrons
    nh: (float)
        the inital number of free holes
    nte: (float)
        the inital number of electrons in the defect
    G_ss: (float)
        The generation rate at which the decay ends at. This assumes the sample
        is in steady state at the start of the decay.
    t_stepf: (float)
        t_stepf is the mutlipler to the smallest lifetime that is used as the length of the simulation.

    Returns
    -------
    ne: (array)
        the number of free electrons with time
    nh: (array)
        the number of holes electrons with time
    t: (array)
        the time
    nte: (array)
        the electrons in the traps
    '''

    return _SRH_trans(ne=ne, nh=nh, nte=nte, Nacc=sample.Nacc,
                      Ndon=sample.Ndon,
                      ni=sample.ni,
                      temp=sample.temp,
                      dft_list=sample.defectlist, bkg_tau=sample.tau_rad,
                      G_ss=G_ss, t_stepf=t_stepf, t_stepNo=t_stepNo,
                      ne0=sample.ne0, nh0=sample.nh0)


def trans_multilevel(sample, ne, nh, ncs, G_ss=1e22, t_stepf=2000, t_stepNo=10000):
    '''
    A function that calculates the carrier density with time, under transient conditions.
    You need to provide the carrier density at which the simulations starts, and the
    illumination intensity under which the carriers being subjected to.

    This function allows easy passing of the samples class.

    Parameters
    ----------
    sample: (class)
        An instance of the sample class
    ne: (float)
        the inital number of free electrons
    nh: (float)
        the inital number of free holes
    ncs: (float)
        the fraction of each charge state
    G_ss: (float)
        The generation rate at which the decay ends at. This assumes the sample
        is in steady state at the start of the decay.
    t_stepf: (float)
        t_stepf is the mutlipler to the smallest lifetime that is used as the length of the simulation.

    Returns
    -------
    ne: (array)
        the number of free electrons with time
    nh: (array)
        the number of holes electrons with time
    t: (array)
        the time
    ncs: (array)
        the charge state of the defect
    '''

    return _SRH_trans_multi(ne=ne, nh=nh, ncs=ncs, Nacc=sample.Nacc,
                            Ndon=sample.Ndon, ni=sample.ni, temp=sample.temp,
                            dft_list=sample.defectlist, bkg_tau=sample.tau_rad,
                            G_ss=G_ss, t_stepf=t_stepf,  t_stepNo=t_stepNo, ne0=sample.ne0, nh0=sample.nh0)


def _SRH_trans(ne, nh, nte,  Nacc, Ndon, ni, ne0, nh0, temp, dft_list, bkg_tau, G_ss=1e22, t_stepf=2000, t_stepNo=10000):
    '''
    Solve the excess carrier density with time. It assumes the

    Parameters
    ----------
    ne: (float)
        the inital number of electonrs
    nh: (float)
        the inital number of holes
    nte:
        the inital number of electrons in the defect
    Na: (float)
        the number of ionised acceptors
    Nd: (float)
        the number of ionised donors
    ni: (float)
        The intrinsic carrier density
    temp: (float, Kelvin)
        the temperature
    dft_list: (list)
        A defect list from the srh class.
    bkg_tau: (float, s)
        A constant background recombiation Lifetime
    G_ss: (float)
        The generation rate to which the transient condition converges
    t_stepf: (float)
        a the mutlipler that controls the size of time step. Change this value is things look strange.
    t_stepNo: (float)
        number of time steps

    '''

    def SRH_fitting(y, t, G):
        '''
        Parameters
        ----------
            y : [ne, nh, nte]
            t : time
            G : generation rate

        '''

        # grab the electron and hole conc, the defect conc is grabbed below.
        ne, nh = y[0], y[1]

        # Adjust the generation by the recombiation
        if bkg_tau is not None:
            G -= (ne - ne0) / bkg_tau

        # for no defects there is no change.
        emitted_e = []
        emitted_h = []
        captured_e = []
        captured_h = []

        # iterate over all the defects in the list
        for i, dft in enumerate(dft_list):
            nte = y[i + 2]
            emitted_e.append(dft.emitted_e(nte, ni, temp))
            emitted_h.append(dft.emitted_h(nte, ni, temp))

            captured_e.append(dft.capture_e(ne, nte))
            captured_h.append(dft.capture_h(nh, nte))
        # all those changes result in
        dne = G - sum(captured_e) + sum(emitted_e)
        dnh = G - sum(captured_h) + sum(emitted_h)

        dydt = [dne, dnh]

        # put the change in the defects into one thingo
        for i in range(len(emitted_e)):
            dydt.append(- captured_h[i] + emitted_h[i] +
                        captured_e[i] - emitted_e[i])

        return dydt

    # build the input vectors

    # get all the recombiation sources
    _temp_list = [bkg_tau]

    for dft in dft_list:
        if Nacc > Ndon:
            _temp_list.append(dft.tau_hmin)

        else:
            _temp_list.append(dft.tau_emin)

    _tau = min(_temp for _temp in _temp_list if _temp is not None)

    # an array of time.
    t = np.linspace(0, _tau * t_stepf, t_stepNo)

    # set the inital conditions
    y0 = [np.array(ne, dtype=np.float64), np.array(nh, dtype=np.float64)]
    # now add the defects
    for dft in nte:
        y0.append(dft)

    # Solve for the steady state illumination condition
    # sol0 = solve_ivp(SRH_fitting, t, y0, rtol=0.0000001)
    # print(type(y0), type(t), type(G_ss))
    # print(y0)
    sol0 = odeint(SRH_fitting, y0, t,  args=(G_ss,), rtol=0.0000001)

    ne, nh = sol0[:, 0], sol0[:, 1]

    return ne, nh, t, sol0[:, 2]


def _SRH_trans_multi(ne, nh, ncs,  Nacc, Ndon, ni, temp, dft_list, bkg_tau, ne0, nh0, G_ss=1e22, t_stepf=2000, t_stepNo=10000):
    '''
    Solve the excess carrier density with time. It assumes the

    Parameters
    ----------
    ne : (float)
        the inital number of electonrs
    nh: (float)
        the inital number of holes
    ncs: (array)
        the number of defects in each charge state
    Nacc: (float)
        the number of ionised acceptors
    Ndon: (float)
        the number of ionised donors
    ni: (float)
        The intrinsic carrier density
    temp: (float, Kelvin)
        the temperature
    dft_list: (list)
        A defect list from the srh class.
    bkg_tau: (float, s)
        A constant background recombiation Lifetime
    G_ss: (float)
        The generation rate to which the transient condition converges
    t_stepf: (float)
        a the mutlipler that controls the size of time step. Change this value is things look strange.
    t_stepNo: (float)
        number of time steps

    '''

    # def jac_de(y, t):
    #     '''
    # The dream of getting a jocobian done.
    #     Currently this only works for a single defect, thought that defect can have multiple levels.
    #
    #     The jacobian is a bunch of partial derivatives
    #
    #     J = [[df0/dx0 .. df0/dxm]..[dfn/dxm..dfn/dxm]]
    #
    #     Here that looks like:
    #
    #     $$d/dn_e dn_e/dt = d/dn_e( G +\Sum_{E_d} e_e - c_e)$$
    #     $$d/dn_e dn_e/dt =  - \Sum_{E_d} \frac{N_{di}}{sigma_e v_{the}} $$
    #
    #     Similarly:
    #     $$d/dn_h dn_e/dt =  0 $$
    #     $$d/dN_d dn_e/dt =   \Sum_{E_d} \Beta_e-\frac{n_{e}}{sigma_e v_{the}} $$
    #
    #     $$d/dne dn_h/dt = 0$$
    #     $$d/dnh dn_h/dt = - \Sum_{N_d} \frac{N_{di}}{sigma_h v_{thh}} $$
    #     $$d/dN_d dn_h/dt = \Sum_{E_d} -\Beta_h+\frac{n_h}{sigma_h v_{thh}} $$
    #     '''
    #     dft = dft_list[0]
    #     j = 0
    #     i = 0  # this needs to be changed.
    #
    #     _ncs = np.copy(y[i + 2 + j:j + i + 2 + len(dft.Ed) + 1])
    #
    #     Nde = dft.emitted_e(_ncs / _ncs, ni, temp) - \
    #         dft.capture_e(_ne, _ncs / _ncs)
    #     Ndh = dft.emitted_h(_ncs / _ncs, ni, temp) - \
    #         dft.capture_h(_ne, _ncs / _ncs)
    #
    #     jac = np.array([[np.sum(dft.capture_e(1, _ncs)), 0] + Nde,
    #                     [0, np.sum(dft.capture_h(1, _ncs))] + Ndh])
    #
    #     jac_ = np.zeros((len(Ndh), len(Ndh)))
    #     jac_[:, 0] = Nde
    #     jac_[:, 1] = Ndh
    #     for i in range(len(Ndh)):
    #         jac_[i, 2 + i] = 1
    #
    #     return jac
    #     dft.emitted_e(_ncs / _ncs, ni, temp) - dft.capture_e(_ne, _ncs / _ncs)
    #     return

    def SRH_fitting(y, t, G):
        '''
        Parameters
        ----------
            y = [ne, nh, rcs]
            t is time
            G is generation rate

        $$dne/dt = G - c_e + e_e$$
        $$dnh/dt = G - c_h + e_h$$
        $$dN_{di}/dt = c_e - e_e - c_h + e_e$$
        '''

        # for no defects there is no change.
        emitted_e = []
        emitted_h = []
        captured_e = []
        captured_h = []

        emitted_e = np.array([])
        emitted_h = np.array([])
        captured_e = np.array([])
        captured_h = np.array([])

        # grab the electron and hole conc, the defect conc is grabbed below.
        _ne, _nh = y[0], y[1]

        # Adjust the generation by the recombiation
        # this isn't quite right and wont work for intrinsic material
        if bkg_tau is not None:
            if ne0 > nh0:
                G -= (_nh - nh0) / bkg_tau
            else:
                G -= (_ne - ne0) / bkg_tau

        # iterate over all the defects in the list
        counter = 0
        # dncs_list = np.zeros((y.shape[0] - 2))
        dncs = np.array([])

        for i, dft in enumerate(dft_list):
            _ncs = np.copy(
                y[2 + i + counter:2 + counter + i + len(dft.Ed) + 1])

            # calculate the emitted carriers
            _emitted_e = dft.emitted_e(_ncs, ni, temp)
            _emitted_h = dft.emitted_h(_ncs, ni, temp)

            # calculate the captured carriers
            _captured_e = dft.capture_e(_ne, _ncs)
            _captured_h = dft.capture_h(_nh, _ncs)

            # make these into a running array
            captured_e = np.concatenate((
                captured_e, _captured_e))
            captured_h = np.concatenate((
                captured_h, _captured_h))

            # make these into a running array
            emitted_e = np.concatenate((
                emitted_e, _emitted_e))
            emitted_h = np.concatenate((
                emitted_h, _emitted_h))

            # adjust for starting length in i for non multi level defects
            counter += len(dft.Ed)  # - 1

            # calculate where the defects went
            moveup = _captured_e + _emitted_h
            movedown = _emitted_e + _captured_h

            # this has to be done here, to ensure we do not have movement between
            # independent defects
            _dncs = np.zeros(_ncs.shape)
            _dncs[:-1] += -moveup + movedown
            _dncs[1:] += moveup - movedown

            # This results in a total change:
            dncs = np.concatenate((dncs, _dncs))

        # all those changes result in
        dne = G - np.sum(captured_e) + np.sum(emitted_e)
        dnh = G - np.sum(captured_h) + np.sum(emitted_h)

        # calculate the gradient
        dydt = np.array([dne, dnh])
        dydt = np.concatenate((dydt, dncs))

        return dydt

    # get all the recombiation sources to estimate the time step
    _temp_list = [bkg_tau]

    for dft in dft_list:
        if Nacc < Ndon:
            for i in dft.tau_hmin:
                _temp_list.append(i)

        else:
            for i in dft.tau_emin:
                _temp_list.append(i)

    # this is our estimate critical time
    _tau = min(_temp for _temp in _temp_list if _temp is not None)
    # an array of time.
    t = np.linspace(0, _tau * t_stepf, t_stepNo)

    # ToDO:
    # make the time spacing log spaced as the decays are exponential like.
    # an example would be:
    # t = np.logspace(np.log10(_tau / 1000),
    # np.log10(_tau * 100 * t_stepf), t_stepNo)
    # t[0] = 0
    # however the start point here is the issue. How close do you start to the
    # minimum tau? This appears to work well and maybe another function should
    # be formed.

    # set the inital conditions
    y0 = [np.array(ne, dtype=np.float64), np.array(nh, dtype=np.float64)]

    # now add the ratio of charge states. Need to weight it by the number of traps
    # so the ODE function has an equal error wieght on it.
    for i in ncs:
        y0.append(i)

    # Solve for the steady state illumination condition
    sol0, other = odeint(SRH_fitting, y0, t,  args=(
        G_ss,), rtol=0.0000001, full_output=True)
    # uncomment below to see if it is stiff or not (2 is stiff)
    # print(other['mused'])

    ne, nh = sol0[:, 0], sol0[:, 1]

    return ne, nh, t, sol0[:, 2:]
