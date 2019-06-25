
import numpy as np
import scipy.constants as C


class SingleLevel():
    '''
    This class represents a Shockley Read Hall defect with two states. This is also know as a single energy level.

    A defect has paramters:
        1. Ed, the energy level of the defect from midgap.
        2. sigma_h or tau_h. The capture cross section for holes, or the minimum hole lifetime, respecitvly.
        3. sigma_n or tau_e  The capture cross section for electrons, or the minimum electron lifetime, respecitvly.

    Other things that are important are:
        1. the concentration of the defect (Nd)
        2. the thermal velocities (vth_h, vth_e)
        3. The charge on the defect (occupied_charge, unoccupied_charge)

    All these are able to be set at the initialsation of the defect or later.

    '''

    def __init__(self, Ed=None, tau_emin=None, tau_hmin=None, sigma_e=None, sigma_h=None, Nd=1e12, vth_h=1.69e7, vth_e=2.05e7, occupied_charge=-1, unoccupied_charge=0, **kwargs):
        '''
        Initalised the class
        '''

        # initiate the vairables
        self.Nd = Nd
        self.vth_h = vth_h
        self.vth_e = vth_e

        self.occupied_charge = occupied_charge
        self.unoccupied_charge = unoccupied_charge

        self.__tau_hmin = None
        self.__tau_emin = None
        self.__sigma_h = None
        self.__sigma_e = None

        self.Ed = Ed

        # set them form the values that were passed
        self.__attrs(kwargs)

        # these must be set after
        self.tau_hmin = tau_hmin
        self.tau_emin = tau_emin

        # self.Ed = Ed

        self.sigma_h = sigma_h
        self.sigma_e = sigma_e

    def __attrs(self, dic):
        '''
        sets the values in provided dictionary
        '''
        assert type(dic) == dict
        for key, val in dic.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def _e_emission_coef(self, ni, temp):
        '''
        the electron emission coefficient
        '''
        Vt = C.k * temp / C.e
        # print(Vt, self.sigma_e, self.vth_e, ni, self.Ed)
        return self.sigma_e * self.vth_e * ni * np.exp(self.Ed / Vt)

    def _h_emission_coef(self, ni, temp):
        '''
        the hole emission coefficient
        '''
        Vt = C.k * temp / C.e
        return self.sigma_h * self.vth_h * ni * np.exp(-self.Ed / Vt)

    def net_charge(self, Ef, temp):
        '''
        returns the net charge of the defect for a given temperature and
        energy level

        Parameters
        ----------
        Ef : (float or list of 2 floats, eV)
            The Fermi energy level or list of quasi fermi energy level.
            If a list is provided it is assumed the electron quasi Fermi
            energy level is first
        temp : (float, kelvin)
            temperature of the sample

        Examples
        --------

        We can see the change on a defect we now define. Note this is an unusual defect as it has a possible charges of +1 or -1. This is just used now as it makes the example simplier

        As a function of the Fermi energy level. It should be negative
        the higher the fermi energy level

        >>> s = SingleLevel(Ed=0, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12, occupied_charge=-1, unoccupied_charge=1)
        >>> print('{0:.2e}'.format(s.net_charge(0.3,300)))
        -1.00e+12

        >>> print('{0:.2e}'.format(s.net_charge(-0.5,300)))
        1.00e+12

        It should be 0 when Ed-Ef

        >>> print('{0:.2f}'.format(s.net_charge(0,300)))
        0.00
        '''
        ncs = self.charge_state_concentration(Ef, temp)
        charge = ncs[0] * self.unoccupied_charge + \
            ncs[1] * self.occupied_charge

        return(float(charge))

    def charge_state_concentration(self, Ef, temp):
        '''
        Retruns the concentration of the the charge states for a defect for specific electron and hole fermy energy levels. The states are reported with the more "positive" levels first.

        This is equilivant to DOI: 10.1063/1.4906465

        Parameters
        ----------
            Ef : (float or list of 2 floats, [Efe, Efh], eV)
                The Fermi energy level or list of quasi fermi energy level.
                If a list is provided it is assumed the electron quasi Fermi
                energy level is first
            temp : (float, kelvin)
                temperature of the sample

        Examples
        --------

        for a defect in the lower half of the band gap in an n-type material it should be full of electrons. So we get the first state empty, and the second state full

        >>> s = SingleLevel(Ed=-0.2, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print(s.charge_state_concentration(0.3, 300))
        [3.98443584e+03 9.99999996e+11]

        if we reverse the defect and fermi level, we reverse the occupation

        >>> s = SingleLevel(Ed=0.2, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print(s.charge_state_concentration(-0.3, 300))
        [9.99999996e+11 3.98443584e+03]

        for a defect level equal to the fermi energy level the occupations is 50%.

        >>> s = SingleLevel(Ed=0.2, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print(s.charge_state_concentration(0.2, 300))
        [5.e+11 5.e+11]

        Finially we can also provide both a hole and electron fermi energy level

        >>> s = SingleLevel(Ed=0.2, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print(s.charge_state_concentration([0.4,0.2], 300))
        [4.36472821e+08 9.99563527e+11]
        '''

        ratio = np.cumprod(np.append(1, self.occupation_ratio_SS(Ef, temp)))
        return np.array(ratio / ratio.sum() * self.Nd)

    def occupation_ratio_SS(self, Ef, temp):
        '''
        The occupation ratio of the negitive to positive defect charge state
        This works for the general case and is the same as in Sah and Shockley 1958 paper.

        This is equilivant to inver alpha from DOI: 10.1063/1.4906465... well its very similar. But this is correct.

        Parameters
        ----------
        Ef : (float or list of 2 floats, eV)
            The Fermi energy level or list of quasi fermi energy level.
            If a list is provided it is assumed the electron quasi Fermi
            energy level is first
        temp : (float, kelvin)
            temperature of the sample


        Example
        -------

        When Ef = Ed the occupation ratio should be 1, i.e. the defects occupy the same number

        >>> s = SingleLevel(Ed=0., sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12, vth_e=1e7, vth_h=1e7)
        >>> print('n_de = {0:.2f}'.format(s.occupation_ratio_SS(0, 300)))
        n_de = 1.00


        The same result occurs for symmetric capture cross sections, thermal velocities, and symmetric quasi Fermi energy levels and a defect at midgap, if a symmetric splitting occurs we get

        >>> print('n_de = {0:.2f}'.format(s.occupation_ratio_SS([0.3,-0.3], 300)))
        n_de = 1.00


        If we move away from midgap, this doesn't hold

        >>> s = SingleLevel(Ed=0.2, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print('n_de = {0:.2f}'.format(s.occupation_ratio_SS([0.3,-0.3], 300)))
        n_de = 1.18

        But if we again make the splitting even around the defect, it works!

        >>> print('n_de = {0:.0f}'.format(s.occupation_ratio_SS([0.4,-0.2], 300)))
        n_de = 1255
        '''
        k = self.sigma_e / self.sigma_h * self.vth_e / self.vth_h

        if type(Ef) is list and len(Ef) == 2:
            qEfe = Ef[0]
            qEfh = Ef[1]
        else:
            qEfe = Ef
            qEfh = Ef

        Vt = C.k * temp / C.e

        return (
            1 + k * np.exp((qEfe + self.Ed) / Vt)
        ) / (
            k + np.exp((-qEfh - self.Ed) / Vt)
        ) * np.exp(-2 * self.Ed / Vt)
#

    def recombination_SS(self, qEfe, qEfh, temp, ni):
        '''
        This calculates the recombiation occuring through a defect in steady state.

        Parameters
        ----------
        qEfe : (float eV)
            The quasi Fermi energy level of electons
        qEfh : (float eV)
            The quasi Fermi energy level of holes

        temp : (float, kelvin)
            temperature of the sample
        ni : (float, cm^-3)
            the intrinsic carrier density


        Notes
        -----

        The emission and capture rates are the same as a shockley read hall defect.

        This uses the Sah and Shockley formalisation, eqn 3.19 in 10.1103/PhysRev.109.1103 but with one level, which is the same as
        Shockley dx.doi.org/10.1103/PhysRev.87.835.

        Examples
        --------
        We can get this value by either providing a float
        >>> s = SingleLevel(Ed=0.2, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print('{0:.2e}'.format(s.recombination_SS(0.3,-0.3, 300, 1e10)))
        1.00e+20

        Or a list of fermi energy levels
        if we reduce the capture cross sections we get less recombiation
        >>> s = SingleLevel(Ed=0, sigma_e=1e-15, sigma_h=1e-15, Nd = 1e12)
        >>> print(s.recombination_SS([0.3,0],[0,-0.3], 300, 1e10))
        [1.68994373e+14 2.04991721e+14]
        '''
        if not isinstance(qEfe, np.ndarray):
            qEfe = np.array([qEfe])
        if not isinstance(qEfh, np.ndarray):
            qEfh = np.array([qEfh])

        ne = ni * np.exp(qEfe * C.e / C.k / temp)
        nh = ni * np.exp(-qEfh * C.e / C.k / temp)

        ne1 = ni * np.exp(self.Ed * C.e / C.k / temp)
        nh1 = ni * np.exp(-self.Ed * C.e / C.k / temp)

        tau_e = 1 / self.vth_e / self.sigma_e
        tau_h = 1 / self.vth_h / self.sigma_h

        _ne = np.zeros((ne.shape[0])) + ne
        _nh = np.zeros((nh.shape[0])) + nh
        _ni = np.zeros((nh.shape[0])) + ni

        # SRH recombiation
        U = self.Nd * (_ne * _nh - ni**2) / (
            (_nh + nh1) * tau_e + (_ne + ne1) * tau_h)

        # suming over all the defect state recombiation
        if U.shape[0] == 1:
            U = U[0]

        return U

    def emitted_e(self, nde, ni, temp):
        '''
        The emission rate of electrons

        Parameters
        ----------

        nde :
            the concentration of electron in the defect
        ni :
            the intrinsic carrier density of the semiconductor
        temp:
            the temperature in Kelvin
        '''
        return self._e_emission_coef(ni, temp) * nde

    def emitted_h(self, nde, ni, temp):
        '''
        The emission rate of holes

        Parameters
        ----------

        nde :
            the concentration of electron in the defect
        ni :
            the intrinsic carrier density of the semiconductor
        temp:
            the temperature in Kelvin
        '''
        ndh = self.Nd - nde
        return self._h_emission_coef(ni, temp) * ndh

    def capture_e(self, ne, nde):
        '''
        The capture rate of electrons

        Parameters
        ----------
        ne :
            the concentration of free electrons
        nde :
            the concentration of electron in the defect
        '''
        ndh = self.Nd - nde
        # note that the division by Nd removes its impact
        return ne * ndh / self.tau_emin / self.Nd

    def capture_h(self, nh, nde):
        '''
        The capture rate of holes

        Parameters
        ----------
        nh :
            the concentration of free hole
        nde :
            the concentration of electron in the defect
        '''
        # note that the division by Nd removes its impact
        return nh * nde / self.tau_hmin / self.Nd

    @property
    def tau_emin(self):
        '''
        The minimum lifetime of electrons

        >>> s = SingleLevel(Ed=0, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print('{0:.2e} us'.format(s.tau_emin))
        4.88e-06 us
        '''
        if self.__tau_emin is None:
            val = 1. / self.__sigma_e / self.vth_e / self.Nd
        else:
            val = self.__tau_emin

        return val

    @tau_emin.setter
    def tau_emin(self, value):

        if value is not None:
            self.__sigma_e = None
            self.__tau_emin = value

    @property
    def tau_hmin(self):
        '''
        The minimum lifetime of holes

        >>> s = SingleLevel(Ed=0, sigma_e=1e-14, sigma_h=1e-14, Nd = 1e12)
        >>> print('{0:.2e} us'.format(s.tau_hmin))
        5.92e-06 us
        '''
        if self.__tau_hmin is None:
            val = 1. / self.__sigma_h / self.vth_h / self.Nd
        else:
            val = self.__tau_hmin

        return val

    @tau_hmin.setter
    def tau_hmin(self, value):
        if value is not None:
            self.__tau_hmin = value
            self.__sigma_h = None

    @property
    def sigma_h(self):
        '''
        The capture cross section of holes
        '''
        if self.__sigma_h is None:
            val = 1. / self.__tau_hmin / self.vth_h / self.Nd
        else:
            val = self.__sigma_h
        return val

    @sigma_h.setter
    def sigma_h(self, value):
        if value is not None:
            self.__tau_hmin = None
            if type(value) == list:
                self.__sigma_h = np.array(value)
            else:
                self.__sigma_h = value

    @property
    def sigma_e(self):
        '''
        The capture cross section of electrons
        '''
        if self.__sigma_h is None:
            val = 1. / self.__tau_emin / self.vth_e / self.Nd
        else:
            val = self.__sigma_e
        return val

    @sigma_e.setter
    def sigma_e(self, value):
        if value is not None:
            self.__tau_emin = None
            if type(value) == list:
                self.__sigma_e = np.array(value)
            else:
                self.__sigma_e = value

    def params(self):
        return {'tau_hmin': self.tau_hmin,
                'tau_emin': self.tau_emin,
                'Ed': self.Ed,
                'Nd': self.Nd,
                }


class MultiLevel(SingleLevel):
    '''
    This class represents a Sah and Shockey defect defect with tow or more states.

    Each state must be defined with its own parameters, being the same for a single level defect.

    A single defect has:
        1. Ed, the energy level of the defect from midgap.
        2. sigma_h or tau_h. The capture cross section for holes, or the minimum hole lifetime, respecitvly.
        3. sigma_n or tau_e  The capture cross section for electrons, or the minimum electron lifetime, respecitvly.
        4. The charge on the defect, now under the vairable charge (occupied_charge, unoccupied_charge).

    so here eveything is passed, instead of as a float as a list.

    Other things that are important are:
        1. the concentration of the defect (Nd)
        2. the thermal velocities (vth_h, vth_e)


    All these are able to be set at the initialsation of the defect or later.
    '''

    def __init__(self, sigma_e, sigma_h, Ed, Nd, charge):

        if Ed is not None:
            Ed = np.array(Ed)

        vals = list(np.average(charge, axis=1))

        if len(charge) > 1:
            index = sorted(range(len(vals)), key=vals.__getitem__)[::-1]
            # remove sorted from line below
            self.charge = [sorted(charge[i]) for i in index]
            # self.charge = [(charge[i]) for i in index]
            # print(self.charge)

        else:
            self.charge = charge
            index = [0]

        print(index)
        # initalise the class using the index from the charge of the defect.
        super().__init__(sigma_e=np.array(sigma_e)[index], sigma_h=np.array(
            sigma_h)[index], Ed=np.array(Ed)[index], Nd=Nd)

    def __attrs(self, dic):
        '''
        sets the values in provided dictionary
        '''
        assert type(dic) == dict
        for key, val in dic.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def current_net_charge(self, ncs):
        '''
        calculates the net charge of the defect given a charge distribution.
        '''
        c = [max(item) for item in self.charge]
        c.append(min(c) - 1)
        return np.sum(c * ncs)

    def net_charge(self, Ef, temp):
        '''
        calculates the net charge of the defect given fermi energy level and temperature.

        Parameters
        ----------

        Ef : (float or array, unit eV)
            the fermi energy level or quasi fermi energy level. If both electron and hole are provided the electron is to be provided first.
        temp : (float in K)
            The temperature of the sample.
        '''
        ncs = self.charge_state_concentration(Ef, temp)

        # get the charge of each level
        c = [max(item) for item in self.charge]
        c.append(min(c) - 1)

        return np.sum(c * ncs)

    def recombination_SS_ateachlevel(self, qEfe, qEfh, temp, ni):
        '''
        This calculates the recombiation occuring through a defect in steady state. This uses the Sah and Shockley formalisation, eqn 3.19 in 10.1103/PhysRev.109.1103.

        Parameters
        ----------
        qEfe : (float eV)
            The quasi Fermi energy level of electons
        qEfh : (float eV)
            The quasi Fermi energy level of holes

        temp : (float, kelvin)
            temperature of the sample
        ni : (float, cm^-3)
            the intrinsic carrier density

        Returns
        -------
        recombiation : (array (nxm), cm^-3)
            where n is the number of quasi fermi energy level set, and m is the number of defect levels.
        '''

        if not isinstance(qEfe, np.ndarray):
            qEfe = np.array([qEfe])
        if not isinstance(qEfh, np.ndarray):
            qEfh = np.array([qEfh])

        # getting the carrier from the fermi energy levels
        ne = ni * np.exp(qEfe * C.e / C.k / temp)
        nh = ni * np.exp(-qEfh * C.e / C.k / temp)

        # getting the other things
        ne1 = ni * np.exp(self.Ed * C.e / C.k / temp)
        nh1 = ni * np.exp(-self.Ed * C.e / C.k / temp)

        tau_e = 1 / self.vth_e / self.sigma_e
        tau_h = 1 / self.vth_h / self.sigma_h

        Ns = np.array([])
        Ns1 = np.array([])

        # getting the number of charge state concentration
        for qefe, qefh in zip(qEfe, qEfh):
            N = self.charge_state_concentration([qefe, qefh], temp)

            if Ns.shape[0] == 0:
                Ns = np.array([N[1:]])
                Ns1 = np.array([N[:-1]])
            else:
                Ns = np.vstack((Ns,  N[1:]))
                Ns1 = np.vstack((Ns1,  N[:-1]))

        # print(Ns.shape)
        if Ns.shape[1] == 1:
            Ns = Ns
            Ns1 = Ns1

        # ensureing the vairables have the right shape
        _ne = (np.zeros((ne1.shape[0], ne.shape[0])) + ne).T
        _nh = (np.zeros((ne1.shape[0], nh.shape[0])) + nh).T
        _ni = (np.zeros((ne1.shape[0], nh.shape[0])) + ni).T

        # SRH recombiation
        U = (_ne * _nh - ni**2) / (
            (_nh + nh1) * tau_e + (_ne + ne1) * tau_h)

        # print(Ns + Ns1, 'the number of defects
        # calculating the recombiation in each state
        U = (Ns + Ns1) * U
        return U

    def recombination_SS(self, qEfe, qEfh, temp, ni):
        '''
        This calculates the total recombiation occuring through all defect levels in steady state.

        Parameters
        ----------
        qEfe : (float eV)
            The quasi Fermi energy level of electons
        qEfh : (float eV)
            The quasi Fermi energy level of holes

        temp : (float, kelvin)
            temperature of the sample
        ni : (float, cm^-3)
            the intrinsic carrier density


        Notes
        -----
        The emission and capture rates are the same as a shockley read hall defect.
        '''

        U = self.recombination_SS_ateachlevel(qEfe, qEfh, temp, ni)

        # suming over all the defect state recombiation
        if len(U.shape) > 1:
            U = np.sum(U, axis=1)

        return U

    def capture_e(self, ne, ncs):
        '''
        The net transition of electonrs into all levels of a multi level defect

        Note here ncs represents the number of defects in each charge stage. It should be listed from the lowest level upwards. It should be one larger than the numer of capture cross sections/energy levels of the defect.


        Parameters
        ----------
        ne : (float)
            The number of electrons in the conduction band

        ncs : (array)
            the number of defects in each charge stage. It should be listed from the lowest level upwards.
        '''
        return ne * ncs[:-1] * self.vth_e * self.sigma_e

    def capture_h(self, nh, ncs):
        '''
        The net transition of holes into all levels of a multi level defect


        Parameters
        ----------
        nh : (float)
            The number of holes in the valance band

        ncs : (array)
            the number of defects in each charge stage. It should be listed from the lowest level upwards.
        '''

        return nh * ncs[1:] * self.vth_h * self.sigma_h

    def emitted_e(self, ncs, ni, temp):
        '''
        The net transition of electonrs out all levels of a multi level defect

        Parameters
        ----------
        ncs : (array)
            the number of defects in each charge stage. It should be listed from the lowest level upwards.

        ni : (float)
            The intrinsic carrier density
        temp : (float)
            The temeprature

        '''
        # print('too many prints', self._e_emission_coef(ni, temp), ncs[1:])
        return self._e_emission_coef(ni, temp) * ncs[1:]

    def emitted_h(self, ncs, ni, temp):
        '''
        The net transition of holes out all levels of a multi level defect

        Parameters
        ----------
        nh : (array)
            The number of holes in the valance band

        ncs : (array)
            the number of defects in each charge stage. It should be listed from the lowest level upwards.
        '''
        return self._h_emission_coef(ni, temp) * ncs[:-1]
