
import numpy as np
import scipy.constants as C
from . import defects
from semiconductor.material import IntrinsicCarrierDensity
from semiconductor.material import ThermalVelocity
import numbers
from scipy.optimize import newton


def getvalue_modelornumber(value, model, extension, **kwargs):
    '''
    Tests if the provded value is a float or numpy array. If it is not,
    it assumes it is a points to a value and attempts to retrieve that values by model.extension()

    Parameters
    ----------
    value :
        the value to test
    model :
        the model to use if the value is a string
    extension :
        the function of the model to be called to find the desribed value
    kwargs : (optional)
        Values to be passed to the model's extension

    Returns
    -------
    value :
        some value
    '''
    if isinstance(value, numbers.Number):
        value = value
    elif isinstance(value, np.ndarray):
        value = value
    elif hasattr(model, extension):
        if isinstance(value, str) or value == None:
            value = getattr(model, extension)(**kwargs)
            if isinstance(value, np.ndarray):
                if value.shape[0] == 1:
                    value = value[0]

    else:
        print('Incorrect type, or not a function')

    return value


class Sample():
    '''
    This is a class that attempts to encapulate all the samples properties.
    One of the problems is that there is a lot of properties so this is a very
    large class

        ni, vth_e, vth_h are all calculated values from theoretical models unless a value is provided. In that case the value is used. This is also a method to speed up calculations. Passing a value of None can then be used to reset these calculations.
    '''

    def __init__(self, **kwargs):
        '''
        initalisates the class
        '''

        self._vth_h = None
        self._vth_e = None

        self.name = None
        self.sample_id = None
        self._dopant_type = None  # takes either 'n-type' or b'p-type'
        self.thickness = None
        self.absorptance = 1
        self._Nacc = 1e16
        self._Ndon = 0
        self._doping = None
        self._ni = None
        self._nieff = 1e10
        self._temp = 300  # as most measurements are done at room temperature
        self.nxc = None
        self._defectlist = []
        self.tau_rad = np.inf

        self.__attrs(kwargs)

        self.equlibrium_concentrations()

    def __attrs(self, dic):
        '''
        sets the values in provided dictionary
        '''
        assert type(dic) == dict
        for key, val in dic.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def defectlist(self):
        return self._defectlist

    @defectlist.setter
    def defectlist(self, value):
        '''
        Adds a defect to the list
        '''
        assert isinstance(value, defects.SingleLevel)
        value.vth_e = self.vth_e
        value.vth_h = self.vth_h
        self._defectlist.append(value)

    @defectlist.deleter
    def defectlist(self, value):
        '''
        Clears all defects
        '''

        self._defectlist = []

    def defectlist_removeDefect(self, number):
        '''
        Removes a single defect from the defect list
        '''
        if number < len(self.defectlist):
            del self.defectlist[number]

    @property
    def temp(self):
        '''The sample temperature in kelvin'''
        return self._temp

    @temp.setter
    def temp(self, temp):
        self._temp = temp
        # update the things that dependent on temperature
        self.vth_e
        self.vth_h
        self.equlibrium_concentrations()

    @property
    def dopant_type(self):
        '''
        The sample's dopant type

        it can be:
            1. 'p' or 'p-type' for p-type
            2. 'n' or 'n-type' for n-type
        '''
        return self._dopant_type

    @dopant_type.setter
    def dopant_type(self, value):
        '''
        sets the dopant type:

        can provide
        'p' or 'p-type' for p-type
        'n' or 'n-type' for n-type
        '''
        if value == 'p' or value == 'p-type':
            self._dopant_type = 'p-type'
        elif value == 'n' or value == 'n-type':
            self._dopant_type = 'n-type'

    @property
    def doping(self):
        '''
        The number of dopants. This is not the ionised dopants
        '''
        if self._Nacc is None or self._Ndon is None:
            _doping = self._doping
        else:
            _doping = abs(self._Nacc - self._Ndon)
        return _doping

    @doping.setter
    def doping(self, value):
        '''
        Sets the number of dopant atoms. It assumes there is only one dopant type
        i.e if it is a p-type material with 1e16 dopants, this function sets
        Na = 1e16 and Nd = 0.
        '''
        self._doping = value

        if self.dopant_type is not None:

            if self.dopant_type == 'p-type':
                self._Nacc = value
                self._Ndon = 0
            elif self.dopant_type == 'n-type':
                self._Ndon = value
                self._Nacc = 0
            else:
                print('\n\n', self.dopant_type, '\n\n')

    @property
    def Nacc(self):
        '''
        The concentration of acceptor dopant atoms
        '''
        return self._Nacc

    def _check_dopant_type(self):
        if self._Nacc is None:
            self._Nacc = 0
        if self._Ndon is None:
            self._Ndon = 0

        if self._Nacc > self._Ndon:
            self.dopant_type = 'p-type'
        else:
            self.dopant_type = 'n-type'

    @Nacc.setter
    def Nacc(self, value):
        self._Nacc = value
        self._check_dopant_type()

    @property
    def Ef(self):
        return self.Vt * np.log(self.ne0 / self.ni)

    @property
    def Ndon(self):
        '''
        The concentration of donor dopant atoms
        '''
        return self._Ndon

    @Ndon.setter
    def Ndon(self, value):
        self._Ndon = value
        self._check_dopant_type()

    @property
    def vth_h(self):
        '''
        The thermal velocity of an electron
        '''
        model = ThermalVelocity(
            material='Si', temp=self.temp,
        )

        val = getvalue_modelornumber(self._vth_h, model, 'update',
                                     author=self._vth_h)[1]

        for i in self.defectlist:
            i.vth_h = val
        return val

    @vth_h.setter
    def vth_h(self, val):
        '''
        The thermal velocity of an electron
        '''
        self._vth_h = val

    @property
    def vth_e(self):
        '''
        The thermal velocity of an electron
        '''
        model = ThermalVelocity(
            material='Si', temp=self.temp,
        )

        val = getvalue_modelornumber(self._vth_e, model, 'update',
                                     author=self._vth_e)[0]
        if isinstance(val, np.ndarray):
            if val.shape[0] == 1:
                val = val[0]

        for i in self.defectlist:
            i.vth_e = val
        return val

    @vth_e.setter
    def vth_e(self, val):
        '''
        The thermal velocity of an electron
        '''
        self._vth_e = val

    @property
    def ni(self):
        '''
        The sample's intrinsic carrier density. If this is not provided it will be calculated.
        '''
        model = IntrinsicCarrierDensity(
            material='Si', temp=self.temp,
        )

        val = getvalue_modelornumber(self._ni, model, 'update',
                                     author=self._ni)

        return val

    @ni.setter
    def ni(self, val):
        self._ni = val

    @property
    def ni_eff(self):
        '''
        Currently not impimented and just returns the intrinsic carrier density.
        '''
        # model = BandGapNarrowing(
        #     material='Si',
        #     temp=self.temp,
        #     nxc=self.nxc,
        #     Na=self.Na,
        #     Nd=self.Nd,
        # )
        # return getvalue_modelornumber(self._nieff, model, 'ni_eff', ni=self.ni,
        #   author=self._nieff)
        return self.ni

    @ni_eff.setter
    def ni_eff(self, val):
        self._nieff = val

    @property
    def Vt(self):
        'The thermal voltage'
        return self.temp * C.k / C.e

    def equlibrium_concentrations(self):
        '''
        Calculate the equlibrium carrier concentration

        Returns
        -------
        ne0 : float
            the free electron concentration in the dark
        nh0 : float
            the free hole concentration in the dark
        '''
        def charges(majoirty_carrier):
            if self.Nacc > self.Ndon:
                nh0 = np.array(majoirty_carrier, dtype=np.float64)
                ne0 = np.array(ni**2 / majoirty_carrier,
                               dtype=np.float64)
            elif self.Ndon > self.Nacc:
                ne0 = np.array(majoirty_carrier, dtype=np.float64)
                nh0 = np.array(ni**2 / majoirty_carrier,
                               dtype=np.float64)
            else:
                ne0 = ni
                nh0 = ni
            return ne0, nh0

        def possion(majoirty_carrier):
            ne0, nh0 = charges(majoirty_carrier)

            Ef = self.Vt * np.log(ne0 / ni)

            # this is just a wrapper to set nte, which the following
            # possion function uses
            defect_charge = 0
            for _dft in self.defectlist:
                defect_charge += _dft.net_charge(Ef, self.temp)

            return self.Ndon - ne0 - self.Nacc + nh0 + defect_charge
        # This line is to speed up calculations
        # recalculation of ni each time is quite slow.
        ni = self.ni
        # initial guess of majority carrier
        mjc = newton(possion, abs(self.Nacc - self.Ndon), maxiter=100)

        # new lets work out the details
        self.ne0, self.nh0 = charges(mjc)

        return self.ne0, self.nh0

    def steady_state_concentraions(self, nxc):
        '''
        Calculats the steady state concentration of carriers,
        provided the number of excess minoirty carriers by peturbing the number of excess majority carriers.

        This assumes a constant intrinsic carrier density.

        Parameters
        ----------
        nxc : float
            the number of excess carrier densities

        Returns
        -------
        ne: float
            the free electron concentration
        nh: float
            the free hole concentration
        nd: array like
            the number of defects in each state

        '''
        def charges(mj):
            '''
            calculations the change in majority carrier density

            this function doesn't really need to be here
            '''

            if self.Nacc > self.Ndon:
                nh = np.array(mj, dtype=np.float64)
                ne = np.array(self.ne0 + nxc[i], dtype=np.float64)

            elif self.Ndon > self.Nacc:
                ne = np.array(mj, dtype=np.float64)
                nh = np.array(self.nh0 + nxc[i], dtype=np.float64)

            return ne, nh

        def possion(mj):
            '''
            determine the charge neutrality

            '''
            # get the number of carriers
            ne, nh = charges(mj)

            # turns them into an quasi fermi energy level
            qEfe = self.Vt * np.log(ne / ni)
            qEfh = -1 * self.Vt * np.log(nh / ni)
            # this is just a wrapper to set nte, which the following
            # possion function uses
            defect_charge = np.array(0, dtype=np.float64)

            # find the charge of the defects
            for _dft in self.defectlist:
                defect_charge += _dft.net_charge([qEfe, qEfh], temp)

            # add up all the charges
            return self.Ndon - ne - self.Nacc + nh + defect_charge

        self.equlibrium_concentrations()

        if not isinstance(nxc, np.ndarray):
            nxc = np.array([nxc])

        ne = np.zeros(nxc.shape[0])
        nh = np.zeros(nxc.shape[0])
        nd = []
        nd = np.array([])
        ni = self.ni
        Vt = self.Vt
        temp = self.temp

        mj = max(self.ne0, self.nh0)
        # need to do each carrier density individually
        for i, nxc_i in enumerate(nxc):
            # initial guess of the number of excess majority carrier
            # then solve for charge conservation
            mj = newton(possion, np.array(
                mj + nxc[i], dtype=np.float64), maxiter=100, tol=1.48e-38)
            # get the values
            ne_i, nh_i = charges(mj)

            # save the values into the array
            ne[i] = ne_i
            nh[i] = nh_i

            # save the defect concentrations
            for _dft in self.defectlist:
                # nd.append(_dft.charge_state_concentration(
                    # [Vt * np.log(ne_i / ni), -Vt * np.log(nh_i / ni)], temp))
                nd = np.concatenate((nd, _dft.charge_state_concentration(
                    [Vt * np.log(ne_i / ni), -Vt * np.log(nh_i / ni)], temp)))

        return ne, nh, nd


if __name__ == "__main__":
    import doctest
    doctest.testmod()
