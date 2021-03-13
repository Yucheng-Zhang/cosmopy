'''
Cosmology models.
Some notebooks for reference are included in this repo:
https://github.com/Yucheng-Zhang/cosmo-notes
'''
import numpy as np
from scipy import interpolate, integrate

import camb
from colossus.cosmology import cosmology


class flatLCDM:
    '''FlatLambdaCDM cosmology.'''

    def __init__(self, cosmoc=None, pars='planck18', Z_CMB=1090):

        # cosmoc is a colossus cosmology instance
        if cosmoc is None:
            self.cosmoc = cosmology.setCosmology(pars)
        else:
            self.cosmoc = cosmoc

        # ------ constants ------
        self.H0 = self.cosmoc.H0
        self.h = self.cosmoc.H0 / 100.
        self.Om0 = self.cosmoc.Om0
        self.Ob0 = self.cosmoc.Ob0
        self.Tcmb0 = self.cosmoc.Tcmb0
        self.Neff = self.cosmoc.Neff
        self.ns = self.cosmoc.ns
        self.sigma8 = self.cosmoc.sigma8
        self.As = None
        self.Dz0 = self.D_unnorm(0.)  # used to normalize D(z=0) to 1

        self._init_camb()
        self._init_interpolation()

        self.chi_CMB = self.chi(Z_CMB)

    def _init_camb(self):
        '''setup CAMB parameters and compute results'''
        camb_pars = camb.CAMBparams()
        camb_pars.set_cosmology(H0=self.H0, ombh2=self.Ob0*self.h**2,
                                omch2=(self.Om0 - self.Ob0)*self.h**2, omk=0.0,
                                TCMB=self.Tcmb0, nnu=self.Neff)
        if self.As is None:
            As_ = 2.1e-9
            camb_pars.InitPower.set_params(As=As_, ns=self.ns)
        else:
            camb_pars.InitPower.set_params(As=self.As, ns=self.ns)

        camb_pars.set_matter_power(redshifts=(0.0,), kmax=10.0, nonlinear=True,
                                   accurate_massive_neutrino_transfers=False)
        camb_pars.set_for_lmax(2500, lens_potential_accuracy=2)
        
        camb_pars.set_accuracy(AccuracyBoost=3.0)

        # calculate all results
        self.camb_results = camb.get_results(camb_pars)
        self.camb_pars = camb_pars
        
        if self.As is None:
            sigma8_ = self.camb_results.get_sigma8()[0]
            self.As = (self.sigma8 / sigma8_)**2 * As_
            self._init_camb() # rerun with true As

    def _init_interpolation(self):
        '''setup interpolation'''
        # log-spaced z samples for interpolation
        zs = np.concatenate([np.array([0.]), np.logspace(np.log10(0.001),
                                                         np.log10(1200.), 699)])
        # chi(z) & z(chi)
        chis = self.chi(zs, interp=False)
        self.z2chi_interp = interpolate.interp1d(zs, chis, kind='cubic',
                                                 bounds_error=False, fill_value='extrapolate')
        self.chi2z_interp = interpolate.interp1d(chis, zs, kind='cubic',
                                                 bounds_error=False, fill_value='extrapolate')
        # D_unnorm(z)
        D_unnorms = self.D_unnorm(zs, interp=False)
        self.z2D_unnorm_interp = interpolate.interp1d(zs, D_unnorms, kind='cubic',
                                                      bounds_error=False, fill_value='extrapolate')
        # T(k, z=0)
        matter_trans = self.camb_results.get_matter_transfer_data()
        ks = matter_trans.q  # in Mpc^{-1}
        Tks = matter_trans.transfer_data[camb.model.Transfer_tot-1, :, -1]
        Tks_norm = Tks / Tks[0]  # normalize to one at low k
        self.Tk0 = Tks[0]
        self.Tk_interp = interpolate.interp1d(ks, Tks_norm, kind='cubic',
                                              bounds_error=False, fill_value='extrapolate')
        # P(k, z=0)
        Pks_linear = self.camb_results.get_linear_matter_power_spectrum(
            hubble_units=False, k_hunit=False)[2][0]
        Pks_nonlin = self.camb_results.get_nonlinear_matter_power_spectrum(
            hubble_units=False, k_hunit=False)[2][0]
        self.Pk_linear_interp = interpolate.interp1d(ks, Pks_linear, kind='cubic',
                                                     bounds_error=False, fill_value='extrapolate')
        self.Pk_nonlin_interp = interpolate.interp1d(ks, Pks_nonlin, kind='cubic',
                                                     bounds_error=False, fill_value='extrapolate')

    # ------ background functions ------

    def E(self, z):
        '''E(z) = H(z)/H(0)'''
        return self.cosmoc.Ez(z)

    def H(self, z):
        '''H(z)'''
        return self.H0 * self.cosmoc.Ez(z)

    def chi(self, z, interp=False):
        '''chi(z) in Mpc'''
        if interp:
            res = self.z2chi_interp(z)
        else:
            res = self.cosmoc.comovingDistance(z_max=z
                                               ) / self.h  # Mpc/h -> Mpc
        return res

    def z_at_chi(self, chi):
        '''z(chi), chi should be in Mpc'''
        return self.chi2z_interp(chi)

    def Om(self, z):
        '''Omega_m(z).'''
        return self.cosmoc.Om(z)

    # ------ linear growth factor ------

    def D_lowz_int(self, z):
        '''D(z) by evaluating the integral at late time (z < 10),
           equivelant to D_unnorm(z) at low z.'''
        def integrand(zp):
            return (1 + zp) / self.E(zp)**3

        return 5./2.*self.Om0 * self.E(z) * integrate.quad(integrand, z, np.inf)[0]

    def D_unnorm(self, z, interp=False):
        '''D(z), normalized to (1+z)^{-1} in matter domination'''
        if interp:
            res = self.z2D_unnorm_interp(z)
        else:
            res = self.cosmoc.growthFactorUnnormalized(z)

        return res

    def D(self, z, interp=False):
        '''D(z), normalized to D(z=0)=1'''
        return self.D_unnorm(z, interp=interp) / self.Dz0

    def f(self, z):
        '''f(z), linear growth rate w/ GR'''
        return np.power(self.Om(z), 0.55)

    # ------ matter power ------

    def Tk(self, k):
        '''matter transfer function, normalized to 1 at low k'''
        return self.Tk_interp(k)

    def Tk_unnorm(self, k):
        '''matter transfer function, unnormalized'''
        return self.Tk_interp(k) * self.Tk0

    def Pk(self, k, nl=False):
        if nl:
            return self.Pk_nonlin_interp(k)
        else:
            return self.Pk_linear_interp(k)
