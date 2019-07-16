'''
LambdaCDM class.
'''
from astropy.cosmology import FlatLambdaCDM
import camb
from scipy import interpolate, integrate
import numpy as np

Z_CMB = 1100


class cosmoLCDM:
    '''FlatLambdaCDM cosmology.'''

    # ------ initialize class ------

    def __init__(self, H0, Om0, Ob0, Tcmb0, As, ns):
        self.H0 = H0
        self.h = self.H0 / 100.
        self.Om0 = Om0
        self.Ob0 = Ob0
        self.Tcmb0 = Tcmb0
        self.As = As
        self.ns = ns

        self.cosmo = FlatLambdaCDM(H0=self.H0, Tcmb0=self.Tcmb0, Om0=self.Om0,
                                   Ob0=self.Ob0)

        self.D_norm = None  # used to normalize D(z=0) to 1

        # for CAMB
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.H0, ombh2=self.Ob0*self.h**2,
                                omch2=(self.Om0-self.Ob0)*self.h**2, TCMB=self.Tcmb0)
        self.pars.InitPower.set_params(As=self.As, ns=self.ns)

        self.pars.WantTransfer = True
        self.pars.Transfer.high_precision = True

        self.results = camb.CAMBdata()

        self.CHI_CMB = self.z2chi(Z_CMB)

    # ------ basic functions ------

    def H_z(self, z):
        '''Hubble parameter at redshift z, i.e. H(z)'''
        return self.H0 * self.cosmo.efunc(z)

    def D_z(self, z):
        '''Normalized (D(0)=1) growth factor at redshift z, i.e. D(z).'''
        def kernel(zp):
            hp = np.power(self.cosmo.efunc(zp), 3.)
            return (1 + zp) / hp

        if self.D_norm is None:
            self.D_norm = integrate.quad(kernel, 0, np.Infinity)[0]

        return self.cosmo.efunc(z) * integrate.quad(kernel, z, np.Infinity)[0] / self.D_norm

    def z2chi(self, z):
        '''Get comoving distance in [Mpc] from redshift'''
        return self.cosmo.comoving_distance(z).value

    def w_z(self, z, zs=Z_CMB):
        '''Lensing kernel.'''
        chi_z = self.z2chi(z)
        if zs is Z_CMB:
            chi_zs = self.CHI_CMB
        else:
            chi_zs = self.z2chi(zs)

        return chi_z * (1. - chi_z / chi_zs)

    def Om_z(self, z):
        '''Omega_m(z).'''
        tmp = self.Om0 * np.power(z, 3)
        return tmp / (tmp + 1 - self.Om0)

    def f_growth_z(self, z):
        '''Linear growth rate w/ GR.'''
        return np.power(self.Om_z(z), 0.55)

    # ------ interpolated functions ------
    # ------ either impossible to calculate directly ------
    # ------ or for fast calculation ------

    def gen_interp_chiz(self, zmin=0, zmax=10, dz=0.001, kind='cubic'):
        '''Generate comoving distance [Mpc] to redshift function with interpolation.'''
        zs = np.arange(zmin, zmax+dz, dz)
        chis = self.z2chi(zs)
        self.interp_chi2z = interpolate.interp1d(chis, zs, kind=kind,
                                                 bounds_error=True)
        self.interp_z2chi = interpolate.interp1d(zs, chis, kind=kind,
                                                 bounds_error=True)
        print('>> self.interp_chi2z(chi) and self.interp_z2chi(z) generated for z in [{0:g}, {1:g}] \
               with dz={2:g} interpolated with {3:s}'.format(zmin, zmax, dz, kind))

    def gen_interp_D_z(self, zmin=0, zmax=10, dz=0.001, kind='cubic'):
        '''Generate interpolated D(z).'''
        zs = np.arange(zmin, zmax+dz, dz)
        Ds = np.array([self.D_z(z_) for z_ in zs])
        self.interp_D_z = interpolate.interp1d(zs, Ds, kind=kind,
                                               bounds_error=True)
        print('>> self.interp_D_z(z) generated for z in [{0:g}, {1:g}] with dz={2:g} interpolated with {3:s}'
              .format(zmin, zmax, dz, kind))

    def gen_interp_pk(self, zmin, zmax, kmax, extrap_kmax=None):
        '''Generate matter power spectrum Pmm(z, k).'''
        self.interp_pk = camb.get_matter_power_interpolator(self.pars, zmin=zmin, zmax=zmax,
                                                            kmax=kmax, nonlinear=True,
                                                            hubble_units=False, k_hunit=False,
                                                            extrap_kmax=extrap_kmax)

        print('>> Matter power spectrum self.interp_pk.P(z,k) generated with CAMB.')

    def gen_interp_Tk(self, kmax, kind='cubic'):
        '''Generate Transfer function T(k).'''
        self.pars.Transfer.kmax = kmax
        self.results.calc_transfers(self.pars)

        trans_data = self.results.get_matter_transfer_data()
        k = trans_data.q
        Tk = trans_data.transfer_data[6, :, 0]
        Tk = Tk / np.amax(Tk)  # normalize

        self.interp_Tk = interpolate.interp1d(k, Tk, kind=kind,
                                              bounds_error=True)

        print('>> Matter transfer function self.interp_Tk(k) generated with CAMB.')

    def interp_w_z(self, z, zs=Z_CMB):
        '''Lensing kernel w/ interpolated z2chi.'''
        chi_z = self.interp_z2chi(z)
        if zs is Z_CMB:
            chi_zs = self.CHI_CMB
        else:
            chi_zs = self.interp_z2chi(zs)

        return chi_z * (1. - chi_z / chi_zs)
