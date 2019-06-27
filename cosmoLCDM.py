'''
LambdaCDM class.
'''
from astropy.cosmology import FlatLambdaCDM
import camb
from scipy import interpolate
import numpy as np

Z_CMB = 1100


class cosmoLCDM:
    '''FlatLambdaCDM cosmology.'''

    def __init__(self, H0, Om0, Ob0, Tcmb0, As, ns):
        self.H0 = H0
        self.h = self.H0 / 100.
        self.Om0 = Om0
        self.Ob0 = Ob0
        self.Tcmb0 = Tcmb0
        self.As = As
        self.ns = ns

        self.cosmo = FlatLambdaCDM(
            H0=self.H0, Tcmb0=self.Tcmb0, Om0=self.Om0, Ob0=self.Ob0)

    def H_z(self, z):
        '''Hubble parameter at redshift z, i.e. H(z)'''
        return self.H0 * self.cosmo.efunc(z)

    def z2chi(self, z):
        '''Get comoving distance in [Mpc] from redshift'''
        return self.cosmo.comoving_distance(z).value

    def gen_chi2z(self, zmin=0, zmax=10, dz=0.001, kind='cubic'):
        '''Generate comoving distance [Mpc] to redshift function with interpolation.'''
        zs = np.arange(zmin, zmax+dz, dz)
        chis = self.z2chi(zs)
        self.chi2z = interpolate.interp1d(chis, zs, kind=kind,
                                          bounds_error=True)
        print(' >> self.chi2z(chi) generated for z in [{0:g}, {1:g}] with dz={2:g} interpolated with {3:s}'
              .format(zmin, zmax, dz, kind))

    def w_z(self, z, zs=Z_CMB):
        '''Lensing kernel.'''
        chi_z = self.z2chi(z)
        chi_zs = self.z2chi(zs)
        return chi_z * (1. - chi_z / chi_zs)

    def gen_pk(self, z1, z2, kmax, extrap_kmax=None):
        '''Generate matter power spectrum Pmm(z, k).'''
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.Ob0*self.h**2,
                           omch2=(self.Om0-self.Ob0)*self.h**2, TCMB=self.Tcmb0)
        pars.InitPower.set_params(As=self.As, ns=self.ns)

        self.pk = camb.get_matter_power_interpolator(pars, zmin=z1, zmax=z2,
                                                     kmax=kmax, nonlinear=True,
                                                     hubble_units=False, k_hunit=False,
                                                     extrap_kmax=extrap_kmax)

        print('>> Matter power spectrum function self.pk.P(z,k) generated with CAMB.')
