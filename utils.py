'''
Some useful functions.
'''
import numpy as np
from scipy import interpolate


def gen_fg_z(dataz, z1, z2, bins=200, kind='linear'):
    '''Generate the redshift distribution function fg(z).'''
    hist, bin_edges = np.histogram(
        dataz, bins=bins, range=(z1, z2), density=True)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    xs = np.concatenate(([z1], bin_mids, [z2]))
    ys = np.concatenate(([hist[0]], hist, [hist[-1]]))

    fg_z = interpolate.interp1d(xs, ys, kind=kind,
                                bounds_error=False, fill_value=(0., 0.))

    print('>> Redshift distribution fg genertaed from data.')
    return fg_z
