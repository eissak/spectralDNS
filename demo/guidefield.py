from time import time
stime = time()

from h5py import File
from numpy import pi, sin, cos, sqrt, float64, \
        sum, conj, arange, digitize, zeros, where, \
        abs, isnan, max, min
from numpy.random import default_rng
from spectralDNS import config, get_solver, solve

def initialize(UB_hat, UB, U, B, X, **context):
    """ Initialization with a uniform guide field
    in z-direction and random fluctuations in xy-
    plane according to Nättilä & Beloborodov,
    ApJ 921 87 (2021)
    """

    # Modes in xy and z
    N_perp = 3
    N_par = 2

    # Compute random phases
    rng = default_rng(294671057703484)
    phi = 2*pi*rng.random((3, N_perp, N_perp, N_par))

    # Velocity field initialization
    U[:] = 0

    # Magnetic field initialization
    B0 = 1
    b = lambda l, m: 2*sqrt(2)*B0 / (N_perp*sqrt(N_par*(l**2 + m**2)))

    B[:2] = 0
    B[2] = B0
    for l in range(1, N_perp+1):
        for m in range(1, N_perp+1):
            for n in range(1, N_par+1):
                B[0] += b(l, m)*m \
                        *sin(l*X[0] + phi[0, l-1, m-1, n-1]) \
                        *cos(m*X[1] + phi[1, l-1, m-1, n-1]) \
                        *sin(n*X[2] + phi[2, l-1, m-1, n-1])

                B[1] -= b(l, m)*l \
                        *cos(l*X[0] + phi[0, l-1, m-1, n-1]) \
                        *sin(m*X[1] + phi[1, l-1, m-1, n-1]) \
                        *sin(n*X[2] + phi[2, l-1, m-1, n-1])

    UB_hat = UB.forward(UB_hat)
    config.params.t = 0
    config.params.tstep = 0

def init_from_file(checkpointfile, solver, context):
    f = File(checkpointfile, driver="mpio", comm=solver.comm)
    assert "0" in f["UB/3D"]
    UB_hat = context.UB_hat
    s = context.T.local_slice(True)
    UB_hat[:] = f["UB/3D/0"][:, s[0], s[1], s[2]]

    config.params.t = f.attrs['t']
    config.params.tstep = f.attrs['tstep']

    f.close()

def isotropic_spectrum(solver, context):
    """ Compute the isotropic kinetic and magnetic energy spectra
    """
    c = context

    U_hat = c.UB_hat[:3]
    B_hat = c.UB_hat[3:]

    # Compute the sums of the squared norms of the Fourier components
    uiui = sum((U_hat*conj(U_hat)).real, axis=0)
    bibi = sum((B_hat*conj(B_hat)).real, axis=0)

    # The last axis is the real transform
    # All terms but the first and the last have to be multiplied by two
    uiui[..., 1:-1] *= 2
    bibi[..., 1:-1] *= 2

    # Create wavenumber bins (0...N/2 for homogeneous box)
    Nb = int(sqrt(sum((config.params.N/2)**2)/3))
    bins = arange(0, Nb) + 0.5
    z = digitize(sqrt(c.K[0]**2+c.K[1]**2+c.K[2]**2), bins, right=True)

    # Sample
    Ek = zeros(Nb)
    Eb = zeros(Nb)
    ll = zeros(Nb)
    for i, k in enumerate(bins[1:]):
        k0 = bins[i] # lower limit, k is upper
        ii = where((z > k0) & (z <= k)) # index array
        ll[i] = len(ii[0]) # number of indices

        # Multiply the sum by the volume of a shell
        Ek[i] = 4*pi/3*(k**3 - k0**3)*sum(uiui[ii])
        Eb[i] = 4*pi/3*(k**3 - k0**3)*sum(bibi[ii])

    Ek = solver.comm.allreduce(Ek)
    Eb = solver.comm.allreduce(Eb)
    ll = solver.comm.allreduce(ll)

    # Compute the average over the shell
    for i in range(Nb):
        if ll[i] != 0:
            Ek[i] /= ll[i]
            Eb[i] /= ll[i]

    return Ek, Eb, bins

def anisotropic_spectrum(solver, context):
    """ Compute the anisotropic kinetic and magnetic energy spectra
    where z is the mean field direction
    """
    c = context

    U_hat = c.UB_hat[:3]
    B_hat = c.UB_hat[3:]

    # Compute the sums of the squared norms of the Fourier components
    uiui_xy = sum((U_hat[:2]*conj(U_hat[:2])).real, axis=0)
    uzuz = (U_hat[2]*conj(U_hat[2])).real
    bibi_xy = sum((B_hat[:2]*conj(B_hat[:2])).real, axis=0)
    bzbz = (B_hat[2]*conj(B_hat[2])).real

    # The last axis is the real transform
    # All terms but the first and the last have to be multiplied by two
    uiui_xy[..., 1:-1] *= 2
    uzuz[..., 1:-1] *= 2
    bibi_xy[..., 1:-1] *= 2
    bzbz[..., 1:-1] *=2

    # Create parallel and perpendicular wavenumber bins
    Nb_xy = int(sqrt(sum((config.params.N[:2]/2)**2)/2))
    Nb_z = int(config.params.N[2]/2)
    bins_xy = arange(0, Nb_xy) + .5
    bins_z = arange(0, Nb_z) + .5
    z_xy = digitize(sqrt(c.K[0]**2+c.K[1]**2), bins_xy, right=True)
    z_z = digitize(abs(c.K[2]), bins_z, right=True)

    # Sample
    Ek_xy = zeros(Nb_xy)
    Ek_z = zeros(Nb_z)
    Eb_xy = zeros(Nb_xy)
    Eb_z = zeros(Nb_z)
    ll_xy = zeros(Nb_xy)
    #ll_z = zeros(Nb_z)

    for i, k in enumerate(bins_xy[1:]):
        k0 = bins_xy[i] # lower limit, k is upper
        ii = where((z_xy > k0) & (z_xy <= k)) # index array
        ll_xy[i] = len(ii[0]) # number of indices

        # Multiply the sum by the volume of a cylindrical shell
        Ek_xy[i] = 2*Nb_z*pi*(k**2 - k0**2)*sum(uiui_xy[ii])
        Eb_xy[i] = 2*Nb_z*pi*(k**2 - k0**2)*sum(bibi_xy[ii])

    for i, k in enumerate(bins_z[1:]):
        k0 = bins_z[i]
        ii = where((z_z > k0) & (z_z <= k))
        #ll_z[i] = len(ii[0])

        # No multiplication needed as it would be cancelled by averaging
        Ek_z[i] = sum(uzuz[ii])
        Eb_z[i] = sum(bzbz[ii])

    Ek_xy = solver.comm.allreduce(Ek_xy)
    Ek_z = solver.comm.allreduce(Ek_z)
    Eb_xy = solver.comm.allreduce(Eb_xy)
    Eb_z = solver.comm.allreduce(Eb_z)
    ll_xy = solver.comm.allreduce(ll_xy)

    # Compute the average over the shell
    for i in range(Nb_xy):
        if ll_xy[i] != 0:
            Ek_xy[i] /= ll_xy[i]
            Eb_xy[i] /= ll_xy[i]

    return Ek_xy, Ek_z, Eb_xy, Eb_z, bins_xy, bins_z

def update(context):
    solver = config.solver
    params = config.params
    if False: #params.tstep % params.write_result == 0:
        Ek_xy, Ek_z, Eb_xy, Eb_z, bins_xy, bins_z = anisotropic_spectrum(solver, context)
        f = File(context.hdf5file.filename+'_spectrum.h5', mode='a', driver='mpio', comm=solver.comm)

        f['Ek_xy/1D'].create_dataset(str(params.tstep), data=Ek_xy)
        f['Ek_z/1D'].create_dataset(str(params.tstep), data=Ek_z)
        f['Eb_xy/1D'].create_dataset(str(params.tstep), data=Eb_xy)
        f['Eb_z/1D'].create_dataset(str(params.tstep), data=Eb_z)

        f.close()

if __name__ == '__main__':
    config.update(
            {'nu': 0.01,
             'eta': 0.01,
             'dt': 0.01,
             'T': 0.1,
             'L': [2*pi, 2*pi, 2*pi],
             'M': [5, 5, 4],
             'convection': 'Divergence',
             'dealias': '3/2-rule',
             'write_result': 10,
             'checkpoint': 3000,
             #'planner_effort': {'fft': 'FFTW_ESTIMATE',
             #					'rfftn': 'FFTW_ESTIMATE',
             #					'irfftn': 'FFTW_ESTIMATE'},
             'optimization': 'cython',
             }, 'MHD')
    solver = get_solver(update=update)
    context = solver.get_context()

    context.hdf5file.results['data'].update({'W': [context.curl]})

    def update_components(**c):
        solver.get_ub(**c)
        solver.get_curl(**c)

    context.hdf5file.update_components = update_components

    # Switch this on if continuing from a checkpoint
    from_checkpoint = False

    comm = solver.comm

    if not from_checkpoint:
        initialize(**context)

        Ek_xy, Ek_z, Eb_xy, Eb_z, bins_xy, bins_z = anisotropic_spectrum(solver, context)
        spectrumname = context.hdf5file.filename+'_spectrum.h5'
        # Create a separate file for the spectra
        f = File(spectrumname, mode='w', driver='mpio', comm=comm)
        for grp in ['Ek_xy', 'Ek_z', 'Eb_xy', 'Eb_z']:
            f.create_group(f'{grp}/mesh')
            f.create_group(f'{grp}/1D')

            if grp.split('_')[1] == 'xy':
                bins = bins_xy
            else:
                bins = bins_z

            f[f'{grp}/mesh'].create_dataset('bins', data=bins)

        f.close()

    else:
        init_from_file(context.hdf5file.filename+'_c.h5', solver, context)
        # By default, append to the existing file
        solver.params.filemode = 'a'

    # Print initialization time
    comm.Barrier()
    rank = comm.Get_rank()
    if (rank == 0):
        dt = (time() - stime) / 60
        print(f'Init: {dt:.1f} min')

    solve(solver, context)
