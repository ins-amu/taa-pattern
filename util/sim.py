"""
Simulation of the seizure activity
"""

import os
from collections import deque
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy as sp

import gdist

from .geometry import create_part
from .surface import Surface, compute_triangle_areas
from .contacts import Contacts
from .post import AnalyzedSeeg, group_taas, add_elec_info, NCONF


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

DIST_LIM_MM = 15.0                             # Distance of the seizure patch center from the electrode contacts
DIST_TWOSTAT_MM = 10.0                         # Distance of the two patches in two-source model
NOISE_PATCH_AREA_MM = 100.                     # Size of the noise patches (outside of the seizure patch)
NOISE_CORR_DIST = 10.0                         # Length scale for the spatial correlation of the noise on the patch
GDIST_CUTOFF = -NOISE_CORR_DIST * np.log(0.1)  # Cutoff distance for geodesic distance calculation (10%)
SIM_DURATION_S = 60                            # Duration of the simulations
SAMPLING_RATE_HZ = 256                         # Sampling rate
NNOISE = 5000                                  # Number of noise time series
DELAY_S = 10.0                                 # Delay of the seizure onset
SNR_SZ = {1: 1.0, 2: 0.1}                      # Signal-to-noise ratio for seizure noise


def gain_matrix_dipole_verts(vertices, normals, areas, sensors):
    """Gain matrix for projection between the sources and sensors"""

    EPS_MM = 1.0

    nverts = vertices.shape[0]
    nsens = sensors.shape[0]

    gain_mtx = np.zeros((nsens, nverts))
    for isens in range(nsens):
        a = sensors[isens] - vertices
        amag = np.linalg.norm(a, axis=1)
        anorm = a/amag[:, None]
        gain_mtx[isens, :] = areas * np.sum(normals * anorm, axis=1)/((amag + EPS_MM)**2)

    return gain_mtx


def generate_noise_from_profile(duration, sampling_rate, n=1):
    """Noise with a given profile"""

    nsamples = duration * sampling_rate + 1
    nhalf = (nsamples - 1)//2
    x = np.zeros((n, nsamples))

    fk = np.array([k/duration for k in range(1, nhalf + 1)])
    ck = np.interp(fk, NOISE_PROFILE['f'], NOISE_PROFILE['x'], left=0., right=0.)

    for i in range(n):
        phik = np.random.uniform(0, 2*np.pi, size=nhalf)
        xh = np.zeros(nsamples, dtype=complex)
        xh[1:nhalf+1] = ck * np.exp(1j*phik)
        xh[-(nhalf):] = (ck * np.exp(-1j*phik))[::-1]
        x[i, :] = np.fft.ifft(xh).real

        # Normalize
        x[i, :] /= np.sqrt((1./nsamples) * np.sum(np.abs(x[i, :])**2))

    return x

def ffill(arr):
    "Forward fill of NaN values. Works only for 2D arrays, filling along the second dimension."
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def voss_stochastic(nx, nt, ncomp):
    """Voss scheme for noise generation"""

    x = np.random.normal(size=(nx, nt))

    for i in range(1, ncomp):
        xp = np.empty((nx, nt))
        xp.fill(np.nan)
        xp[:, 0] = np.random.normal(size=nx)

        n = int(round(nt / (2 ** i)))

        for j in range(nx):
            inds_t = np.random.choice(nt, size=n, replace=False)
            xp[j, inds_t] = np.random.normal(size=n)

        x += ffill(xp)

    x /= np.sqrt(ncomp)

    return x


def generate_noise(duration, sampling_rate, n=1):
    VOSS_NCOMP = 7
    return voss_stochastic(n, duration * sampling_rate + 1, VOSS_NCOMP)


def create_n_patches(surf, seeds, areas, remove_hanging_nodes=False):
    """Create n surface patches from seeds on surface surf"""

    regmap = -1 * np.ones(surf.nverts, dtype=int)

    # Initialize
    neighbors = [deque() for _ in seeds]
    patch_areas = [0.0 for _ in seeds]
    n_assigned = 0

    for i, seed in enumerate(seeds):
        regmap[seed] = i
        neighbors[i].extend(surf.vert_neighbors[seed])
        n_assigned += 1
        patch_areas[i] += surf.vertex_areas[seed]


    # Expand patches
    while n_assigned < surf.nverts and any([(pa < a) and (len(n) > 0) for pa, a, n
                                            in zip(patch_areas, areas, neighbors)]):
        for i, _ in enumerate(seeds):
            while (patch_areas[i] < areas[i]) and len(neighbors[i]) > 0:
                vert = neighbors[i].popleft()
                if regmap[vert] == -1:
                    regmap[vert] = i
                    n_assigned += 1
                    patch_areas[i] += surf.vertex_areas[vert]
                    neighbors[i].extend(surf.vert_neighbors[vert])
                    break

    if remove_hanging_nodes:
        # Make sure no created patch contains any node that is not part of a triangle
        # That is to make sure a triangulation can be created.
        # It introduces some small error, as small patches of cortex might disappear
        good_verts = np.zeros(surf.nverts, dtype=bool)
        for v1, v2, v3 in surf.triangles:
            if regmap[v1] == regmap[v2] == regmap[v3]:
                good_verts[v1] = True
                good_verts[v2] = True
                good_verts[v3] = True

        # Unassign bad nodes
        regmap[~good_verts] = -1

    return regmap


def divide_to_patches(surf, area):
    """Return mapping of the surface divided into patches with mean area `area`."""

    num_patches = int(surf.area / area)
    seeds = np.random.choice(surf.nverts, size=num_patches, replace=False)
    desired_areas = [np.inf for _ in seeds]     # no limit

    regmap = create_n_patches(surf, seeds, desired_areas, remove_hanging_nodes=False)

    return num_patches, regmap


def compute_gdist_wrapper(verts, triangles, source, queue):
    dist = gdist.compute_gdist(verts, triangles, source)
    queue.put(dist)


def local_gdist_matrix_wrapper(verts, triangles, max_distance, queue):
    mtx = gdist.local_gdist_matrix(verts.astype(np.float64), triangles.astype(np.int32), max_distance)
    queue.put(mtx)


def dist_mtx(verts, triangles, max_distance):

    # Separate thread to avoid the memory leak
    queue = mp.Queue()
    p = mp.Process(target=local_gdist_matrix_wrapper, args=(verts, triangles, max_distance, queue))
    p.start()
    mtx = queue.get()
    p.join()

    return mtx


def wave_box(phase):
    NORM_FACTOR = np.sqrt(16./3.)    # Normalize power
    return NORM_FACTOR * np.where((phase % (2*np.pi)) < np.pi/2., 1., 0.)


def wave_triangle(phase):
    NORM_FACTOR = np.sqrt(3.0)

    phase = phase % (2*np.pi)
    return NORM_FACTOR * np.where(phase < np.pi, 2*phase/np.pi - 1., -2*(phase/np.pi) + 3.)




def generate_source_wavefront(surf, duration, sampling_rate, freq, wave_source,
                              wavefront_velocity, wave_velocity,
                              snr_bg, bg_noise, sz_noise):
    """Generate source activity in the spreading seizure model"""


    nt = duration*sampling_rate + 1
    tt = np.linspace(0, duration, nt)

    # This is dirty...
    # It seems that there is a memory leak in compute_gdist. I don't feel like looking in the C and Cython
    # code, so let's rather work around it: run it in separate process, so that the memory leak is isolated.
    queue = mp.Queue()
    p = mp.Process(target=compute_gdist_wrapper, args=(np.array(surf.vertices),
                                                       np.array(surf.triangles, dtype=np.int32),
                                                       np.array([wave_source], dtype=np.int32),
                                                       queue))
    p.start()
    dist = queue.get()
    p.join()

    x = bg_noise[:, :]

    for it in range(nt):
        szmask = dist < (wavefront_velocity * (tt[it] - DELAY_S))
        x[szmask, it] = np.sqrt(snr_bg) * (wave_box(freq * 2*np.pi*(tt[it] - DELAY_S - dist[szmask] / wave_velocity))
                                           + sz_noise[szmask, it])

    return x


def generate_source_nstat(surf, duration, sampling_rate, regmap, onset_delays, freqs, onset_durations,
                          snr_bg, bg_noise, sz_noise):
    """Generate source activity in the homogeneous source(s) model"""

    nt = duration*sampling_rate + 1
    tt = np.linspace(0, duration, nt)

    x = bg_noise[:, :]

    for i, (freq, onset_duration, onset_delay) in enumerate(zip(freqs, onset_durations, onset_delays)):
        tmask = tt > onset_delay
        smask = regmap == i

        x[np.ix_(smask, tmask)] = np.sqrt(snr_bg) * ((np.minimum(1.0, (tt - onset_delay)/onset_duration))[tmask]
                                                     * (wave_triangle(freq * 2*np.pi*(tt - onset_delay))[tmask]
                                                        + sz_noise[np.ix_(smask, tmask)]))

    return x


def remap(old_regmap, new_n, old_to_new):
    new_regmap = np.zeros(new_n, dtype=old_regmap.dtype)
    for i, value in enumerate(old_regmap):
        if old_to_new[i] != -1:
            new_regmap[old_to_new[i]] = value
    return new_regmap


def cor_clip_eigenvalues(a, threshold=1e-10):
    w, v = sp.linalg.eigh(a)
    a_clipped = v @ np.diag(np.maximum(w, threshold)) @ v.T
    norm = np.sqrt(np.diag(a_clipped))
    a_clipped = a_clipped / norm / norm[:, None]
    return a_clipped



class SimResults():
    """Object to hold the simulation results"""
    def __init__(self, t, source, seeg, surf, surf_sz, regmap, vertex_mapping, model, params):
        self.t = t
        self.source = source
        self.seeg = seeg
        self.surf = surf
        self.surf_sz = surf_sz
        self.regmap = regmap
        self.vertex_mapping = vertex_mapping
        self.model = model
        self.params = params


def simulate_one(sim_duration, model, snr_bg, noise_level, surf, close_verts, gain, noise, params=None, ch_names=None, save=None):
    """Simulate one seizure with model on surface surf"""

    if params is None:
        params = {}

    nt = sim_duration * SAMPLING_RATE_HZ + 1
    nsens = gain.shape[0]

    freq = params.get('freq', np.random.uniform(4., 13.0))

    # Generate patch
    patch_size = params.get('patch_size', np.random.uniform(400., 2500.))

    if model in ['onestat', 'propwaves', 'propsync']:
        patch_center = params['patch_center'] if ('patch_center' in params) else np.random.choice(close_verts)
        regmap = create_n_patches(surf, [patch_center], [patch_size], remove_hanging_nodes=True)
        sz_indices = np.where(regmap == 0)[0]
        surf_sz, vertex_mapping = create_part(surf, sz_indices, mapping=True)
        patch_regmap = remap(regmap, surf_sz.nverts, vertex_mapping)
        sim_params = dict(patch_size=patch_size, patch_center=patch_center, freq=freq, model=model, snr_bg=snr_bg)

    elif model in ['twostat']:
        patch_center1 = params['patch_center1'] if ('patch_center1' in params) else np.random.choice(close_verts)
        try:
            patch_center2 = params['patch_center2']
        except KeyError:
            dists = np.linalg.norm(surf.vertices[close_verts, :] - surf.vertices[patch_center1, :], axis=-1)
            dists[close_verts == patch_center1] = np.inf # To avoid selecting the same one
            patch_center2 = params.get('patch_center2', np.random.choice(close_verts[dists < DIST_TWOSTAT_MM]))

        regmap = create_n_patches(surf, [patch_center1, patch_center2], [patch_size/2, patch_size/2],
                                  remove_hanging_nodes=True)
        sz_indices = np.where(regmap >= 0)[0]
        surf_sz, vertex_mapping = create_part(surf, sz_indices, mapping=True)
        patch_regmap = remap(regmap, surf_sz.nverts, vertex_mapping)
        sim_params = dict(patch_size=patch_size, patch_center1=patch_center1, patch_center2=patch_center2,
                          freq=freq, model=model, snr_bg=snr_bg)
    else:
        raise ValueError("Unknown model")

    # Generate noise patches
    npatches, noise_patch_map = divide_to_patches(surf, NOISE_PATCH_AREA_MM)
    noise_patch_map_ns = np.copy(noise_patch_map)
    noise_patch_map_ns[sz_indices] = -1
    noise_patch_map_sz = remap(noise_patch_map, surf_sz.nverts, vertex_mapping)

    # Data arrays
    seeg = np.zeros((nsens, nt))
    patch_bg_noise = np.zeros((surf_sz.nverts, nt))

    # Generate noise
    for ipatch in range(npatches):
        inoise = np.random.choice(noise.shape[0])
        seeg += np.outer(np.sum(gain[:, noise_patch_map_ns == ipatch], axis=1), noise[inoise])
        patch_bg_noise[noise_patch_map_sz == ipatch, :] = noise[inoise]

    if noise_level == 0:
        patch_sz_noise = np.zeros((surf_sz.nverts, nt))
    elif noise_level > 0:
        # Get distances and symmetrize (small differences may occur in gdist)
        dists = dist_mtx(surf_sz.vertices, surf_sz.triangles, max_distance=GDIST_CUTOFF).todense()
        dists = np.tril(dists) + np.triu(dists.T, 1)

        # Apply kernel
        cor = np.exp(-dists / NOISE_CORR_DIST)
        cor[dists == 0] = 0.
        np.fill_diagonal(cor, 1.)

        # Clip eigenvalues
        cor = cor_clip_eigenvalues(cor)

        # Cholesky
        acor = np.linalg.cholesky(cor)

        # Get and mix noise
        patch_sz_noise_uncorr = noise[np.random.choice(noise.shape[0], size=surf_sz.nverts)]
        patch_sz_noise = np.sqrt(1./SNR_SZ[noise_level]) * (acor @ patch_sz_noise_uncorr)
    else:
        raise ValueError(f"Unexpected noise_level {noise_level}")

    # Generate activity
    if model in ['propsync', 'propwaves']:
        uslow = params.get('uslow', np.random.uniform(0.5, 4.0))

        if model == 'propsync':
            ufast = 100000.
        else:
            ufast = params.get('ufast', np.random.uniform(-500., -100.))

        wave_source = params.get('wave_source', np.random.choice(sz_indices))
        sim_params.update(dict(uslow=uslow, ufast=ufast, wave_source=wave_source))
        source_activity = generate_source_wavefront(surf_sz, sim_duration, SAMPLING_RATE_HZ,
                                                    freq, vertex_mapping[wave_source],
                                                    uslow, ufast,
                                                    snr_bg, patch_bg_noise, patch_sz_noise)

    elif model == 'onestat':
        onset_duration = params.get('onset_duration', np.random.uniform(1., 30.))
        sim_params.update(dict(onset_duration=onset_duration))
        source_activity = generate_source_nstat(surf_sz, sim_duration, SAMPLING_RATE_HZ, patch_regmap,
                                                [DELAY_S], [freq], [onset_duration],
                                                snr_bg, patch_bg_noise, patch_sz_noise)

    elif model == 'twostat':
        onset_duration1 = params.get('onset_duration1', np.random.uniform(1., 30.))
        onset_duration2 = params.get('onset_duration2', np.random.uniform(1., 30.))
        onset_delay = params.get('onset_delay', np.random.uniform(0., 10.))

        sim_params.update(dict(onset_duration1=onset_duration1, onset_duration2=onset_duration2,
                               onset_delay=onset_delay))
        source_activity = generate_source_nstat(surf_sz, sim_duration, SAMPLING_RATE_HZ, patch_regmap,
                                                [DELAY_S, DELAY_S + onset_delay], [freq, freq],
                                                [onset_duration1, onset_duration2],
                                                snr_bg, patch_bg_noise, patch_sz_noise)

    # Project activity
    seeg += gain[:, sz_indices] @ source_activity
    b, a = sig.butter(3, 0.5/(SAMPLING_RATE_HZ/2.), 'high')
    seeg = sig.filtfilt(b, a, seeg, axis=1)

    t = np.linspace(0, sim_duration, nt)

    if save is not None:
        np.savez(save, t=t, seeg=seeg, names=ch_names, regmap=regmap, noisemap=noise_patch_map)

    return SimResults(t, source_activity, seeg, surf, surf_sz, patch_regmap, vertex_mapping, model, sim_params)



def simulate(subject, surface_file, contact_file, model, snr_bg, noise_level, nsims, config_file,
             res_file, grouped_file, plot='none'):
    """Run multiple simulations with given model"""

    np.random.seed(43 + int(subject[2:]))

    img_dir = os.path.join(os.path.split(res_file)[0], f"img/{model}_{noise_level}")
    data_dir = os.path.join(os.path.split(res_file)[0], f"data/{model}_{noise_level}")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Load geometry and setup stuff
    surf = Surface.from_npz_file(surface_file)
    contacts = Contacts(contact_file)

    vert_contact_dist = np.linalg.norm(surf.vertices[:, None, :] - contacts.xyz[None, :, :], axis=-1)
    close_verts = np.where(np.any(vert_contact_dist < DIST_LIM_MM, axis=1))[0]

    gain = gain_matrix_dipole_verts(surf.vertices, surf.vertex_normals, surf.vertex_areas, contacts.xyz)
    noise = generate_noise(SIM_DURATION_S, SAMPLING_RATE_HZ, n=NNOISE)

    rows_config = []
    dfs_solo = []
    dfs_multi = []
    for isim in range(nsims):
        print(isim)

        while True:
            datafile = os.path.join(data_dir, f"{subject}_sim{isim:05d}.npz")
            simres = simulate_one(SIM_DURATION_S, model, snr_bg, noise_level, surf, close_verts, gain, noise, ch_names=contacts.names)
            ans = AnalyzedSeeg(simres.t, simres.seeg, contacts.names, DELAY_S, simres.t[-1])

            if any(ans.is_seizing[0]):
                break
            else:
                print("...failed, running again.")

        fn_template = os.path.join(img_dir, f"{{state}}_{subject}_sim{isim:05d}_{{conf}}_{{channel}}.png")
        ans.plot_all(fn_template, what=plot)


        dist_to_cortex  = np.min(np.linalg.norm(surf.vertices[:, None, :]           - contacts.xyz[None, :, :], axis=2), axis=0)
        dist_to_szpatch = np.min(np.linalg.norm(simres.surf_sz.vertices[:, None, :] - contacts.xyz[None, :, :], axis=2), axis=0)

        rows_config.append(simres.params)

        for k in range(NCONF):
            df_solo = pd.DataFrame(
                dict(model=model, noise=noise_level, subject=subject, rec=isim, detconf=k,
                     contact=contacts.names, is_seizing=ans.is_seizing[k], is_taa=ans.is_taa[k],
                     tfr=ans.tfr[k], tto=ans.tto[k], freq=ans.freq[k],
                     lnta_p80=np.percentile(ans.lnta, 80, axis=1),
                     dist_to_cortex=dist_to_cortex, dist_to_szpatch=dist_to_szpatch)
            )
            add_elec_info(df_solo, contacts)
            df_multi = group_taas(df_solo, simres.t, simres.seeg, contacts.names, contact_file)

            dfs_solo.append(df_solo)
            dfs_multi.append(df_multi)

    if config_file is not None:
        pd.DataFrame(rows_config).to_pickle(config_file)

    if res_file is not None:
        pd.concat(dfs_solo, ignore_index=True).to_pickle(res_file)

    if grouped_file is not None:
        pd.concat(dfs_multi, ignore_index=True).to_pickle(grouped_file)



def mesh_dependence(subject, surf0_file, surf1_file, contact_file, snr_bg, noise_level, nsims,
                    config_file, res_file, grouped_file):
    """Mesh dependence study: same simulations on two differently resolved surfaces"""


    # Load surfaces
    surf0 = Surface.from_npz_file(surf0_file)
    surf1 = Surface.from_npz_file(surf1_file)
    surf1.set_vertex_neighbors()
    contacts = Contacts(contact_file)

    vcd0 = np.linalg.norm(surf0.vertices[:, None, :] - contacts.xyz[None, :, :], axis=-1)
    vcd1 = np.linalg.norm(surf1.vertices[:, None, :] - contacts.xyz[None, :, :], axis=-1)
    close_verts0 = np.where(np.any(vcd0 < DIST_LIM_MM, axis=1))[0]
    close_verts1 = np.where(np.any(vcd1 < DIST_LIM_MM, axis=1))[0]

    gain0 = gain_matrix_dipole_verts(surf0.vertices, surf0.vertex_normals, surf0.vertex_areas, contacts.xyz)
    gain1 = gain_matrix_dipole_verts(surf1.vertices, surf1.vertex_normals, surf1.vertex_areas, contacts.xyz)

    dist_to_cortex0  = np.min(np.linalg.norm(surf0.vertices[:, None, :] - contacts.xyz[None, :, :], axis=2), axis=0)
    dist_to_cortex1  = np.min(np.linalg.norm(surf1.vertices[:, None, :] - contacts.xyz[None, :, :], axis=2), axis=0)

    noise = generate_noise(SIM_DURATION_S, SAMPLING_RATE_HZ, n=NNOISE)

    rows_config = []
    dfs_solo = []
    dfs_multi = []

    for isim in range(nsims):
        print(isim)

        while True:
            # Generate params
            freq = np.random.uniform(4., 13.0)
            patch_size = np.random.uniform(400., 2500.)
            uslow = np.random.uniform(0.5, 4.0)
            ufast = np.random.uniform(-500, -100)

            patch_center0 = np.random.choice(close_verts0)
            patch_center1 = np.argmin(np.linalg.norm(surf1.vertices - surf0.vertices[patch_center0], axis=-1))

            # Build surface
            regmap0 = create_n_patches(surf0, [patch_center0], [patch_size], remove_hanging_nodes=True)
            sz_indices0 = np.where(regmap0 == 0)[0]

            # Resample to higher ------------------------------------------------------------#
            # This is little messy, but quick explanation:
            # We create the coarse surface, then in the fine surface we find the nodes in the coarse seizure patch.
            # Then we add the nodes which were created by refinement. To do so, we take the neighbors of the
            # fine seizure patch, and if the neighbor lie in the midpoint between two vertices in the fine patch,
            # we add it as well.

            regmap1p = create_n_patches(surf1, [patch_center1], [4*patch_size], remove_hanging_nodes=True)
            sz_indices1p = np.where(regmap1p == 0)[0]

            dists = np.linalg.norm(surf0.vertices[sz_indices0, None, :] - surf1.vertices[None, sz_indices1p, :], axis=2)
            retain = np.min(dists, axis=0) < 1e-10

            szmask = np.zeros(surf1.nverts, dtype=bool)
            szmask[sz_indices1p[retain]] = True
            newszmask = np.copy(szmask)

            for ind in sz_indices1p[retain]:
                for nind in surf1.vert_neighbors[ind]:

                    # Two neighbors are retained
                    neighs = np.array(list(surf1.vert_neighbors[nind]), dtype=int)
                    if sum(szmask[neighs]) == 2:
                        n1, n2 = neighs[szmask[neighs]][[0, 1]]

                        # And the new one lies in the middle between them
                        if (   np.linalg.norm(surf1.vertices[n1] - surf1.vertices[nind])
                             + np.linalg.norm(surf1.vertices[n2] - surf1.vertices[nind])
                             < np.linalg.norm(surf1.vertices[n1] - surf1.vertices[n2]) + 1e-6):

                            newszmask[nind] = True

            sz_indices1 = np.where(newszmask)[0]
            regmap1 = -1 * np.ones(surf1.nverts, dtype=int)
            regmap1[sz_indices1] = 0

            # ---------------------------------------------------------------------------- #

            # Wave sources
            ws0 = np.random.choice(sz_indices0)
            ws1 = np.argmin(np.linalg.norm(surf1.vertices - surf0.vertices[ws0], axis=-1))

            params = {'freq': freq, 'patch_size': patch_size, 'uslow': uslow, 'ufast': ufast}

            # Use wave_source == patch_center to avoid building the seizure surface
            params0 = {**params, 'patch_center': patch_center0, 'wave_source': ws0, 'regmap': regmap0, 'sz_indices': sz_indices0}
            params1 = {**params, 'patch_center': patch_center1, 'wave_source': ws1, 'regmap': regmap1, 'sz_indices': sz_indices1}

            # Simulate
            simres0a = simulate_one(SIM_DURATION_S, 'propwaves', snr_bg, noise_level, surf0, close_verts0, gain0, noise, params=params0)
            ans0a = AnalyzedSeeg(simres0a.t, simres0a.seeg, contacts.names, DELAY_S, simres0a.t[-1])

            if any(ans0a.is_seizing):
                simres0b = simulate_one(SIM_DURATION_S, 'propwaves', snr_bg, noise_level, surf0, close_verts0, gain0, noise, params=params0)
                ans0b = AnalyzedSeeg(simres0b.t, simres0b.seeg, contacts.names, DELAY_S, simres0b.t[-1])

                simres1 = simulate_one(SIM_DURATION_S, 'propwaves', snr_bg, noise_level, surf1, close_verts1, gain1, noise, params=params1)
                ans1 = AnalyzedSeeg(simres1.t, simres1.seeg, contacts.names, DELAY_S, simres1.t[-1])

                break
            else:
                print("...failed, running again")


        rows_config.append(simres0a.params)

        for model, ans, simres, d2c in [('propwaves0a', ans0a, simres0a, dist_to_cortex0),
                                        ('propwaves0b', ans0b, simres0b, dist_to_cortex0),
                                        ('propwaves1',  ans1,  simres1,  dist_to_cortex1)]:

            df_solo = pd.DataFrame(
                dict(model=model, noise=noise_level, subject=subject, rec=isim, detconf=0,
                     contact=contacts.names, is_seizing=ans.is_seizing[0], is_taa=ans.is_taa[0],
                     tfr=ans.tfr[0], tto=ans.tto[0], freq=ans.freq[0],
                     lnta_p80=np.percentile(ans.lnta, 80, axis=1),
                     dist_to_cortex=d2c, dist_to_szpatch=0.)
            )
            add_elec_info(df_solo, contacts)
            df_multi = group_taas(df_solo, simres.t, simres.seeg, contacts.names, contact_file)

            dfs_solo.append(df_solo)
            dfs_multi.append(df_multi)


    if config_file is not None:
        pd.DataFrame(rows_config).to_pickle(config_file)

    if res_file is not None:
        pd.concat(dfs_solo, ignore_index=True).to_pickle(res_file)

    if grouped_file is not None:
        pd.concat(dfs_multi, ignore_index=True).to_pickle(grouped_file)
