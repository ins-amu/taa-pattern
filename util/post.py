"""
Analysis of the recorded or simulated SEEG signals
"""


import glob
import os
import itertools
from collections import OrderedDict

import mne
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import stats
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from .contacts import Contacts
from .surface import Surface

# Detection configs are in the format
# (LTA_SZ_THRESHOLD, REL_THRESHOLD_LOWER, REL_THRESHOLD_UPPER)
DETECTION_CONFIGS = [
    (np.log10(30.), 0.15, 0.85),

    (np.log10(10.), 0.15, 0.85),
    (np.log10(20.), 0.15, 0.85),
    (np.log10(40.), 0.15, 0.85),
    (np.log10(50.), 0.15, 0.85),

    (np.log10(30.), 0.100, 0.85),
    (np.log10(30.), 0.125, 0.85),
    (np.log10(30.), 0.175, 0.85),
    (np.log10(30.), 0.200, 0.85),

    (np.log10(30.), 0.15, 0.800),
    (np.log10(30.), 0.15, 0.825),
    (np.log10(30.), 0.15, 0.875),
    (np.log10(30.), 0.15, 0.900)
]
NCONF = len(DETECTION_CONFIGS)

MAX_TAA_DURATION = 20.0
MIN_TAA_R2 = 0.75

# Peak analysis
MIN_PSD_INTERVAL = 8.0
DECIM_FS = 32.
PEAK_HEIGHT = 0.25
PEAK_DIST = 1.0
PEAK_PROMINENCE = 0.1
HARMONIC_LIM = 0.15

PCA_NCOMP = 4


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

def smoothen(x, fs, window, axis=-1):
    """
    Smoothen over the last dimension.

    Based on http://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
    """
    window_len = round_up_to_odd(window * fs)
    wh = window_len // 2

    xm = np.moveaxis(x, axis, -1)

    if xm.shape[-1] < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    w = np.ones(window_len, 'd')
    xsm = np.zeros_like(xm)

    inds = itertools.product(*[range(_) for _ in xm.shape[:-1]])
    for ind in inds:
        xp = np.r_[np.repeat(xm[ind][0], wh), xm[ind], np.repeat(xm[ind][-1], wh)]
        xsm[ind] = np.convolve(w/w.sum(), xp, mode='valid')

    return np.moveaxis(xsm, -1, axis)


class AnalyzedSeeg():
    """Object to hold the results of analysis"""

    def __init__(self, time, seeg, channel_names, t_onset, t_term):
        nc, _ = seeg.shape
        assert len(channel_names) == nc

        self.time = time
        self.seeg = seeg
        self.names = channel_names
        self.t_onset = t_onset
        self.t_term = t_term

        self.lnta_upper_threshold = np.zeros((NCONF, nc), dtype=float)
        self.lnta_lower_threshold = np.zeros((NCONF, nc), dtype=float)
        self.is_seizing = np.zeros((NCONF, nc), dtype=bool)
        self.is_taa = np.zeros((NCONF, nc), dtype=bool)
        self.tfr = np.zeros((NCONF, nc), dtype=float)
        self.tto = np.zeros((NCONF, nc), dtype=float)
        self.freq = np.zeros((NCONF, nc), dtype=float)

        self.slope = np.zeros((NCONF, nc))
        self.r2 = np.zeros((NCONF, nc))
        self.intercept = np.zeros((NCONF, nc))

        self.sa_freqs  = np.full((NCONF, nc), None, dtype=object)
        self.sa_spect  = np.full((NCONF, nc), None, dtype=object)
        self.sa_fpeaks = np.full((NCONF, nc), None, dtype=object)
        self.sa_peaks  = np.full((NCONF, nc), None, dtype=object)

        self.analyze()


    def analyze(self):
        nc, nt = self.seeg.shape

        fs = 1. / (self.time[1] - self.time[0])
        decim_factor = round(float(fs / DECIM_FS))
        time_ds = self.time[::decim_factor]

        # TFA
        freqs = np.linspace(4, 13.0, 20)
        tfr = mne.time_frequency.tfr_array_multitaper(np.expand_dims(self.seeg[:, :], 0), fs, freqs,
                                                      time_bandwidth=2.0, n_cycles=8,
                                                      decim=decim_factor, zero_mean=False, output='avg_power', n_jobs=1)

        band_ta = np.mean(tfr[:, :, :], axis=1)
        baseline_ta = np.mean(band_ta[:, time_ds < self.t_onset], axis=1)
        mask_ds = (time_ds >= self.t_onset - 10.0) * (time_ds <= self.t_term)
        norm_ta = (band_ta / baseline_ta[:, None])
        time_ds_sz = time_ds[mask_ds]

        self.t_lnta = time_ds_sz
        self.lnta = np.log10(norm_ta[:, mask_ds])

        lnta_p90 = np.percentile(self.lnta, 90, axis=1)

        for k in range(len(DETECTION_CONFIGS)):
            LNTA_SZ_THRESHOLD, REL_THRESHOLD_LOWER, REL_THRESHOLD_UPPER = DETECTION_CONFIGS[k]

            self.lnta_upper_threshold[k] = REL_THRESHOLD_UPPER * lnta_p90
            self.lnta_lower_threshold[k] = REL_THRESHOLD_LOWER * lnta_p90
            self.is_seizing[k] = (lnta_p90 >= LNTA_SZ_THRESHOLD)

            for i in range(nc):
                if self.is_seizing[k, i]:
                    try:
                        ito = np.where(self.lnta[i] > self.lnta_upper_threshold[k,i])[0][0]
                        ifr = np.where(self.lnta[i, :ito] < self.lnta_lower_threshold[k,i])[0][-1]
                        self.tfr[k,i], self.tto[k,i] = time_ds_sz[ifr], time_ds_sz[ito]
                        self.slope[k,i], self.intercept[k,i], rval, _, _ = stats.linregress(
                            time_ds_sz[ifr:ito], self.lnta[i, ifr:ito])
                        self.r2[k,i] = rval**2
                    except IndexError:
                        pass

                    if ((self.tto[k,i] - self.tfr[k,i]) <= MAX_TAA_DURATION) and (self.r2[k,i] > MIN_TAA_R2):
                        taa_mask = (self.time > self.tfr[k,i]) * (self.time < max(self.tto[k,i], self.tfr[k,i] + MIN_PSD_INTERVAL))
                        try:
                            psds, sa_freqs = mne.time_frequency.psd_array_multitaper(
                                np.expand_dims(self.seeg[i][taa_mask], 0), fs, fmax=100., verbose='WARNING')
                        except ValueError:
                            continue

                        spect = psds[0, :] * sa_freqs
                        spect /= np.max(spect)
                        peaks = sig.find_peaks(spect, distance=PEAK_DIST / (sa_freqs[1] - sa_freqs[0]), height=PEAK_HEIGHT,
                                               prominence=PEAK_PROMINENCE)[0]
                        fpeaks = [sa_freqs[peak] for peak in peaks]

                        self.sa_freqs[k,i] = sa_freqs
                        self.sa_spect[k,i] = spect
                        self.sa_fpeaks[k,i] = fpeaks
                        self.sa_peaks[k,i] = spect[peaks]

                        if len(peaks) > 0:
                            ks = np.array([round(k) for k in fpeaks / fpeaks[0]])
                            fund = np.sum(fpeaks) / np.sum(ks)  # Minimizes: sum_i (f_i - k_i*fund)**2

                            if ((4.0 <= fund <= 13.0)
                                and all(spect[peaks[0]] > spect[peaks[1:]])
                                and all(np.abs(fpeaks / fund - ks) < HARMONIC_LIM)):

                                self.is_taa[k,i] = True
                                self.freq[k,i] = fund


    def plot_all(self, filename_template, what, k=0):
        for i, channel_name in enumerate(self.names):
            if (what == 'all') or (what == 'seizing' and self.is_seizing[k,i]) or (what == 'taa' and self.is_taa[k,i]):
                state = 'taa' if self.is_taa[k,i] else ('seizing' if self.is_seizing[k,i] else 'nonseizing')
                img_file = filename_template.format(channel=channel_name, state=state, conf=k)
                self.plot_one(k, i, img_file)


    def plot_one(self, k, i, img_file):
        plot_mask = (self.time > self.t_onset - 20.) * (self.time <= self.t_term + 20.)
        t = self.time[plot_mask]
        x = self.seeg[i, plot_mask]

        plt.figure(figsize=(18, 14))

        ax = plt.subplot(2, 2, 1)
        plt.title(self.names[i])
        plt.plot(t, x / np.max(np.abs(x)), color='b', lw=0.5)
        plt.ylim([-3.0, 3.0])

        plt.plot(self.t_lnta, self.lnta[i], color='g')
        plt.axhline(self.lnta_lower_threshold[k,i], color='0.5', ls='--')
        plt.axhline(self.lnta_upper_threshold[k,i], color='0.5', ls='--')
        if self.is_seizing[k,i]:
            plt.axvspan(self.tfr[k,i], self.tto[k,i], color='orange', alpha=0.2)

        # Time series detail
        plt.subplot(2, 2, 3)
        if self.is_seizing[k,i]:
            plt.plot(t, x / np.max(np.abs(x)), color='b', lw=0.5)
            plt.xlim(self.tfr[k,i] - 5.0, self.tto[k,i] + 5.0)
            plt.plot(self.t_lnta, self.lnta[i], color='g')
            t = np.linspace(self.tfr[k,i], self.tto[k,i], 2)
            plt.plot(t, self.intercept[k,i] + t * self.slope[k,i], color='r', ls='--', zorder=-1)
            plt.axhline(self.lnta_lower_threshold[k,i], color='0.5', ls='--')
            plt.axhline(self.lnta_upper_threshold[k,i], color='0.5', ls='--')
            plt.axvspan(self.tfr[k,i], self.tto[k,i], color='orange', alpha=0.2)
            plt.text(0.1, 0.9, f"r2 = {self.r2[k,i]:0.3f}\nduration = {self.tto[k,i] - self.tfr[k,i]:.2f}",
                     transform=plt.gca().transAxes)
        plt.ylim([-2.0, 2.0])

        # Spectral analysis
        if self.sa_freqs[k,i] is not None:
            plt.subplot(2, 2, 2)
            plt.plot(self.sa_freqs[k,i], self.sa_spect[k,i])
            plt.scatter(self.sa_fpeaks[k,i], self.sa_peaks[k,i], color='r')
            plt.axvspan(4.0, 13.0, color='orange', alpha=0.2)
            plt.xlim([0, 50])
            plt.axhline(PEAK_HEIGHT, color='r', ls='--')
            plt.text(0.8, 0.9, f"Is TAA: {self.is_taa[k,i]}", transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(img_file)
        plt.close()



def get_consecutive_contacts(dfs, consec_min):
    """Get consecutive contacts"""

    if len(dfs) == 0:
        return []

    consecs = []
    consec = [next(dfs.iterrows())[0]]

    for (i1, r1), (i2, r2) in zip(dfs[:-1].iterrows(), dfs[1:].iterrows()):
        if all([r1[a] == r2[a] for a in ['subject', 'rec', 'elec']]) and (r1.elecn + 1 == r2.elecn):
            consec.append(i2)
        else:
            consecs.append(consec)
            consec = [i2]
    consecs.append(consec)
    return [c for c in consecs if len(c) >= consec_min]


def get_taa_group_features(t, seeg, coords, tfr, tto):
    mask = (t > np.min(tfr)) * (t < max(np.max(tto), np.min(tfr) + 5.0))
    seeg -= np.mean(seeg, axis=1)[:, None]

    pca = PCA(n_components=PCA_NCOMP)
    comps = pca.fit_transform(seeg.T)
    var_explained = pca.explained_variance_ratio_
    duration = np.mean(tto - tfr)

    line_coords = np.linalg.norm(coords - coords[0], axis=1)
    slope, _, rval, _, _ = stats.linregress(line_coords, tfr)

    return duration, abs(slope), rval**2, var_explained[0], sum(var_explained[0:2])



def group_taas(df, t, seeg, ch_names, contact_file):
    """Find the TAA groups"""

    CONSEC_MIN = 4

    df['is_ingroup'] = False
    consecs = get_consecutive_contacts(df[df.is_taa], CONSEC_MIN)

    rows = []
    for i, consec in enumerate(consecs):
        df.loc[consec, 'is_ingroup'] = True

        dff = df.loc[consec]

        fst = dff.iloc[0]

        mask = (t > np.min(dff.tfr)) * (t < max(np.max(dff.tto), np.min(dff.tfr) + 5.0))
        inds = [ch_names.index(ch) for ch in dff.contact]

        seegt = seeg[inds][:, mask]
        seegt -= np.mean(seegt, axis=1)[:, None]

        pca = PCA(n_components=PCA_NCOMP)
        comps = pca.fit_transform(seegt.T)
        var_explained = pca.explained_variance_ratio_

        cc = np.corrcoef(seegt)

        contacts = list(dff.contact)

        cnames = list(np.genfromtxt(contact_file, usecols=(0,), dtype=str))
        cpos = np.genfromtxt(contact_file, usecols=(1,2,3), dtype=float)
        cpos_taa = np.array([cpos[cnames.index(name)] for name in contacts])
        line_coord = np.linalg.norm(cpos_taa - cpos_taa[0], axis=1)
        slope, _, rval, _, _ = stats.linregress(line_coord, list(dff.tfr))

        rows.append(OrderedDict(
            model=fst.model,
            noise=fst.noise,
            subject=fst.subject,
            rec=fst.rec,
            detconf=fst.detconf,
            ncontacts=len(dff),
            contacts=contacts,
            tfr=list(dff.tfr),
            tto=list(dff.tto),
            tdur=np.mean(dff.tto - dff.tfr),
            freq=list(dff.freq),

            cc=cc,

            pca_ve1=var_explained[0],
            pca_ve2=var_explained[1],
            pca_ve3=var_explained[2],
            pca_ve4=var_explained[3],
            pca_ve2ac=var_explained[0] + var_explained[1],

            tfr_slope=np.abs(slope),
            tfr_r2=rval**2,

            dist_to_cortex=np.mean(dff.dist_to_cortex),
            dist_to_szpatch=np.mean(dff.dist_to_szpatch),
        ))

    dfm = pd.DataFrame(rows)
    return dfm


def add_elec_info(df, contacts):
    df['elec'] = None
    df['elecn'] = None

    for i, row in df.iterrows():
        elec, num = contacts.split(row.contact)
        df.loc[i, 'elec'] = elec
        df.loc[i, 'elecn'] = num


def get_taa_in_recordings(subject, surface_file, contact_file, rec_directory, solo_file, grouped_file, plot='none'):
    """Find the TAA instances in the recording"""

    img_dir = os.path.join(os.path.split(solo_file)[0], "img")
    os.makedirs(img_dir, exist_ok=True)

    contacts = Contacts(contact_file)
    surf = Surface.from_npz_file(surface_file)
    dist_to_cortex = np.min(np.linalg.norm(surf.vertices[:, None, :] - contacts.xyz[None, :, :], axis=2), axis=0)

    dfs_solo = []
    dfs_multi = []
    for filename in glob.glob(os.path.join(rec_directory, 'rec_*.npz')):
        rec = os.path.splitext(os.path.split(filename)[1])[0]
        rid = int(rec[4:])

        res = np.load(filename)
        ans = AnalyzedSeeg(res['t'], res['seeg'], res['names'], res['onset'], res['termination'])

        fn_template = os.path.join(img_dir, f"{{state}}_{subject}_{rec}_{{conf}}_{{channel}}.png")
        ans.plot_all(fn_template, what=plot)

        cnames = list(res['names'])
        dists_to_ctx = [dist_to_cortex[contacts.names.index(c)] for c in cnames]

        for k in range(NCONF):
            df_solo = pd.DataFrame(
                dict(model=None, noise=None, subject=subject, rec=rid, detconf=k,
                     contact=cnames, is_seizing=ans.is_seizing[k], is_taa=ans.is_taa[k],
                     tfr=ans.tfr[k], tto=ans.tto[k], freq=ans.freq[k],
                     lnta_p80=np.percentile(ans.lnta, 80, axis=1),
                     dist_to_cortex=dists_to_ctx, dist_to_szpatch=0.0)
            )
            add_elec_info(df_solo, contacts)
            df_multi = group_taas(df_solo, res['t'], res['seeg'], list(res['names']), contact_file)

            dfs_solo.append(df_solo)
            dfs_multi.append(df_multi)

    pd.concat(dfs_solo, ignore_index=True).to_pickle(solo_file)
    pd.concat(dfs_multi, ignore_index=True).to_pickle(grouped_file)
