"""
SEEG signals preprocessing
"""


import os
import sys

import numpy as np
import scipy.signal as sig

import pipelineloader as pl

FINAL_FS = 256.
HIGHPASS_LIM_HZ = 0.5

# Some seizures are artefacted, so we exclude them
EXCLUDE = [("id038_ct", "CT_C1_PSG_170623M-BEX_0002.json"), ("id038_ct", "CT_C2_PSG_170623M-BEX_0008.json")]


def prepare_seeg(subj_ids, out_direc):
    if type(subj_ids) == str:
        subj_ids = [int(a) for a in subj_ids.split(",")]

    os.makedirs(out_direc)

    for subj_id in subj_ids:

        subj = pl.Subject(subj_id)
        for rid, rec in enumerate(subj.seizure_recordings):
            if (subj.name, os.path.basename(rec.json_file)) in EXCLUDE:
                continue
            else:
                print(subj.name, rid)

            rec.load()

            t = rec.time
            ttot = t[-1] - t[0]
            data = rec.get_data_monopolar()

            b, a = sig.butter(3, HIGHPASS_LIM_HZ/(rec.sfreq/2.), 'high')
            data = sig.filtfilt(b, a, data, axis=1)

            downsample_factor = round(rec.sfreq/FINAL_FS)
            datar = sig.resample_poly(data, up=1, down=downsample_factor, axis=1)
            tr = t[::downsample_factor]

            filename = os.path.join(out_direc, f"rec_{rid:04d}.npz")
            np.savez(filename, seeg=datar, names=rec.get_ch_names_monopolar(), t=tr,
                     onset=rec.onset, termination=rec.termination, subject=subj.name[:5], rec=rid)

            rec.clear()



if __name__ == '__main__':
    prepare_seeg(*sys.argv[1:])
