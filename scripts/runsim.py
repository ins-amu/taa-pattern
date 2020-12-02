"""
Run one simulation with spreading seizure model.
"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

from util import sim, geometry
from util.surface import Surface
from util.contacts import Contacts


DIST_LIM_MM = 15.0            # Max distance of the patch center from the contacts
SIM_DURATION_S = 60           # Duration of the simulation (in seconds)
SAMPLING_RATE_HZ = 256        # Sampling rate of the generated signal
NNOISE = 1000                 # Number of noise time series. For effectivity reasons, noise is pregenerated before
                              # the simulations and potentially reused. NNOISE should thus ideally be higher than
                              # number of vertices on the simulated patch.
SNR_BG = 160.00               # Signal-to-noise ratio for background noise
NOISE_LEVEL = 1               # Noisy simulation
MODEL = "propwaves"           # Spreading seizure simulation
SUBJECT = "id001"             # Subject name


# Set seed for reproducibility
np.random.seed(42)

# Load the surface and the contacts of the implanted electrodes
surf = Surface.from_npz_file(f"data/Geometry/{SUBJECT}/surface.npz")
contacts = Contacts(f"data/Geometry/{SUBJECT}/seeg.txt")

# Find vertices close to the electrode
vert_contact_dist = np.linalg.norm(surf.vertices[:, None, :] - contacts.xyz[None, :, :], axis=-1)
close_verts = np.where(np.any(vert_contact_dist < DIST_LIM_MM, axis=1))[0]

# Calculate the gain matrix
gain = sim.gain_matrix_dipole_verts(surf.vertices, surf.vertex_normals, surf.vertex_areas, contacts.xyz)

# Generate the noise
noise = sim.generate_noise(SIM_DURATION_S, SAMPLING_RATE_HZ, n=NNOISE)

# Run the simulation
res = sim.simulate_one(SIM_DURATION_S, MODEL, SNR_BG, NOISE_LEVEL, surf, close_verts, gain, noise,
                       ch_names=contacts.names)


# Plot
nc = res.seeg.shape[0]
plt.figure(figsize=(16,12))
for i in range(nc):
    plt.plot(res.t, res.seeg[i] / np.max(np.abs(res.seeg)) + i, color='k', lw=0.5)
plt.yticks(np.r_[:nc], contacts.names)
plt.ylim(nc+0.5, -1.5)
plt.xlabel("Time [s]")
plt.xlim(0, SIM_DURATION_S)
plt.tight_layout()
plt.savefig("seizure.png")


# Save data
np.savez("seizure.npz", seeg=res.seeg, t=res.t, names=contacts.names,
         onset=sim.DELAY_S, termination=SIM_DURATION_S, subject=SUBJECT, rec=0)
