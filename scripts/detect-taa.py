"""
Detect TAA patterns in a recording.
"""

import os
import numpy as np
import pandas as pd

from util import post
from util.contacts import Contacts

subject = "id001"
rec_index = 1
rec_file = f"data/Recordings/{subject}/rec_{rec_index:04d}.npz"    # Seizure file
contact_file = f"data/Geometry/{subject}/seeg.txt"                 # SEEG contact file
conf = 0                                                           # The default detection configuration


# Find the TAA instances
rec = np.load(rec_file)
ans = post.AnalyzedSeeg(rec['t'], rec['seeg'], rec['names'], rec['onset'], rec['termination'])
contacts = Contacts(contact_file)
nc = len(contacts.names)

print("===========================================")
print("Contact name   Is TAA    From [s]    To [s]")
print("-------------------------------------------")
for i in range(nc):
    if ans.is_taa[conf, i]:
        print(f"{contacts.names[i]:13s}     Yes       {ans.tfr[0, i]:5.2f}     {ans.tto[0, i]:5.2f}")
    else:
        print(f"{contacts.names[i]:13s}     No")
print("===========================================")



# Find the TAA groups
elec, elecn = zip(*[contacts.split(n) for n in contacts.names])
df = pd.DataFrame(dict(subject=subject, rec=rec_index, elec=elec, elecn=elecn, is_taa=ans.is_taa[conf]))
groups = post.get_consecutive_contacts(df[df.is_taa], 4)

for group in groups:
    features = post.get_taa_group_features(rec['t'], rec['seeg'][group], contacts.xyz[group],
                                           ans.tfr[conf, group], ans.tto[conf, group])
    print("--------------------------------------")
    print(f"TAA group: {[contacts.names[i] for i in group]}")
    print(f"duration = {features[0]:6.2f} s")
    print(f"slope    = {features[1]:6.2f} mm/s")
    print(f"R^2      = {features[2]:6.2f}")
    print(f"PCA VE1  = {features[3]:6.2f}")
    print(f"PCA VE2  = {features[4]:6.2f}")
    print("--------------------------------------")
