"""
Preparation of the data.
"""


import os
import shutil

import util
import pipelineloader as pl

SIDS = range(1, 51)

localrules: prepGeom, prepGeomFine, getSeeg, allGeom, allSeeg

def prep_geometry(sid, output_dir, level=0):
    os.makedirs(output_dir)
    subj = pl.Subject(sid)
    util.geometry.gen_full_cort_surf(subj.direc, level=level, out_dir=output_dir)
    shutil.copyfile(os.path.join(subj.direc, "elec", "seeg.xyz"), os.path.join(output_dir, "seeg.txt"))


rule prepGeom:
    input:
    output: directory("data/Geometry/id{sid}")
    run: prep_geometry(int(wildcards.sid), output[0])


rule prepGeomFine:
    input:
    output: directory("data/GeometryFine/id{sid}")
    run: util.geometry.prep_geometry(int(wildcards.sid), output[0], level=1)


rule getSeeg:
    input:
    output: directory("data/Recordings/id{sid}")
    shell:
        "python util/pre.py {wildcards.sid} {output[0]}"

rule allGeom:
    input: expand("data/Geometry/id{sid:03d}", sid=SIDS)

rule allSeeg:
    input: expand("data/Recordings/id{sid:03d}", sid=SIDS)
