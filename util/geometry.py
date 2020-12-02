"""
Dealing with triangulated surfaces
"""


import os
import shutil

import numpy as np
import nibabel as nib

from .surface import Surface
from . import triang


def merge_surfaces(surfaces):
    offsets = np.cumsum([0] + [vs.shape[0] for vs in [surf.vertices for surf in surfaces]][:-1])
    vertices = np.vstack([surf.vertices for surf in surfaces])
    triangles = np.vstack([ts + offset for ts, offset in zip([surf.triangles for surf in surfaces], offsets)])
    region_mappings = np.hstack([surf.region_mapping for surf in surfaces])
    return Surface(vertices, triangles, region_mappings, basic=True)


def create_part(surf, inds, mapping=False):
    """
    Return new surface created from `inds` of the old surface `surf`.

    If mapping == True, the mapping of the old to new vertices is returned as well.
    """

    inds = sorted(inds)

    # Vertices
    old_to_new = -1 * np.ones(surf.vertices.shape[0], dtype=int)
    for new, old in enumerate(inds):
        old_to_new[old] = new
    verts_part = surf.vertices[inds, :]

    # Region mapping
    region_inds = np.unique(surf.region_mapping[inds])
    regmap_old_to_new = dict(zip(region_inds, np.r_[:len(region_inds)]))
    regmap_part = np.array([regmap_old_to_new[reg] for reg in surf.region_mapping[inds]])
    new_names = None
    if surf.region_names is not None:
        new_names = [surf.region_names[i] for i in region_inds]

    # Triangles
    triangs_part = []
    for t1, t2, t3 in surf.triangles:
        t1n = old_to_new[t1]
        t2n = old_to_new[t2]
        t3n = old_to_new[t3]

        if t1n != -1 and t2n != -1 and t3n != -1:
            triangs_part.append([t1n, t2n, t3n])

    triangs_part = np.array(triangs_part)

    surf = Surface(verts_part, triangs_part, regmap_part, region_names=new_names, basic=True)
    if not mapping:
        return surf
    else:
        return surf, old_to_new



def get_cortical_surfaces(cort_surf_direc, label_direc):
    vl_pial, tl_pial, mtl = nib.freesurfer.io.read_geometry(os.path.join(cort_surf_direc, "lh.pial"), read_metadata=True)
    vr_pial, tr_pial, mtr = nib.freesurfer.io.read_geometry(os.path.join(cort_surf_direc, "rh.pial"), read_metadata=True)
    vl_pial += mtl['cras']
    vr_pial += mtr['cras']

    vl_wmgm, tl_wmgm, mtl = nib.freesurfer.io.read_geometry(os.path.join(cort_surf_direc, "lh.white"), read_metadata=True)
    vr_wmgm, tr_wmgm, mtr = nib.freesurfer.io.read_geometry(os.path.join(cort_surf_direc, "rh.white"), read_metadata=True)
    vl_wmgm += mtl['cras']
    vr_wmgm += mtr['cras']

    verts_l = (vl_pial + vl_wmgm)/2.
    verts_r = (vr_pial + vr_wmgm)/2.

    region_mapping_l, _, _ = nib.freesurfer.io.read_annot(os.path.join(label_direc, "lh.aparc.annot"))
    region_mapping_r, _, _ = nib.freesurfer.io.read_annot(os.path.join(label_direc, "rh.aparc.annot"))

    regmap_l = np.zeros_like(region_mapping_l)
    regmap_r = np.zeros_like(region_mapping_r)

    regmap_l[region_mapping_l > 0] = 1
    regmap_r[region_mapping_r > 0] = 1

    sl = Surface(verts_l, tl_pial, regmap_l, basic=True)
    sr = Surface(verts_r, tr_pial, regmap_r, basic=True)

    surface = merge_surfaces([sl, sr])
    surface.set_region_names(["Unknown", "Cortex"])

    return surface


def gen_full_cort_surf(subject_dir, level, out_dir):
    surf = get_cortical_surfaces(os.path.join(subject_dir, "surf"), os.path.join(subject_dir, "label"))

    # Remove unknown area
    surf = create_part(surf, np.where(surf.region_mapping != 0)[0])
    surf = triang.refine(surf, level)
    surf.compute_all()

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    surf.save_surf_npz(os.path.join(out_dir, "surface.npz"))


