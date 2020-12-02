"""
Manipulating with triangulated surface
"""

import logging
import os.path
import tempfile
import shutil
from zipfile import ZipFile

import numpy as np

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

logger = get_logger()


class Surface:
    def __init__(self, vertices, triangles, region_mapping, region_names=None,
                 triangle_normals=None, vertex_normals=None, triangle_areas=None, vertex_areas=None,
                 vertex_neighbors=None, basic=False):

        assert vertices.ndim == 2
        assert triangles.ndim == 2
        assert region_mapping.ndim == 1

        assert vertices.shape[1] == 3
        assert triangles.shape[1] == 3
        assert region_mapping.shape[0] == vertices.shape[0]

        self.vertices = vertices
        self.triangles = triangles
        self.region_mapping = region_mapping

        self.nverts = self.vertices.shape[0]
        self.ntriangs = self.triangles.shape[0]

        self.region_names = None
        if region_names is not None:
            self.set_region_names(region_names)

        self.triangle_normals = triangle_normals
        if (self.triangle_normals is None) and (not basic):
            self.triangle_normals = compute_triangle_normals(self.triangles, self.vertices)

        self.vertex_normals = vertex_normals
        if (self.vertex_normals is None) and (not basic):
            self.vertex_normals = compute_vertex_normals(self.vertices, self.triangles, self.triangle_normals)

        self.triangle_areas = triangle_areas
        if (self.triangle_areas is None) and (not basic):
            self.triangle_areas = compute_triangle_areas(self.vertices, self.triangles)

        self.vertex_areas = vertex_areas
        if (self.vertex_areas is None) and (not basic):
            self.vertex_areas = compute_vertex_areas(self.vertices, self.triangles, self.triangle_areas)

        self.area = 0.0 if (self.triangle_areas is None) else np.sum(self.triangle_areas)

        self.vertex_neighbors = vertex_neighbors
        if (self.vertex_neighbors is None) and (not basic):
            self.set_vertex_neighbors()


    @classmethod
    def from_file(cls, surface_file, basic=False):
        # Alias for backward compatibility
        return cls.from_zip_file(surface_file, basic)


    @classmethod
    def from_zip_file(cls, surface_file, basic=False):
        optional_data = {n: None for n in ['triangle_normals', 'vertex_normals', 'triangle_areas', 'vertex_areas']}

        with ZipFile(surface_file) as zf:
            with zf.open("vertices.txt", 'r') as fh:
                verts = np.genfromtxt(fh, dtype=float)
            with zf.open("triangles.txt", 'r') as fh:
                triangs = np.genfromtxt(fh, dtype=int)

            for name in optional_data.keys():
                filename = name + ".txt"
                if filename not in zf.namelist():
                    continue
                with zf.open(filename, 'r') as fh:
                    optional_data[name] = np.genfromtxt(fh, dtype=float)

        regmap = np.zeros(verts.shape[0], dtype=int)
        return cls(verts, triangs, regmap, basic=basic, **optional_data)


    @classmethod
    def from_npz_file(cls, surface_file, basic=False):
        optional_data = {n: None for n in ['triangle_normals', 'vertex_normals', 'triangle_areas', 'vertex_areas']}

        npz = np.load(surface_file)
        verts = npz['vertices']
        triangs = npz['triangles']
        for name in optional_data.keys():
            if name in npz:
                optional_data[name] = npz[name]
        regmap = np.zeros(verts.shape[0], dtype=int)
        return cls(verts, triangs, regmap, basic=basic, **optional_data)



    def save_surf_zip(self, surface_file):
        self.check_data()

        tmpdir = tempfile.mkdtemp()

        with ZipFile(surface_file, 'w') as zip_file:
            for name, data, fmt in [('vertices',         self.vertices,          '%.6f %.6f %.6f'),
                                    ('triangles',        self.triangles,         '%d %d %d'),
                                    ('triangle_normals', self.triangle_normals,  '%.6f %.6f %.6f'),
                                    ('vertex_normals',   self.vertex_normals,    '%.6f %.6f %.6f'),
                                    ('triangle_areas',   self.triangle_areas,    '%.6f'),
                                    ('vertex_areas',     self.vertex_areas,      '%.6f')]:
                if data is not None:
                    tmp_file = os.path.join(tmpdir, name + ".txt")
                    np.savetxt(tmp_file, data, fmt=fmt)
                    zip_file.write(tmp_file, os.path.basename(tmp_file))

        shutil.rmtree(tmpdir)


    def save_surf_npz(self, surface_file):
        self.check_data()
        data_dict = {k:v for k, v in [('vertices',         self.vertices),
                                      ('triangles',        self.triangles),
                                      ('triangle_normals', self.triangle_normals),
                                      ('vertex_normals',   self.vertex_normals),
                                      ('triangle_areas',   self.triangle_areas),
                                      ('vertex_areas',     self.vertex_areas)]
                     if v is not None}
        np.savez(surface_file, **data_dict)


    def set_vertex_neighbors(self):
        self.vert_neighbors = [set() for _ in range(self.vertices.shape[0])]
        for v1, v2, v3 in self.triangles:
            self.vert_neighbors[v1].update([v2, v3])
            self.vert_neighbors[v2].update([v1, v3])
            self.vert_neighbors[v3].update([v1, v2])


    def set_region_names(self, region_names):
        assert np.max(self.region_mapping) < len(region_names)
        self.region_names = region_names

    def check_data(self):
        assert not np.any(np.isnan(self.vertices))
        assert np.all(np.in1d(np.arange(self.nverts), self.triangles.flatten()))
        assert not np.any(np.isnan(self.vertex_normals))
        assert not np.any(np.isnan(self.vertex_areas))


    def compute_all(self, force=False):
        if force or (self.triangle_normals is None):
            self.triangle_normals = compute_triangle_normals(self.triangles, self.vertices)
        if force or (self.vertex_normals is None):
            self.vertex_normals = compute_vertex_normals(self.vertices, self.triangles, self.triangle_normals)
        if force or (self.triangle_areas is None):
            self.triangle_areas = compute_triangle_areas(self.vertices, self.triangles)
            self.area = np.sum(self.triangle_areas)
        if force or (self.vertex_areas is None):
            self.vertex_areas = compute_vertex_areas(self.vertices, self.triangles, self.triangle_areas)
        if force or (self.vertex_neighbors is None):
            self.set_vertex_neighbors()


def compute_triangle_normals(triangles, vertices):
    """Calculates triangle normals."""
    tri_u = vertices[triangles[:, 1], :] - vertices[triangles[:, 0], :]
    tri_v = vertices[triangles[:, 2], :] - vertices[triangles[:, 0], :]
    tri_norm = np.cross(tri_u, tri_v)

    try:
        triangle_normals = tri_norm / np.sqrt(np.sum(tri_norm ** 2, axis=1))[:, np.newaxis]
    except FloatingPointError:
        triangle_normals = tri_norm
    return triangle_normals



def compute_vertex_normals(vertices, triangles, triangle_normals):
    """
    Estimates vertex normals, based on triangle normals weighted by the
    angle they subtend at each vertex...
    """
    nverts = vertices.shape[0]
    ntriangs = triangles.shape[0]

    # Vertex triangles
    vertex_triangles = [[] for _ in range(nverts)]
    for k in range(triangles.shape[0]):
        vertex_triangles[triangles[k, 0]].append(k)
        vertex_triangles[triangles[k, 1]].append(k)
        vertex_triangles[triangles[k, 2]].append(k)

    # Triangle angles
    triangle_angles = np.zeros((ntriangs, 3))
    for tt in range(ntriangs):
        triangle = triangles[tt, :]
        for ta in range(3):
            ang = np.roll(triangle, -ta)
            triangle_angles[tt, ta] = np.arccos(np.dot(
                (vertices[ang[1], :] - vertices[ang[0], :]) /
                np.sqrt(np.sum((vertices[ang[1], :] - vertices[ang[0], :]) ** 2, axis=0)),
                (vertices[ang[2], :] - vertices[ang[0], :]) /
                np.sqrt(np.sum((vertices[ang[2], :] - vertices[ang[0], :]) ** 2, axis=0))))

    # Vertex normals
    vert_norms = np.zeros((nverts, 3))
    bad_normal_count = 0
    for k in range(nverts):
        try:
            tri_list = list(vertex_triangles[k])
            angle_mask = triangles[tri_list, :] == k
            angles = triangle_angles[tri_list, :]
            angles = angles[angle_mask][:, np.newaxis]
            angle_scaling = angles / np.sum(angles, axis=0)
            vert_norms[k, :] = np.mean(angle_scaling * triangle_normals[tri_list, :], axis=0)
            # Scale by angle subtended.
            vert_norms[k, :] = vert_norms[k, :] / np.sqrt(np.sum(vert_norms[k, :] ** 2, axis=0))
            # Normalise to unit vectors.
        except (ValueError, FloatingPointError) as e:
            # If normals are bad, default to position vector
            # A nicer solution would be to detect degenerate triangles and ignore their
            # contribution to the vertex normal
            vert_norms[k, :] = vertices[k] / np.sqrt(vertices[k].dot(vertices[k]))
            bad_normal_count += 1
    if bad_normal_count:
        print(" %d vertices have bad normals" % bad_normal_count)

    return vert_norms


def compute_triangle_areas(vertices, triangles):
    """Calculates the area of triangles making up a surface."""
    tri_u = vertices[triangles[:, 1], :] - vertices[triangles[:, 0], :]
    tri_v = vertices[triangles[:, 2], :] - vertices[triangles[:, 0], :]
    tri_norm = np.cross(tri_u, tri_v)
    triangle_areas = np.sqrt(np.sum(tri_norm ** 2, axis=1)) / 2.0
    triangle_areas = triangle_areas[:, np.newaxis]
    return triangle_areas


def compute_vertex_areas(vertices, triangles, triangle_areas=None):
    if triangle_areas is None:
        triangle_areas = compute_triangle_areas(vertices, triangles)

    vertex_areas = np.zeros((vertices.shape[0]))
    for triang, vertices in enumerate(triangles):
        for i in range(3):
            vertex_areas[vertices[i]] += 1./3. * triangle_areas[triang]

    return vertex_areas
