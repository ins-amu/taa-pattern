"""
Various tools related to the triangulated surfaces
"""

import numpy as np
import itertools
from .surface import Surface

def get_component_mask(init, conn_list):
    """
    Return an array with the mask of all nodes connected to the `init` node.

    >>> list(get_component_mask(0, [[1], [0], [3, 4], [2, 4], [2, 3]]))
    [True, True, False, False, False]

    >>> list(get_component_mask(4, [[1], [0], [3, 4], [2, 4], [2, 3]]))
    [False, False, True, True, True]
    """

    nverts = len(conn_list)
    mask = np.zeros(nverts, dtype=bool)

    stack = [init]
    while stack:
        node = stack.pop()
        if mask[node]:
            continue
        mask[node] = True
        stack.extend(conn_list[node])

    return mask

def create_conn_list(nverts, triangles):
    """
    >>> create_conn_list(4, [[0, 1, 2], [0, 2, 3]])
    [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]
    """
    conn_list = [[] for _ in range(nverts)]
    for triangle in triangles:
        for i, j in itertools.product(range(3), range(3)):
            if i != j:
                conn_list[triangle[i]].append(triangle[j])
                conn_list[triangle[j]].append(triangle[i])

    for i in range(nverts):
        conn_list[i] = list(np.unique(conn_list[i]))

    return conn_list


def components(nverts, triangles):
    """
    Return an array with the connected component indices, given number of vertices and a list of triangles.

    >>> list(components(4, [[1, 2, 3]]))
    [1, 2, 2, 2]
    """

    conn_list = create_conn_list(nverts, triangles)
    comps = np.zeros(nverts, dtype=int)
    ncomps = 0

    for i in range(nverts):
        if comps[i] == 0:
            ncomps += 1
            mask = get_component_mask(i, conn_list)
            comps[mask] = ncomps

    return comps


def refine(surf, level=1):
    """
    Return once refined surface by adding new nodes to the middle of the triangle faces.

    0
    | \
    |  \
    3   5
    |    \
    |     \
    1--4---2

    >>> surf = Surface(np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]]), np.array([[0, 1, 2]], dtype=int),\
                       np.array([0, 0, 0], dtype=int))
    >>> refsurf = refine(surf)
    >>> len(refsurf.vertices), len(refsurf.triangles)
    (6, 4)
    """
    if level == 0:
        return surf

    added_verts = [dict() for _ in range(surf.nverts)] # old indices of existing vertices
                                                       # to new index of the added vertex
    old_to_new_verts = -1 * np.ones(surf.nverts, dtype=int)

    new_vertices = []
    new_triangles = []
    new_region_mapping = []

    n_new_verts = 0
    for v0, v1, v2 in surf.triangles:
        triangle_verts = []

        # Copy old vertices
        for vert in [v0, v1, v2]:
            if old_to_new_verts[vert] == -1:
                new_vertices.append(surf.vertices[vert, :])
                old_to_new_verts[vert] = n_new_verts
                new_region_mapping.append(surf.region_mapping[vert])
                n_new_verts += 1

            new_ind = old_to_new_verts[vert]
            triangle_verts.append(new_ind)

        # Create new vertices
        for va, vb in [[v0, v1], [v1, v2], [v2, v0]]:
            va, vb = min(va, vb), max(va, vb)
            if vb not in added_verts[va]:
                new_vertices.append(np.mean(surf.vertices[[va, vb], :], axis=0))
                new_region_mapping.append(surf.region_mapping[va]) # Same as of vb, if both are equal.
                                                                   # Otherwise, we use the lower with no justification.
                                                                   # This may (?) cause come problems on the interfaces.
                added_verts[va][vb] = n_new_verts
                n_new_verts += 1
            triangle_verts.append(added_verts[va][vb])

        # Now triangle_verts should contain 6 vertices, indexed as in the sketch in docstring
        assert(len(triangle_verts) == 6)

        # Add all triangles
        for inds in [[0, 3, 5], [3, 1, 4], [3, 4, 5], [5, 4, 2]]:
            new_triangles.append([triangle_verts[inds[i]] for i in range(3)])

    assert(np.all(old_to_new_verts >= 0))

    refined_surface = Surface(np.array(new_vertices),
                              np.array(new_triangles, dtype=int),
                              np.array(new_region_mapping, dtype=int),
                              surf.region_names, basic=True)

    if level == 1:
        return refined_surface
    elif level >= 2:
        return refine(refined_surface, level=level - 1)
