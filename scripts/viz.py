"""
Script for visualizing subject's cortical surface and the implanted electrodes.
"""


import vtkplotter as vp

from util.surface import Surface
from util.contacts import Contacts

subject = "id001"
surf = Surface.from_npz_file(f"data/Geometry/{subject}/surface.npz")
contacts = Contacts(f"data/Geometry/{subject}/seeg.txt")

mesh = vp.Mesh([surf.vertices, surf.triangles], alpha=0.1)
points = vp.Points(contacts.xyz, r=20, c=(1,1,0))

vplotter = vp.Plotter(axes=0)
vplotter.show([mesh, points])
