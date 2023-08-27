# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:33:29 2023

@author: mchamaillard
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer la surface 3D
surf = ax.plot_surface(delta, cmap='viridis')

# Ajouter une barre de couleur
fig.colorbar(surf)

# Ajouter des étiquettes pour les axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
