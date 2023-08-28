# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:00:49 2023

@author: mchamaillard
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dill
u_L = cv2.imread('u_L.jpg',cv2.IMREAD_GRAYSCALE)
u_R = cv2.imread('u_R.jpg',cv2.IMREAD_GRAYSCALE)


# Nous appliquons un seuillage.
u_L=(u_L<50)*255
u_R=(u_R<50)*255


u_L=u_L*1.
u_R=u_R*1.
#u_L=np.roll(u_R,axis=1,shift=100)






plt.subplot(1,2,1)
plt.imshow(u_L, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(u_R, cmap='gray')

taille_fenetre=5
delta_min=50
pas_decimation=1

NX=np.shape(u_L)[1]
NY=np.shape(u_L)[0]

def Pi_produit(u,v):
    les_correlation=np.array([np.correlate(u[kk], v[kk],mode='valid') for kk in range(taille_fenetre)])
    return np.sum(les_correlation,axis=0 )





delta=np.zeros((NY-taille_fenetre,NX-taille_fenetre))
delta=np.full(np.shape(delta), np.nan)

I_recherche=range(0,NY-taille_fenetre,pas_decimation)
J_recherche=range(0,NX-taille_fenetre,pas_decimation)



for ii in I_recherche:
    print(ii/NY)
    
    
    bande_gauche=u_L[ii:(ii+taille_fenetre),:]        
    masque_de_1=np.ones((taille_fenetre,taille_fenetre))
    bande_gauche_carre= np.array([[x ** 2 for x in row] for row in bande_gauche])
    
    for jj in J_recherche:

        
        petite_image_droite=u_R[ii:(ii+taille_fenetre),jj:(jj+taille_fenetre)]
        if(u_R[ii,jj]>10):   
            
            bg=bande_gauche
            bd=petite_image_droite
            II=ii
            JJ=jj
            correlation_croise=Pi_produit(bande_gauche, petite_image_droite)/np.sqrt(Pi_produit(bande_gauche_carre, masque_de_1))/np.linalg.norm(petite_image_droite)
            
            correlation_croise[np.isnan(correlation_croise)] = -1
            delta[ii,jj]=np.argmax(correlation_croise[jj+delta_min:])+delta_min



U, V = np.meshgrid(range(np.shape(delta)[1]),range(np.shape(delta)[0]))




def plot_surface(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='gray',linewidth=0, antialiased=False)
    fig.colorbar(surf)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    plt.show()


U_decim=U[::pas_decimation,::pas_decimation]
V_decim=V[::pas_decimation,::pas_decimation]
delta_decim=delta[::pas_decimation,::pas_decimation]


uu=u_R[0:(NY-taille_fenetre),0:(NX-taille_fenetre)]

plt.figure()

plt.imshow(delta[::pas_decimation,::pas_decimation])

