# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:33:19 2026

@author: Admin
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, eye

def lire_image_gris(chemin, taille=None):
    """
    Lit une image, la convertit en niveaux de gris,
    la redimensionne et normalise les valeurs entre 0 et 1.
    """
    img = Image.open(chemin).convert("L")
    if taille is not None:
        img = img.resize(taille)

    f = np.array(img, dtype=np.float64) / 255.0

    return f


# %%

f = lire_image_gris("lena.png",taille=(256,256))
nx, ny = f.shape                 # Taille de l'image
hx, hy = 1.0 / nx, 1.0 / ny      # Discrétisation de l'espace
dt = hx**2/8
nb_iter =100
c1 = 0.8
c2 = 0.2
epsilon = hx/2
eta = 2e-2



#%%
def laplacien_implicite(Nx, Ny):
    """
    Construit la matrice creuse du Laplacien 2D avec conditions de Neumann homogènes.
    """

    hx = 1 / Nx
    hy = 1 / Ny

    N = Nx * Ny
    A = lil_matrix((N, N))

    def indice(i, j):
        return Ny * i + j

    # =========================
    # 1. Points intérieurs
    # =========================
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            k = indice(i, j)

            A[k, indice(i, j)]     = -2 / hx**2 - 2 / hy**2
            A[k, indice(i+1, j)]   =  1 / hx**2
            A[k, indice(i-1, j)]   =  1 / hx**2
            A[k, indice(i, j+1)]   =  1 / hy**2
            A[k, indice(i, j-1)]   =  1 / hy**2

    # =========================
    # 2. Bords sans les coins
    # =========================

    # Bord supérieur : i = 0
    for j in range(1, Ny - 1):
        k = indice(0, j)

        A[k, indice(0, j)]   =  1 / hx**2 - 2 / hy**2
        A[k, indice(1, j)]   = -2 / hx**2
        A[k, indice(2, j)]   =  1 / hx**2
        A[k, indice(0, j+1)] =  1 / hy**2
        A[k, indice(0, j-1)] =  1 / hy**2

    # Bord inférieur : i = Nx-1
    for j in range(1, Ny - 1):
        k = indice(Nx - 1, j)

        A[k, indice(Nx - 3, j)]   =  1 / hx**2
        A[k, indice(Nx - 2, j)]   = -2 / hx**2
        A[k, indice(Nx - 1, j)]   =  1 / hx**2 - 2 / hy**2
        A[k, indice(Nx - 1, j+1)] =  1 / hy**2
        A[k, indice(Nx - 1, j-1)] =  1 / hy**2

    # Bord gauche : j = 0
    for i in range(1, Nx - 1):
        k = indice(i, 0)

        A[k, indice(i+1, 0)] =  1 / hx**2
        A[k, indice(i-1, 0)] =  1 / hx**2
        A[k, indice(i, 0)]   = -2 / hx**2 + 1 / hy**2
        A[k, indice(i, 1)]   = -2 / hy**2
        A[k, indice(i, 2)]   =  1 / hy**2

    # Bord droit : j = Ny-1
    for i in range(1, Nx - 1):
        k = indice(i, Ny - 1)

        A[k, indice(i+1, Ny - 1)] =  1 / hx**2
        A[k, indice(i-1, Ny - 1)] =  1 / hx**2
        A[k, indice(i, Ny - 3)]   =  1 / hy**2
        A[k, indice(i, Ny - 2)]   = -2 / hy**2
        A[k, indice(i, Ny - 1)]   = -2 / hx**2 + 1 / hy**2

    # =========================
    # 3. Les 4 coins
    # =========================

    # Coin supérieur gauche
    k = indice(0, 0)
    A[k, indice(0, 0)] =  1 / hx**2 + 1 / hy**2
    A[k, indice(1, 0)] = -2 / hx**2
    A[k, indice(2, 0)] =  1 / hx**2
    A[k, indice(0, 1)] = -2 / hy**2
    A[k, indice(0, 2)] =  1 / hy**2

    # Coin supérieur droit
    k = indice(0, Ny - 1)
    A[k, indice(0, Ny - 1)] =  1 / hx**2 + 1 / hy**2
    A[k, indice(1, Ny - 1)] = -2 / hx**2
    A[k, indice(2, Ny - 1)] =  1 / hx**2
    A[k, indice(0, Ny - 3)] =  1 / hy**2
    A[k, indice(0, Ny - 2)] = -2 / hy**2

    # Coin inférieur gauche
    k = indice(Nx - 1, 0)
    A[k, indice(Nx - 3, 0)] =  1 / hx**2
    A[k, indice(Nx - 2, 0)] = -2 / hx**2
    A[k, indice(Nx - 1, 0)] =  1 / hx**2 + 1 / hy**2
    A[k, indice(Nx - 1, 1)] = -2 / hy**2
    A[k, indice(Nx - 1, 2)] =  1 / hy**2

    # Coin inférieur droit
    k = indice(Nx - 1, Ny - 1)
    A[k, indice(Nx - 3, Ny - 1)] =  1 / hx**2
    A[k, indice(Nx - 2, Ny - 1)] = -2 / hx**2
    A[k, indice(Nx - 1, Ny - 1)] =  1 / hx**2 + 1 / hy**2
    A[k, indice(Nx - 1, Ny - 3)] =  1 / hy**2
    A[k, indice(Nx - 1, Ny - 2)] = -2 / hy**2

    return A.tocsr()


# %%

from scipy.sparse.linalg import factorized

def segmentation_euler_implicite(f, dt, nb_iter, c1, c2, epsilon, eta,afficher = True):
    f_original = f.copy().astype(float)
    f_original = f_original >= 0.5
    u = (f_original >= 0.5).astype(float)

    Nx, Ny = u.shape
    N = Nx * Ny

    A = laplacien_implicite(Nx, Ny)
    I = eye(N, format="csr")

    M_imp = I - dt * A
    
    if afficher:
        plt.ion()  # mode interactif
    
        fig, ax = plt.subplots(figsize=(6, 6))
        image = ax.imshow(u, cmap="jet", vmin=0, vmax=1)
        ax.set_title("Itération 0")
        ax.axis("off")
        plt.show()

    # Factorisation une seule fois
    solve_M = factorized(M_imp.tocsc())

    for n in range(nb_iter):

        u_vec = u.reshape(N)

        h_u = ((c1 - c2) / (epsilon * eta)) * (
            f_original - (c1 * u + c2 * (1 - u))
        )

        W_prime = u * (1 - u) * (1 - 2*u)

        b = (-1 / epsilon**2) * W_prime + h_u*np.sqrt(u**2*(1-u)**2)*6
        b_vec = b.reshape(N)

        # Résolution rapide grâce à la factorisation
        u_vec_new = solve_M(u_vec + dt * b_vec)

        u = u_vec_new.reshape(Nx, Ny)

        print(n)
        
        if afficher:
            image.set_data(u)
            ax.set_title(f"Itération {n+1}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            #plt.pause(0.05)
    if afficher:
        plt.ioff()
        plt.show()

    return u

u = segmentation_euler_implicite(
    f=f,

    nb_iter=1000,
    c1=0.8,
    c2=0.2,
    epsilon=hx/2,
    dt = epsilon**2,
    eta=0.05/8
)
#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Image originale
ax[0].imshow(f, cmap="gray", vmin=0, vmax=1)
ax[0].set_title("Image originale")
ax[0].axis("off")

# Image après évolution
ax[1].imshow(u, cmap="gray", vmin=0, vmax=1)
ax[1].set_title("Image après évolution de u")
ax[1].axis("off")

plt.tight_layout()
plt.show()
    
    
    