import os
import sys
import readline
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

plt.style.use('estilo_latex.mplstyle')
# 'Annulus' viene con matplotlib 3.5+ ('pip3 install -U matplotlib' para actualizar)
# https://svgutils.readthedocs.io/en/latest/tutorials/publication_quality_figures.html

def unbounded(x):
    """La sucesión de ángulos 'x' en [-pi, pi] pasa a ser una
    sucesión 'y' de ángulos no acotados que va 'acumulando' las vueltas.
    """
    y = np.copy(x)
    for i in range(1, len(y)):
        dy = y[i] - y[i-1]
        if dy < -3:
            y[i:] = y[i:] + 2*np.pi
        elif dy > 3:
            y[i:] = y[i:] - 2*np.pi
    return y

def german(csv_dir):
    '''A partir de los dos CSVs de data y metadatos alojados en el
    directorio 'csv_dir', calcula y guarda en un diccionario todas
    las magnitudes necesarias.
    '''
    file = os.path.basename(csv_dir)
    data_path = os.path.join(csv_dir, f'{file}_data.csv')
    meta_path = os.path.join(csv_dir, f'{file}_meta.csv')
    data = pd.read_csv(data_path)
    meta = pd.read_csv(meta_path)
    
    TIME = data['TIME'].to_numpy()
    # Posiciones (en px) de los ojalillos
    X_L, Y_L = data['X_L'].to_numpy(), data['Y_L'].to_numpy()
    X_R, Y_R = data['X_R'].to_numpy(), data['Y_R'].to_numpy()
    # Posición (en px) del centro de la arena
    x_a, y_a = meta['x_a'][0], meta['y_a'][0]
    # Radio (en px y en mm) de la arena
    r_a, r = meta['r_a'][0], meta['r'][0]
    # Factor de escala (px -> mm)
    px2mm = meta['px2mm'][0]

    # A partir de ahora, el centro de la arena pasa a ser el origen de
    # coordenadas, invierto el sentido del eje 'y' y expreso todo en mm

    # Vectores posición L y R de los ojalillos
    L_x, L_y = (X_L-x_a) * px2mm, -(Y_L-y_a) * px2mm
    R_x, R_y = (X_R-x_a) * px2mm, -(Y_R-y_a) * px2mm
    # Vector D de distancia entre ojalillos (L->R)
    D_x, D_y = R_x - L_x, R_y - L_y
    # Vector N de orientación (= vector D rotado 90° CCW)
    N_x, N_y = -D_y, D_x
    # Coordenadas cartesianas del centro de masa
    X, Y = L_x + 0.5*D_x, L_y + 0.5*D_y
    # Coordenadas polares del centro de masa
    R, PHI = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
    # Ángulo ALPHA de orientación en [-pi, pi]
    ALPHA = np.arctan2(N_y, N_x)
    # Ángulo ALPHA_UN de orientación no acotado que 'acumula' las vueltas
    ALPHA_UN = unbounded(ALPHA)

    # Versores radial R_HAT y angular PHI_HAT
    R_HAT_x, R_HAT_y = X / R, Y / R
    PHI_HAT_x, PHI_HAT_y = -R_HAT_y, R_HAT_x

    todo = {
        'TIME': TIME, 'X': X, 'Y': Y, 'R': R, 'PHI': PHI,
        'ALPHA': ALPHA, 'ALPHA_UN': ALPHA_UN,
        'r_a': r_a, 'r': r, 'px2mm': px2mm,
        'R_HAT_x': R_HAT_x, 'R_HAT_y': R_HAT_y,
        'PHI_HAT_x': PHI_HAT_x, 'PHI_HAT_y': PHI_HAT_y,
        'left': meta['left'][0], 'right': meta['right'][0],
        'step': meta['step'][0], 'stop': meta['stop'][0],
        'real': meta['real'][0], 'file': file,
        }
    return todo

def graficador_arena(csv_dir):
    todo = german(csv_dir)
    t, x, y = todo['TIME'], todo['X'], todo['Y']
    R_a = todo['r_a'] * todo['px2mm']

    fig, ax = plt.subplots(figsize=(7, 6))
    # ---------- Trayectoria del Kilobot ----------
    ax.scatter(x, y, c=range(len(x)), cmap='autumn', s=1, zorder=5)
    # ---------- Arena y posición inicial del Kilobot ----------
    ax.add_patch(mpl.patches.Annulus(xy=(0, 0), r=R_a+20, width=20, fc='#d19556'))
    ax.add_patch(mpl.patches.Annulus(xy=(0, 0), r=R_a, width=5, fc='k'))
    ax.add_patch(mpl.patches.Annulus(xy=(x[0], y[0]), r=16.5, width=5, fc='green'))
    # ---------- Escala ----------
    x0, y0, ruler = 160, -160, 50
    ax.plot([x0-ruler, x0], [y0, y0], c='k', lw=3)
    ax.text(x0-(ruler/2), y0+5, fr'\SI{{{ruler}}}{{\mm}}', ha='center')
    # ---------- Colorbar temporal ----------
    cmap = mpl.cm.autumn
    norm = mpl.colors.Normalize(vmin=t[0], vmax=t[-1]/60)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                 pad=0.04, fraction=0.046, label=r'tiempo [\si{\minute}]')
    # ---------- Configuración para una correcta representación ----------
    lim = todo['r'] + 20
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

def evolucion_temporal_orientacion(*csv_dir):
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for i in csv_dir:
        todo = german(i)
        file = todo['file']
        t, un = todo['TIME'], todo['ALPHA_UN']
        ax.plot(t, un, 'o', ms=2, label=file)
    ax.set_xlabel(r'tiempo $t$ [\si{\s}]')
    ax.set_ylabel(r'orientación $\alpha(t)$ [\si{\radian}]')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()

def histograma(*csv_dir):
    fig, ax = plt.subplots()
    for i in csv_dir:
        todo = german(i)
        r = todo['R']
        r_avg = np.mean(r)
        file = todo['file']
        my_bins = np.arange(0, 151, step=5)
        ax.hist(r, bins=my_bins, alpha=0.7, edgecolor='k',
                density=True, label=f'{file} {r_avg:.2f}')
    ax.set_xlabel(r'$r$ [\si{\mm}]')
    ax.set_ylabel(r'$P(r)$')
    ax.legend(title=r'distancia radial $r$')
    plt.show()

def main():
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    FILE = ['2000_a', '2000_b', '2000_c']
    csv_dir = [os.path.join(repo_dir, 'Data', file) for file in FILE]
    #evolucion_temporal(*csv_dir)
    histograma(*csv_dir)

if __name__ == '__main__':
    main()