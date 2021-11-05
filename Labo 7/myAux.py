import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('../estilo_latex.mplstyle')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.cbook as cbook
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from scipy.optimize import curve_fit
import cv2

def shift(x):
    """Paso de tener un ángulo en [-pi, pi]
    a un ángulo no acotado que va acumulando las vueltas.
    """
    y = np.copy(x)
    for i in range(1, len(y)):
        dy = y[i] - y[i-1]
        if dy < -3:
            y[i:] = y[i:] + 2*np.pi
        elif dy > 3:
            y[i:] = y[i:] - 2*np.pi
    return y


def lineal(x, m):
    return m*x


def german(video, carpeta):
    # Cargo los .csv que tienen toda la información necesaria
    data_csv = home + 'Data/' + carpeta + video + '/' + video + '_data.csv'
    param_csv = home + 'Data/' + carpeta + video + '/' + video + '_param.csv'
    data = np.genfromtxt(data_csv, delimiter=',', names=True)
    param = np.genfromtxt(param_csv, delimiter=',', names=True, dtype=int)
    # Posiciones (matriciales) de los ojalillos y del centro de la arena
    I_LEFT, J_LEFT = data['I_LEFT'], data['J_LEFT']
    I_RIGHT, J_RIGHT = data['I_RIGHT'], data['J_RIGHT']
    i_arena, j_arena = data['I_ARENA'][0], data['J_ARENA'][0]
    # Coordenadas (cartesianas) de los ojalillos
    X_LEFT, Y_LEFT = J_LEFT - j_arena, -(I_LEFT - i_arena)
    X_RIGHT, Y_RIGHT = J_RIGHT - j_arena, -(I_RIGHT - i_arena)
    # Vector LR que va del ojalillo Left al Right
    LR_X, LR_Y = X_RIGHT - X_LEFT, Y_RIGHT - Y_LEFT
    # Vector N de orientación (Luego de rotar 90° CCW al vector LR)
    N_X, N_Y = -LR_Y, LR_X
    # Instantes de tiempo con tiempo inicial cero
    T = data['TIME'] - data['TIME'][0]
    # Coordenadas (cartesianas) del punto medio
    X, Y = X_LEFT + 0.5*LR_X, Y_LEFT + 0.5*LR_Y
    # Coordenadas (polares) del punto medio
    R, PHI = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
    # Ángulo ALPHA de orientación en [-pi, pi]
    ALPHA = np.arctan2(N_Y, N_X)
    # Ángulo ALPHA_UN de orientación no acotado que 'acumula' las vueltas
    ALPHA_UN = shift(ALPHA)
    # Radio de la arena (en píxeles y en milímetros)
    r_arena, radio = data['R_ARENA'][0], param['radio']
    # Cuántos píxeles representan un milímetro
    mm = r_arena / radio
    # Las coordenadas espaciales en milímetros
    X_MM, Y_MM = X / mm, Y / mm
    R_MM = R / mm
    # Versores radial RV y angular PHIV
    RV_X, RV_Y = X / R, Y / R
    PHIV_X, PHIV_Y = -RV_Y, RV_X
    # Todo lo que voy a necesitar...
    myDict = {
        'T': T,
        'X': X, 'Y': Y,
        'R': R, 'PHI': PHI,
        'ALPHA': ALPHA, 'ALPHA_UN': ALPHA_UN,
        'r_arena': r_arena, 'radio': radio,
        'mm': mm,
        'X_MM': X_MM, 'Y_MM': Y_MM,
        'R_MM': R_MM,
        'RV_X': RV_X, 'RV_Y': RV_Y,
        'PHIV_X': PHIV_X, 'PHIV_Y': PHIV_Y,
        'step': param['step'],
        'left': param['left'], 'right': param['right'],
    }
    return myDict


def velocidad_angular(video, carpeta, dt=1):
    todo = german(video, carpeta)
    t, a = todo['T'], todo['ALPHA_UN']
    ventana = t[t <= dt] # Ventana inicial
    L = len(ventana) - 1
    t0 = t[t <= t[-1]-dt] # Tiempos iniciales de las ventanas
    N = len(t0) # Cuántas ventanas
    w = np.zeros(N) # Velocidades angulares
    for i in range(N):
        da = a[i+L] - a[i]
        dt = t[i+L] - t[i]
        w[i] = da / dt
    return w


def velocidad_sinstop(video, carpeta):
    todo = german(video, carpeta)
    t, x, y = todo['T'], todo['X_MM'], todo['Y_MM']
    r_x, r_y = todo['RV_X'], todo['RV_Y'] # Versor r
    phi_x, phi_y = todo['PHIV_X'], todo['PHIV_Y'] # Versor phi
    stop = 500*(1e-3)
    paso = todo['step']*(1e-3)
    ciclos = 1
    dt = ciclos*(paso+stop)
    ventana = t[t <= dt]
    L = len(ventana) - 1
    paso_t = t[t <= stop]
    M = len(paso_t) - 1
    teff = t[t <= t[-1]-dt] # Tiempos permitidos
    t0 = teff[::M]
    N = len(t0)
    v_x, v_y = np.zeros(N), np.zeros(N) # Vector velocidad
    v_r, v_phi, v = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        j = np.where(t == t0[i])[0][0]
        dx = x[j+L] - x[j]
        dy = y[j+L] - y[j]
        v_x[i] = dx / (ciclos*paso)
        v_y[i] = dy / (ciclos*paso)
        v_r[i] = v_x[i]*r_x[j] + v_y[i]*r_y[j]
        v_phi[i] = v_x[i]*phi_x[j] + v_y[i]*phi_y[j]
        v[i] = np.sqrt(v_r[i]**2 + v_phi[i]**2)
        #v[i] = np.sqrt(v_x[i]**2 + v_y[i]**2)
    return t0, v_r, v_phi, v


def velocidad(video, carpeta, dT=0.5):
    todo = german(video, carpeta)
    t, x, y = todo['T'], todo['X_MM'], todo['Y_MM']
    ventana = t[t <= dT] # Ventana inicial
    L = len(ventana) - 1
    t0 = t[t <= t[-1]-dT] # Tiempos iniciales de las ventanas
    N = len(t0) # Cuántas ventanas
    v_x, v_y = np.zeros(N), np.zeros(N) # Vector velocidad
    alpha = todo['ALPHA'][:N] # Ángulos de orientación
    beta = np.zeros(N) # Ángulos de dirección
    for i in range(N):
        dx = x[i+L] - x[i]
        dy = y[i+L] - y[i]
        dt = t[i+L] - t[i]
        v_x[i] = dx / dt
        v_y[i] = dy / dt
        beta[i] = np.arctan2(v_y[i], v_x[i])
    r_x, r_y = todo['RV_X'][:N], todo['RV_Y'][:N] # Versor r
    phi_x, phi_y = todo['PHIV_X'][:N], todo['PHIV_Y'][:N] # Versor phi
    v_r = v_x*r_x + v_y*r_y # Proyecto la velocidad sobre el versor r
    v_phi = v_x*phi_x + v_y*phi_y # Proyecto la velocidad sobre el versor phi
    v = np.sqrt(v_r**2 + v_phi**2) # Módulo de la velocidad
    return t0, v_r, v_phi, v, alpha, beta


def DCM(t, x, y, T=60*5):
    tdcm = t[t <= T]
    dcm = np.zeros(len(tdcm))
    teff = t[t <= t[-1]-T]
    for i in range(len(teff)):
        dc = (x[i:i+len(tdcm)]-x[i])**2 + (y[i:i+len(tdcm)]-y[i])**2
        dcm = dcm + dc/len(teff)
    return tdcm, dcm


def DCMA(t, x, T=60*10):
    tdcm = t[t <= T]
    dcm = np.zeros(len(tdcm))
    teff = t[t <= t[-1]-T]
    for i in range(len(teff)):
        dc = (x[i:i+len(tdcm)]-x[i])**2
        dcm = dcm + dc/len(teff)
    return tdcm, dcm


def circles(video, i, th=13):
    img_jpg = home + 'Data/' + video + '/frames/' + 'frame_' + str(i) + '.jpg'
    img = cv2.imread(img_jpg, 0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, 
                               param2=th, minRadius=5, maxRadius=7)
    print('Los círculos detectados son:')
    print(circles)
    circles = np.uint16(np.around(circles))
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        cv2.circle(cimg, (i[0],i[1]), i[2], (0,255,0), 2)
        cv2.circle(cimg, (i[0],i[1]), 2, (0,0,255), 3)
    cv2.imshow(video, cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def graficador(video, carpeta):
    todo = german(video, carpeta)
    t, x, y = todo['T'], todo['X'], todo['Y']
    r = todo['r_arena']
    mm = todo['mm']
    # Grafico
    fig, ax = plt.subplots(figsize=(1.3, 6))
    # Arenas
    arena = plt.Circle((0, 0), r, fill=False, lw=0.8, color='lime')
    madera = plt.Circle((0, 0), r+25, fill=False, lw=24, color='#d19556')
    cinta = plt.Circle((0, 0), r-3, fill=False, lw=4, color='k')
    #ax.add_patch(arena)
    #ax.add_patch(madera)
    #ax.add_patch(cinta)
    #ax.add_patch(arena)
    # Trayectoria
    #ax.scatter(x, y, c=range(len(x)), cmap='autumn', s=2)
    # Posición inicial (bicho)
    #ax.plot([x[0], x[0]], [y[0]-16*mm, y[0]+16*mm], c='lime', zorder=5)
    #path_kb = cbook.get_sample_data(home + 'kb.png')
    #img_kb = OffsetImage(plt.imread(path_kb), zoom=0.5)
    #bicho = AnnotationBbox(img_kb, (x[0], y[0]), frameon=False)
    #ax.add_artist(bicho)
    # Escala
    x0, y0 = 400, -400
    rule = 50
    #ax.plot([x0-rule*mm, x0], [y0, y0], c='k') # ----
    #ax.plot([x0-rule*mm, x0-rule*mm], [y0-8, y0+8], c='k') # |-
    #ax.plot([x0, x0], [y0-8, y0+8], c='k') # -|
    #ax.text(x0-0.5*rule*mm, y0+10, r'\SI{%d}{\mm}' % rule, size=FS-1, ha='center')
    # Colorbar temporal
    cmap = mpl.cm.autumn
    norm = mpl.colors.Normalize(vmin=t[0], vmax=t[-1]/60)
    #fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #             pad=0.04, fraction=0.046, label=r'tiempo [\si{\minute}]')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax,
                 pad=0.04, fraction=0.046, label=r'tiempo [\si{\minute}]')
    #ax.set_xlim(-415, 415)
    #ax.set_ylim(-415, 415)
    #ax.set_aspect('equal')
    #ax.axis('off')
    #ax.text(0.9, 0.9, r'\textbf{(c)}', size=FS+3, ha='center', transform=ax.transAxes)
    plt.show()


def temporal(lista, carpeta, VAR='ALPHA'):
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    
    axins = ax.inset_axes([0.07, 0.08, 0.27, 0.38])
    mark_inset(ax, axins, loc1=1, loc2=3, ec='0.5', zorder=2.1)
    axins.tick_params(labelsize=21)
    axins.set_xlim(-5, 70)
    axins.set_ylim(-10, 5)
    
    #labels = [r'potencia $%d$ CCW' % 64, r'potencia $%d$ CW' % 74]
    #colores = ['red', 'blue']
    for i, video in enumerate(lista):
        todo = german(video, carpeta)
        t, y = todo['T'], todo[VAR]
        ax.plot(t, y, label=r'\SI{%d}{\ms}' % todo['step'])
        #line, = ax.plot(t, y, c='C2')
        axins.plot(t, y)
        ax.set_xlabel(r'tiempo~$t$ [\si{\s}]')
        ax.set_ylabel(r'orientación~$\alpha(t)$ [\si{\radian}]')
    ax.legend(title=r'paso temporal', loc='upper left', bbox_to_anchor=(0.45, 0.62))
    #ax.legend([line], [r'\SI{3000}{\ms}'], title=r'paso temporal')
    plt.show()


def temporal2(lista, carpeta, VAR='ALPHA'):
    fig, ax = plt.subplots()
    labels = [r'potencia $%d$ CCW' % 64, r'potencia $%d$ CW' % 78]
    colores = ['red', 'blue']
    lines = []
    for i, video in enumerate(lista):
        todo = german(video, carpeta)
        t, y = todo['T'], todo[VAR]
        rng = np.logical_and(t<=60, t>=38)
        line = ax.scatter(t[rng], y[rng], c=colores[i], s=2, zorder=2)
        lines.append(line)
        ax.set_xlabel(r'tiempo~$t$ [\si{\s}]')
        ax.set_ylabel(r'orientación~$\alpha(t)$ [\si{\radian}]')
    ax.plot(55, 0.79, 'o', ms=40, mec='r', mfc='none', mew=3.5)
    ax.legend(lines, labels, markerscale=4, handletextpad=-0.3)
    plt.show()


def histograma(lista, carpeta, VAR):
    fig, ax = plt.subplots()
    for i, video in enumerate(lista):
        todo = german(video, carpeta)
        #t0, v_r, v_phi, v, alpha, beta = velocidad(video, carpeta)
        t0, v_r, v_phi, v = velocidad_sinstop(video, carpeta)
        step = todo['step']
        alpha = 0.7
        if VAR == 'R_MM':
            y = todo[VAR]
            y_avg = np.mean(y)
            my_bins = np.arange(0, 151, step=6) # Bins de 'step' mm
            color = 'C0'
            media = r'$r_\mathrm{avg} = \SI{%d}{\mm}$' % (y_avg)
            ax.text(0.1, 0.9, media, size=FS, ha='left', transform=ax.transAxes)
            ax.set_xlabel(r'$r$ [\si{\mm}]')
            ax.set_ylabel(r'$P(r)$')
        elif VAR == 'ALPHA':
            y = todo[VAR]
            y = np.rad2deg(y)
            my_bins = np.arange(-180, 181, step=10)
            ax.hist(np.rad2deg(todo['PHI']), bins=my_bins, color='C2',edgecolor='k',
                    density=True, alpha=alpha, label=r'polar $\phi$', zorder=2)
            color = 'C3'
            ax.set_xlabel(r'ángulo [\si{\degree}]')
            ax.set_ylabel(r'$P(\phi)$, $P(\alpha)$')
            #ax.set_xlim(-190, 190)
        elif VAR == 'V_PHI':
            y = v_phi
            my_bins = 30
            color = 'C1'
            label = r'paso = \SI{%d}{\ms}' % todo['step']
            ax.set_xlabel(r'$v_{\phi}$ [\si{\mm\per\s}]')
            ax.set_ylabel(r'$P(v_{\phi})$')
        elif VAR == 'V_R':
            y = v_r
            my_bins = 30
            color = 'C1'
            label = r'paso = \SI{%d}{\ms}' % todo['step']
            ax.set_xlabel(r'$v_{r}$ [\si{\mm\per\s}]')
            ax.set_ylabel(r'$P(v_{r})$')
        elif VAR == 'V':
            y = v
            my_bins = 30
            color = 'C1'
            label = r'paso = \SI{%d}{\ms}' % todo['step']
            ax.set_xlabel(r'$v$ [\si{\mm\per\s}]')
            ax.set_ylabel(r'$P(v)$')
        ax.hist(y, bins=my_bins, color=color, alpha=alpha, edgecolor='k',
                density=True, label=r'orientación $\alpha$', zorder=2)
        #ax.hist(y, bins=my_bins, color=color, edgecolor='k', density=True,
        #        alpha=alpha, zorder=2)
    ax.legend()
    ax.text(0.4, 0.9, r'\textbf{(d)}', size=FS+3, ha='center', transform=ax.transAxes)
    plt.show()


def curvas_calibracion(lista, carpeta):
    fig, ax = plt.subplots()
    for video in lista:
        todo = german(video, carpeta)
        w = velocidad_angular(video, carpeta)
        w = np.rad2deg(w)
        w_avg, w_std = np.mean(w), np.std(w)
        if todo['left'] == 0:
            potencia = todo['right']
            right = ax.errorbar(potencia, abs(w_avg), yerr=w_std, c='b',
                                ls='None', marker='o', ms=7, capsize=5, zorder=4)
        if todo['right'] == 0:
            potencia = todo['left']
            left = ax.errorbar(potencia, abs(w_avg), yerr=w_std, c='r',
                               ls='None', marker='o', ms=7, capsize=5, zorder=4)
    
    w66 = velocidad_angular('left66', carpeta)
    w66 = np.rad2deg(w66)
    w66_avg, w66_std = np.mean(w66), np.std(w66)
    xmin, xmax = ax.get_xlim()
    ax.fill_between([xmin, xmax], w66_avg-w66_std, w66_avg+w66_std,
                    color='violet', alpha=0.3, zorder=2)
    ax.set_xlim(xmin, xmax)
    
    ax.set_xlabel(r'potencia motor')
    ax.set_ylabel(r'$\lvert \dot{\alpha} \rvert$ [\si{\degree\per\s}]')
    ax.legend([left, right], [r'rotación CCW', r'rotación CW'], handletextpad=-0.3)
    plt.show()


def curvas_dcm(lista, carpeta):
    fig, ax = plt.subplots()
    for video in lista:
        todo = german(video, carpeta)
        t, x, y = todo['T'], todo['X_MM'], todo['Y_MM']
        tdcm, dcm = DCM(t, x, y)
        ax.loglog(tdcm[1:], dcm[1:], lw=4, label=r'\SI{%d}{\ms}' % todo['step'], zorder=2)
        ax.set_xlabel(r'tiempo~$t$ [\si{\s}]')
        ax.set_ylabel(r'$\langle\lVert \mathbf{r}(t)-\mathbf{r}_{0} \rVert^{2}\rangle$ [\si{\mm\squared}]')
    
    fit_bal = [(25, np.linspace(0.2, 0.9)),
               (35, np.linspace(0.2, 0.9))]
    fit_diff = [(129, np.linspace(20, 80)),
                (88, np.linspace(20, 80))]
    for i in range(2):
        A_bal, t_bal = fit_bal[i][0], fit_bal[i][1]
        A_diff, t_diff = fit_diff[i][0], fit_diff[i][1]
        ax.loglog(t_bal, A_bal*t_bal**2, c='k', ls='--')
        ax.loglog(t_diff, A_diff*t_diff, c='k', ls='--')
    
    ymin, ymax = ax.get_ylim()
    
    x_bal = np.linspace(0.2, 0.9)
    y_bal = 25 * x_bal**2
    plt.fill_between(x_bal, y_bal, 1e6, color='b', alpha=0.1, zorder=2)
    ax.text(0.42, 30, r'$\propto t^2$', size=FS+1, ha='center')
    x_diff = np.linspace(20, 80)
    y_diff = 88 * x_diff
    plt.fill_between(x_diff, y_diff, 1e6, color='r', alpha=0.1, zorder=2)
    ax.text(40, 12000, r'$\propto t$', size=FS+1, ha='center')
    
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='x', which='major', pad=6)
    ax.text(0.1, 0.9, r'\textbf{(a)}', size=FS+7, ha='center', transform=ax.transAxes)
    ax.legend(title=r'paso temporal', loc='lower right')
    plt.show()


def curvas_dcm_fit(lista, carpeta):
    fig, ax = plt.subplots()
    axins = ax.inset_axes([0.1, 0.4, 0.3, 0.35])
    mark_inset(ax, axins, loc1=3, loc2=4, ec='0.5', zorder=2.1)
    axins.tick_params(labelsize=FS)
    axins.set_xlim(-0.2, 1.2)
    axins.set_ylim(-5, 40)
    for video in lista:
        todo = german(video, carpeta)
        t, x, y = todo['T'], todo['X_MM'], todo['Y_MM']
        tdcm, dcm = DCM(t, x, y)
        ax.plot(tdcm, dcm, lw=4, zorder=2)
        axins.plot(tdcm, dcm, lw=3)
        ax.set_xlabel(r'tiempo~$t$ [\si{\s}]')
        ax.set_ylabel(r'$\langle\lVert \mathbf{r}(t)-\mathbf{r}_{0} \rVert^{2}\rangle$ [\si{\mm\squared}]')
    
    x_bal = np.linspace(0.2, 0.9)
    y_bal = 25 * x_bal**2
    ax.fill_between(x_bal, y_bal, 1e6, color='b', alpha=0.1, zorder=2)
    x_diff = np.linspace(20, 80)
    y_diff = 88 * x_diff
    ax.fill_between(x_diff, y_diff, 1e6, color='r', alpha=0.1, zorder=2)

    for video in lista[1:]:
        todo = german(video, carpeta)
        t, x, y = todo['T'], todo['X_MM'], todo['Y_MM']
        tdcm, dcm = DCM(t, x, y)
        rng_fit = np.logical_and(tdcm>20, tdcm<80)
        params = curve_fit(lineal, tdcm[rng_fit], dcm[rng_fit])
        [m,] = params[0]
        x_fit = tdcm
        y_fit = m * x_fit
        ax.plot(x_fit, y_fit, lw=2, ls='--', c='k')

    axins.fill_between(x_bal, y_bal, 1e6, color='b', alpha=0.1, zorder=2)
    ax.plot([], [], lw=2, ls='--', c='k', label=r'ajuste lineal')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    ax.yaxis.offsetText.set_visible(False)
    ax.text(-0.09, 0.94, r'$\times 10^{-4}$', size=FS+3, ha='center', transform=ax.transAxes)
    ax.text(0.9, 0.1, r'\textbf{(b)}', size=FS+7, ha='center', transform=ax.transAxes)
    ax.set_xlim(-25, 110)
    ax.set_ylim(-800, 13000)
    ax.legend()
    plt.show()


def curvas_dcma(lista, carpeta):
    fig, ax = plt.subplots()
    for video in lista:
        todo = german(video, carpeta)
        t, phi= todo['T'], todo['PHI']
        tdcm, dcm = DCMA(t, phi) # Para no plotear el (0, 0) en loglog
        ax.plot(tdcm[1:], dcm[1:], label=r'\SI{%d}{\s}' % todo['step'], zorder=2)
        ax.set_xlabel(r'tiempo [\si{\s}]')
        ax.loglog(np.linspace(0.1, 10, 100), np.linspace(0.1, 10, 100))
        #ax.set_ylabel(r'$\langle||\phi(t)-\phi_{0}||^{2}\rangle$ [\si{\rad\squared}]')
        #ax.set_xscale('log')
        #ax.set_yscale('log')
    '''
    fit_bal = [(1.8, np.linspace(8, 12)),
               (17, np.linspace(0.2, 1)),
               (31, np.linspace(0.1, 1))]
    fit_diff = [(34, np.linspace(30, 100)),
                (128, np.linspace(28, 100)),
                (85, np.linspace(8, 55))]
    '''
    #for i in range(3):
    #    A_bal, t_bal = fit_bal[i][0], fit_bal[i][1]
    #    A_diff, t_diff = fit_diff[i][0], fit_diff[i][1]
    #    ax.loglog(t_bal, A_bal*t_bal**2, c='k')
    #    ax.loglog(t_diff, A_diff*t_diff, c='k', ls='--')
    #ax.text(57, 570, r'$\propto t$', size=20, ha='center')
    #ax.text(0.25, 31, r'$\propto t^2$', size=20, ha='center')
    ax.legend(title=r'paso temporal')
    plt.show()


def delay_velocidad(lista, carpeta):
    fig, ax = plt.subplots()
    for video in lista:
        t0, v_r, v_phi, v, alpha, beta = velocidad(video, carpeta, dT=0.4)
        rng = np.logical_and(t0>560, t0<600)
        ax.scatter(t0[rng], alpha[rng], c='r', s=2, label=r'orientación~$\alpha(t)$', zorder=2)
        ax.scatter(t0[rng], beta[rng], c='k', s=2, label=r'dirección~$\beta(t)$', zorder=2)
        ax.set_xlabel(r'tiempo~$t$ [\si{\s}]')
        ax.set_ylabel(r'ángulo [\si{\radian}]')
    ax.legend(markerscale=4, handletextpad=0.1, bbox_to_anchor=(0.5, 0.72))
    plt.show()


def corr(lista, carpeta):
    fig, ax = plt.subplots()
    for video in lista:
        todo = german(video, carpeta)
        t, r = todo['T'], todo['ALPHA']
        #rng = t > 100
        #t = t[rng]
        #r = r[rng]
        auto = np.correlate(r, r, "full")
        auto = auto[-len(r):] # Por la simetría al autocorrelacionar
        maximo = np.max(auto)
        auto = auto / maximo
        ax.plot(t, auto, label=r'%d' % todo['step'])
    ax.legend()
    plt.show()


def hexbug(lista, carpeta):
    fig, ax = plt.subplots()
    for video in lista:
        todo = german(video, carpeta)
        t0, v_r, v_phi, alpha, beta = velocidad(video, carpeta)
        #ax.plot(t0, v_phi, label=r'%d' % todo['step'])
        ax.hist(v_phi, bins=30, edgecolor='k', zorder=2)
    ax.legend()
    plt.show()


home = '/home/tom/Documentos/Exactas/Laboratorio 7/'

#folder = 'Arenas Viernes/'
#folder = 'Calibración Miércoles/'
folder = 'Arenas Domingo/'

carpeta = home + 'Data/' + folder

videos = sorted([v for v in os.listdir(carpeta) if os.path.isdir(carpeta+v)])
#videos = ['65_74']
#videos = ['L_2000_69_78']
#videos = ['L_3000_69_79']
#videos = ['left64', 'right78']

#video = 'right65'
#video = '64_74'
print(videos)

#circles(video, i=24754, th=12)
#graficador(video, carpeta=folder)
#histograma(videos, carpeta=folder, VAR='ALPHA')
temporal(videos, carpeta=folder, VAR='ALPHA_UN')
#temporal2(videos, carpeta=folder, VAR='ALPHA')
#curvas_calibracion(videos, carpeta=folder)