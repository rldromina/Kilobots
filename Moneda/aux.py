import os
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('../estilo_latex.mplstyle')
from matplotlib.gridspec import GridSpec
import math
import random
import itertools
import cv2


plt.rc('font', size=12)
fps = 30
tasa = 1
frames = int(fps/tasa) # Cuántos frames respresentan 1 segundo


def importo(directory, file):
    path = os.path.expanduser('~/Escritorio/Repos/Kilobots/Data/') + directory + '/' 
    data_fname = path + file + '/' + file + '_data.csv'
    meta_fname = path + file + '/' + file + '_meta.csv'
    data = np.genfromtxt(data_fname, delimiter=',', names=True)
    meta = np.genfromtxt(meta_fname, delimiter=',', names=True)
    return data, meta


def umbral(x):
    """A los elementos del array 'x' que son menores al valor
    de umbral 'th' les asigno el valor 0 (LED apagado). A los demás
    les asigno el valor 1 (LED prendido).
    """
    th = 20
    y = np.where(x<th, 0, 1)
    return y


def on_off(x):
    """Fracciones de tiempo con el LED prendido (on, 1) y apagado (off, 0).
    """
    on = np.count_nonzero(x==1) / len(x)
    off = np.count_nonzero(x==0) / len(x)
    return on, off


def autocorrelacion(x):
    """Calcula la autocorrelación R de una señal x(t).
    """
    x_avg = np.mean(x)
    auto = np.correlate(x-x_avg, x-x_avg, mode='full')
    auto = auto[-len(x):]
    maximo = np.max(auto)
    R = auto / maximo
    return R


def contador_consecutividades(x):
    groups = itertools.groupby(x)
    # Todas las longitudes (normalizadas) de las consecutividades. Estas
    # longitudes pueden resultar NO enteras por efectos de la filmación.
    lenght = [sum(1 for _ in group)/frames for label, group in groups]
    # No debo considerar las 'tiradas' inicial y final porque son incompletas
    lenght = lenght[1:-1]
    # Qué longitudes se presentan efectivamente
    N = np.arange(1, math.ceil(max(lenght))+1)
    # Cuántas consecutividades tienen determinada longitud. Como puede haber
    # longitudes NO enteras, cuento cuántas de estas caen dentro de (n-e, n+e)
    e = 0.25
    count = [np.logical_and(lenght>n-e, lenght<n+e).sum() for n in N]
    return count


def moneda(tiradas):
    """Tiro la moneda 'tiradas' veces. Una tirada por segundo, el 
    cual está representado por 'frames' puntos en el vector 't' de tiempos.
    """
    x = np.empty(tiradas*frames)
    t = np.arange(len(x)) / frames
    random.seed()
    for i in range(tiradas):
        coin = random.randint(0, 1)
        seg = frames * i
        x[seg:seg+frames] = coin
    return t, x


def estadistica_moneda(realizaciones, tiradas):
    """Tiro la moneda 'tiradas' veces. Esto es una realización que repito
    'realizaciones' veces. Sobre estas calculo la aparición media
    de las longitudes de las consecutividades.
    """
    todas = []
    for i in range(realizaciones):
        t, I = moneda(tiradas)
        count = contador_consecutividades(I)
        todas.append(count)
    media = [np.mean(k) for k in itertools.zip_longest(*todas, fillvalue=0)]
    return media


def graficador(directory, file):

    data, meta = importo(directory, file)
    t = data['TIME']
    I = data['INT']

    count = contador_consecutividades(umbral(I))
    media = estadistica_moneda(100, int(meta['segundos']))
    t_, I_ = moneda(int(meta['segundos']))
    
    ax0 = plt.subplot(321)
    ax0.plot(t, umbral(I), 'o', label=r'umbral aplicado')
    ax0.plot(t, I/255, 'x', c='C2', label=r'medición cruda (normalizada)')
    ax0.set_ylabel(r'intensidad $I(t)$')
    ax0.legend(title=r'%.9f on - %.9f off' % on_off(umbral(I)))

    ax1 = plt.subplot(322)
    ax1.plot(t_, I_, 'x', c='C1', label=r'una simulación con igual cantidad de tiradas')
    ax1.set_ylabel(r'intensidad $I_{\mathrm{sim}}(t)$')
    ax1.legend(title=r'%.3f on - %.3f off' % on_off(I_))
    
    ax2 = plt.subplot(323)
    ax2.plot(t, autocorrelacion(umbral(I)), '.')
    ax2.set_xlabel(r'tiempo $t$ [\si{\s}]')
    ax2.set_ylabel(r'autocorrelación de $I(t)$')

    ax3 = plt.subplot(324)
    ax3.plot(t_, autocorrelacion(I_), c='C1', marker='.')
    ax3.set_xlabel(r'tiempo $t$ [\si{\s}]')
    ax3.set_ylabel(r'autocorrelación de $I_{\mathrm{sim}}(t)$')

    ax4 = plt.subplot(313)
    ax4.plot(range(1, len(count)+1), count, 'o', label='medido')
    ax4.plot(range(1, len(media)+1), media, 'x', label='simulado (100 veces)')
    ax4.set_xlabel(r'longitud $n$ de la consecutividad')
    ax4.set_ylabel(r'frecuencia')
    ax4.legend()
    
    plt.show()


graficador('Moneda', 'seed_hard_1h')
