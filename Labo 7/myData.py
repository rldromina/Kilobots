import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = os.path.expanduser('~/Escritorio/Data/Calibración/')
#path = os.path.expanduser('~/Escritorio/Data/Experimetos/')
file = '65_74'

#################### PREPARO LOS FRAMES ####################
frames_dir = path + file + '/frames/'
frames_ = []

for fn in os.listdir(frames_dir):
    if fn.endswith('.jpg'):
        frames_.append(fn)
    else:
        continue

frames = sorted(frames_, key=lambda x: int(x[6:-4])) # Porque 'frame_####.jpg'
N = len(frames)

#################### IMPORTO LOS METADATOS ####################
meta_fname = path + file + '/' + file + '_meta.csv'
meta = np.genfromtxt(meta_fname, delimiter=',', dtype=int, names=True)

print('Los parámetros que se van a usar son:')
print(meta.dtype.names)
print(meta)

#################### DETECTO LA ARENA ####################
img_arena = cv2.imread(frames_dir + frames[0], 0) # Uso el primer frame

circle_arena = cv2.HoughCircles(img_arena, cv2.HOUGH_GRADIENT, 1, 20, 
                                param1=50, param2=22, minRadius=meta['r_ar']-2, 
                                maxRadius=meta['r_ar']+2)

# Arena de centro (i_arena, j_arena) y radio 'r_arena'
i_arena, j_arena = circle_arena[0, 0, 1], circle_arena[0, 0, 0]
r_arena = circle_arena[0, 0, 2]
"""
#------------------------------------------------------------------------------
print('\nLa arena detectada es...\n', circle_arena)

circle_arena = np.uint16(np.around(circle_arena))
cimg_arena = cv2.cvtColor(img_arena, cv2.COLOR_GRAY2BGR)

for i in circle_arena[0,:]:
    cv2.circle(cimg_arena, (i[0],i[1]), i[2], (0,255,0), 2) # Perímetro
    cv2.circle(cimg_arena, (i[0],i[1]), 2, (0,0,255), 3) # Centro

cv2.imshow('arena detectada', cimg_arena)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------------------------------------
"""
#################### LEVANTO LA DATA #################### 
# En estos arrays voy a guardar las posiciones (i,j) de
# los ojalillos (left y right) y en qué instante de tiempo sale cada frame.
I_LEFT, J_LEFT = np.zeros(N), np.zeros(N)
I_RIGHT, J_RIGHT = np.zeros(N), np.zeros(N)
TIME = np.zeros(N)

for c, fr in enumerate(frames):
    print('\n--------------- %s (%d de %d) ---------------' % (fr, c+1, N))

    img = cv2.imread(frames_dir + fr, 0) # Cargo un frame en escala de grises
    #img = cv2.medianBlur(img, 3) # Promedio sobre pxs que elimina el ruido

    # Recorto entornos cuadrados alrededor de los centros de los ojalillos
    img_left = img[meta['i_left']-meta['e'] : meta['i_left']+meta['e']+1,
                   meta['j_left']-meta['e'] : meta['j_left']+meta['e']+1]
    img_right = img[meta['i_right']-meta['e'] : meta['i_right']+meta['e']+1,
                    meta['j_right']-meta['e'] : meta['j_right']+meta['e']+1]

    # Busco los círculos (ojalillos) en esos recortes
    circles_left = cv2.HoughCircles(img_left, cv2.HOUGH_GRADIENT, 1, 20, 
                                    param1=50, param2=int(meta['th']), 
                                    minRadius=meta['r_oj']-1, 
                                    maxRadius=meta['r_oj']+1)
    circles_right = cv2.HoughCircles(img_right, cv2.HOUGH_GRADIENT, 1, 20, 
                                     param1=50, param2=int(meta['th']), 
                                     minRadius=meta['r_oj']-1, 
                                     maxRadius=meta['r_oj']+1)

    # Posiciones LOCALES (i0,j0) de los ojalillos en esos recortes
    i0_left, j0_left = circles_left[0, 0, 1], circles_left[0, 0, 0]
    i0_right, j0_right = circles_right[0, 0, 1], circles_right[0, 0, 0]

    # Recupero las posiciones GLOBALES (i,j) a partir de las locales
    i_left, j_left = i_left + (i0_left-meta['e']), j_left + (j0_left-meta['e'])
    i_right, j_right = i_right + (i0_right-meta['e']), j_right + (j0_right-meta['e'])

    print('i_left = %.1f, j_left = %.1f        i_right = %.1f, j_right = %.1f' 
          % (i_left, j_left, i_right, j_right))

    # Guardo las posiciones y el instante de tiempo
    I_LEFT[c], J_LEFT[c] = i_left, j_left
    I_RIGHT[c], J_RIGHT[c] = i_right, j_right
    TIME[c] = int(fr[6:-4]) / meta['fps']

    # Para poder recortar en la siguiente iteración, "redondeo" y lo paso a int
    i_left, j_left = int(np.around(i_left)), int(np.around(j_left))
    i_right, j_right = int(np.around(i_right)), int(np.around(j_right))

DIST = np.sqrt((I_LEFT-I_RIGHT)**2 + (J_LEFT-J_RIGHT)**2) # Distancia entre ojalillos

print('\n-------------- RESUMEN DEL TRACKEO PARA %s.mp4 --------------' % file)
print('Cargamos %d frames: %s, %s, ..., %s' % (N, frames[0], frames[1], frames[-1]))
print('Dimensión (matricial) de los frames: %s' % (img.shape,))
print('Arena de centro (i = %.1f, j = %.1f) y radio = %.1f' % (i_arena, j_arena, r_arena))
print('Distancias entre ojalillos...')
print('...promedio = %.2f px' % np.mean(DIST))
print('...mínima   = %.2f px (en el %s)' % (np.amin(DIST), frames[np.argmin(DIST)]))
print('...máxima   = %.2f px (en el %s)' % (np.amax(DIST), frames[np.argmax(DIST)]))

################################### GUARDO LA DATA COMO .CSV ###################################
I_ARENA, J_ARENA = i_arena * np.ones(N), j_arena * np.ones(N)
R_ARENA = r_arena * np.ones(N)

DATA = np.stack((TIME, I_LEFT, J_LEFT, I_RIGHT, J_RIGHT, I_ARENA, J_ARENA, R_ARENA), axis=-1)
my_data = 'TIME,I_LEFT,J_LEFT,I_RIGHT,J_RIGHT,I_ARENA,J_ARENA,R_ARENA'

np.savetxt(archivo_data, DATA, fmt='%.5f', delimiter=',', comments='', header=my_data)

################################### HISTOGRAMA DE DISTANCIAS ###################################
archivo_data   = path + file + '/' + file + '_data.csv'
d_min, d_max = np.floor(np.amin(DIST)), np.ceil(np.amax(DIST))
my_bins = np.linspace(d_min, d_max, int(10*(d_max-d_min)+1))

fig, ax = plt.subplots(1, 1)

ax.hist(DIST, bins=my_bins)

titulo = r'$d_\mathrm{min}=%.2f \qquad d_\mathrm{avg}=%.2f \qquad d_\mathrm{max}=%.2f$'
ax.set_title(titulo % (np.amin(DIST), np.mean(DIST), np.amax(DIST)))
ax.set_xlabel(r'$d_\mathrm{ojalillos}$')
ax.grid()

fig.tight_layout()
plt.show()
