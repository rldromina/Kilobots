import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = '/home/tom/Escritorio/Data/Calibración/'
video = '65_74'

archivo_video  = path + video + '.mp4'
carpeta_frames = path + video + '/frames/'
archivo_param  = path + video + '/' + video + '_param.csv'
archivo_data   = path + video + '/' + video + '_data.csv'

################################### PREPARO LOS FRAMES ###################################
frames_ = []

for filename in os.listdir(carpeta_frames):
    if filename.endswith('.jpg'):
        frames_.append(filename)
    else:
        continue

frames = sorted(frames_, key=lambda x: int(x[6:-4])) # Porque 'frame_#####.jpg'
N = len(frames)

################################### IMPORTO LOS PARÁMETROS ###################################
param = np.genfromtxt(archivo_param, delimiter=',', dtype=int, names=True)

r_ar, r_oj = param['r_ar'], param['r_oj']
e, th = param['e'], int(param['th'])
fps = param['fps']

print('Los parámetros que se van a usar son:')
print(param.dtype.names)
print(param)

################################### DETECTO LA ARENA ###################################
img_arena = cv2.imread(carpeta_frames + frames[0], 0) # Uso el primer frame

circles_arena = cv2.HoughCircles(img_arena, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=22, 
                                minRadius=r_ar-2, maxRadius=r_ar+2)

i_arena, j_arena = circles_arena[0, 0, 1], circles_arena[0, 0, 0] # Centro de la arena
r_arena = circles_arena[0, 0, 2] # Radio de la arena
"""
#------------------------------------------------------------------------------------------
print('\nLa arena detectada es...\n', circles_arena)

circles_arena = np.uint16(np.around(circles_arena))
cimg_arena = cv2.cvtColor(img_arena, cv2.COLOR_GRAY2BGR)

for i in circles_arena[0,:]:
    cv2.circle(cimg_arena, (i[0],i[1]), i[2], (0,255,0), 2) # Dibujo el perímetro
    cv2.circle(cimg_arena, (i[0],i[1]), 2, (0,0,255), 3) # Dibujo el centro

cv2.imshow('arena detectada', cimg_arena)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------------------------------------------------
"""
################################### LEVANTO LA DATA ################################### 
# En estos arrays voy a guardar las posiciones (i,j) de los ojalillos (left y right)
# y en qué instante de tiempo sale cada frame.
I_LEFT, J_LEFT = np.zeros(N), np.zeros(N)
I_RIGHT, J_RIGHT = np.zeros(N), np.zeros(N)
TIME = np.zeros(N)

# Empiezo a trackear con estos guesses
i_left, j_left = param['i_left'], param['j_left']
i_right, j_right = param['i_right'], param['j_right']

for c, fr in enumerate(frames):
    print('\n------------------ %s (%d de %d) ------------------' % (fr, c+1, N))

    img = cv2.imread(carpeta_frames + fr, 0) # Cargo un frame en escala de grises
    #img = cv2.medianBlur(img, 3) # Un promedio sobre pxs que ayuda a eliminar el ruido

    # Recorto entornos cuadrados alrededor de (i,j)
    img_left = img[i_left-e : i_left+e+1, j_left-e : j_left+e+1]
    img_right = img[i_right-e : i_right+e+1, j_right-e : j_right+e+1]

    # Busco los círculos (ojalillos) en esos recortes
    circles_left = cv2.HoughCircles(img_left, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=th, 
                                    minRadius=r_oj-1, maxRadius=r_oj+1)
    circles_right = cv2.HoughCircles(img_right, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=th, 
                                    minRadius=r_oj-1, maxRadius=r_oj+1)

    # Posiciones LOCALES (i0,j0) de los ojalillos en esos recortes
    i0_left, j0_left = circles_left[0, 0, 1], circles_left[0, 0, 0]
    i0_right, j0_right = circles_right[0, 0, 1], circles_right[0, 0, 0]

    # Recupero las posiciones GLOBALES (i,j) a partir de las locales
    i_left, j_left = i_left + (i0_left-e), j_left + (j0_left-e)
    i_right, j_right = i_right + (i0_right-e), j_right + (j0_right-e)

    print('i_left = %.1f, j_left = %.1f        i_right = %.1f, j_right = %.1f'
            % (i_left, j_left, i_right, j_right))

    # Guardo las posiciones y el instante de tiempo
    I_LEFT[c], J_LEFT[c] = i_left, j_left
    I_RIGHT[c], J_RIGHT[c] = i_right, j_right
    TIME[c] = int(fr[6:-4]) / fps

    # AHORA SÍ "redondeo" y lo paso a int() (Para poder recortar en la siguiente iteración)
    i_left, j_left = int(np.around(i_left)), int(np.around(j_left))
    i_right, j_right = int(np.around(i_right)), int(np.around(j_right))

DIST = np.sqrt((I_LEFT-I_RIGHT)**2 + (J_LEFT-J_RIGHT)**2) # Distancia entre ojalillos

print('\n-------------- RESUMEN DEL TRACKEO PARA %s.mp4 --------------' % video)
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
