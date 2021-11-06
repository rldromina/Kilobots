import os
import numpy as np
import cv2

path = os.path.expanduser('~/Escritorio/Data/Moneda/')
file = '1000_1h'

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

print('Los metadatos asociados a este video son:')
print(meta.dtype.names)
print(meta)

#################### LEVANTO LA DATA #################### 
# Acá guardo la evolución temporal de la intensidad del LED del Kilobot
INT = np.empty(N)
TIME = np.empty(N)

for c, fr in enumerate(frames):
    print('\n--------------- %s (%d de %d) ---------------' % (fr, c+1, N))

    img = cv2.imread(frames_dir + fr, 0) # Cargo un frame en escala de grises
    #img = cv2.medianBlur(img, 3) # Promedio sobre pxs que elimina el ruido

    # Recorto un entorno cuadrado alrededor del centro del LED
    img_ = img[meta['i']-meta['e'] : meta['i']+meta['e']+1, 
               meta['j']-meta['e'] : meta['j']+meta['e']+1]

    # Intensidades como el valor medio de los elementos de matriz
    I = np.mean(img_)
    print('intensidad I = %d' % I)

    # Guardo la intensidad I y el instante de tiempo
    INT[c] = I
    TIME[c] = int(fr[6:-4]) / meta['fps']

print('\n--------------- RESUMEN PARA %s.mp4 ---------------' % file)
print('Cargamos %d frames de dimensión matricial %s:\n%s, %s, %s, ... , %s' 
      % (N, img.shape, frames[0], frames[1], frames[2], frames[-1]))

#################### EXPORTO LA DATA COMO .CSV ####################
data = np.stack((TIME, INT), axis=-1)
data_fname = path + file + '/' + file + '_data.csv'
data_header = 'TIME,INT'

np.savetxt(data_fname, data, header=data_header, fmt='%.4f', delimiter=',', comments='')
