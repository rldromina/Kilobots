import os
import numpy as np
import pandas as pd
import cv2

cwd = os.getcwd()
exp = os.path.basename(cwd)
file = 'cuadro'

#################### PREPARO LOS FRAMES ####################
frames_dir = f'{cwd}/../Media/{exp}/{file}(frames)'
frames_ = os.listdir(frames_dir)
frames = sorted(frames_, key=lambda x: int(x[6:-4])) # Porque 'frame_###.jpg'
N = len(frames)

#################### IMPORTO Y ACTUALIZO LOS METADATOS ####################
csv_dir = f'{cwd}/../Data/{exp}/{file}(csv)'
meta_fname = f'{csv_dir}/{file}_meta.csv'
meta = pd.read_csv(meta_fname)

meta['e'] = e = 10 # 2*e es el lado del cuadrado que voy a recortar

meta.to_csv(meta_fname, index=False)
print(f'Los metadatos asociados a esta medición son:\n{meta}')

#################### SELECCIONO EL CENTRO ####################
global clicks
clicks = []

def callback(event, x, y, flags, param):
    if event == 1:
        clicks.append((x,y))

cv2.namedWindow('presione ESC para cerrar')
cv2.setMouseCallback('presione ESC para cerrar', callback)
img_frame0 = cv2.imread(f'{frames_dir}/{frames[0]}')

while True:
    cv2.imshow('presione ESC para cerrar', img_frame0)    
    k = cv2.waitKey(1)
    if k == 27:
        break
cv2.destroyAllWindows()

i, j = clicks[0][1], clicks[0][0] # Centro del LED

#################### LEVANTO LA DATA #################### 
# Acá guardo la evolución temporal de la intensidad del LED del Kilobot
TIME, INT = np.empty(N), np.empty(N)

for c, fr in enumerate(frames):
    print(f'\n--------------- {fr} ({(c+1)/N:.0%}) ---------------')
    # Cargo la imagen de un frame en escala de grises
    img = cv2.imread(f'{frames_dir}/{fr}', 0)
    # Recorto un entorno cuadrado alrededor del centro del LED
    img_ = img[i-e:i+e+1, j-e:j+e+1]
    #img_ = img # Si quiero medir la intensidad de TODO el ambiente
    # Intensidades como el valor medio de los elementos de matriz
    I = np.mean(img_)
    print(f'intensidad I = {I:.4f}')
    # Guardo la intensidad I y el instante de tiempo
    INT[c] = I
    TIME[c] = int(fr[6:-4]) / meta['fps']

print(f"\n--------------- Resumen para '{file}.mp4' ---------------")
print(f'Cargamos {N} frames de dimensión matricial {img.shape}:\n'
      f'{frames[0]}, {frames[1]}, {frames[2]}, ... , {frames[-1]}')

#################### EXPORTO LA DATA COMO .CSV ####################
data = pd.DataFrame({'TIME': TIME, 'INT': INT})
data_fname = f'{csv_dir}/{file}_data.csv'
data.to_csv(data_fname, index=False)
