import os
import numpy as np
import pandas as pd
import cv2

REPO = os.path.expanduser('~/Escritorio/Repos/Kilobots') 
file = '1minuto'

step = 1000 # Tiempo mínimo de Kilobot girando (en milisegundos)
left, right = 65, 74 # Calibración del Kilobot
tasa = 1 # Voy a guardar uno de cada 'tasa' frames
radio = 150 # Radio de la arena (en milímetros)

#################### CREO LA CARPETA DE FRAMES ####################
frames_dir = f'{REPO}/Media/{file}(frames)'
try:
    os.makedirs(frames_dir)
    print(f'Se creó {frames_dir}')
except FileExistsError:
    print(f'Ya existe {frames_dir}')

#################### CARGO EL VIDEO Y EXTRAIGO SUS FRAMES ####################
video_fname = f'{REPO}/Media/{file}.mp4'
cap = cv2.VideoCapture(video_fname)

fps = cap.get(cv2.CAP_PROP_FPS)
frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
segundos = frames_count / fps

print(f"Este video '{file}.mp4', grabado a {fps:.1f} fps, "
      f"tiene {frames_count} frames y dura {segundos/60:.1f} minutos")
print(f'Extraigo 1 de cada {tasa} frames...')

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    if i%tasa == 0:
        cv2.imwrite(f'{frames_dir}/frame_{str(i)}.jpg', frame)      
    if i%1000 == 0:
        print(f'...{i/frames_count:.0%} completo')
    i+=1

cap.release()

#################### CREO LA CARPETA DE CSV's ####################
csv_dir = f'{REPO}/Data/{file}(csv)'
try:
    os.makedirs(csv_dir)
    print(f'Se creó {csv_dir}')
except FileExistsError:
    print(f'Ya existe {csv_dir}')

#################### EXPORTO LOS METADATOS COMO .CSV ####################
meta_dict = {
    'step': [step], 'left': [left], 'right': [right], 
    'tasa': [tasa], 'radio': [radio], 'fps': [fps], 
    'frames_count': [frames_count], 'segundos': [segundos],
}
meta = pd.DataFrame(meta_dict)
meta_fname = f'{csv_dir}/{file}_meta.csv'
meta.to_csv(meta_fname, index=False)
