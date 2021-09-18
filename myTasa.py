import os
import numpy as np
import cv2
import datetime

path = '/home/tom/Escritorio/Paper/Data/'
video = '1000'

tasa = 1 # Voy a guardar uno de cada 'tasa' frames
time = 1000 # Tiempo de LED prendido (en milisegundos)

archivo_video  = path + video + '.mp4'
carpeta_frames = path + video + '/frames/'
archivo_param  = path + video + '/' + video + '_param.csv'

################################ CARGO EL VIDEO Y CREO SU CARPETA ################################
cap = cv2.VideoCapture(archivo_video)

try:
    os.makedirs(carpeta_frames)
    print("Se creó '%s'\n" % carpeta_frames)
except FileExistsError:
    print("Ya existe '%s'\n" % carpeta_frames)

################################ EXTRAIGO LOS FRAMES ################################
fps = int(cap.get(cv2.CAP_PROP_FPS))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
length = datetime.timedelta(seconds=count/fps)

print("Este video '%s.mp4' (grabado a %d fps) dura %s y tiene %d frames\n" 
    % (video, fps, str(length).split('.')[0], count))
print('Extraigo frames con una tasa = %d...' % tasa)

i = 1

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    if i%tasa == 0:
        cv2.imwrite(carpeta_frames + 'frame_' + str(i) + '.jpg', frame)
        if i%1200 == 0:
            tiempo_parcial = datetime.timedelta(seconds=i/fps)
            print('...%s completo (frame %d)' % (str(tiempo_parcial), i))
    i+=1

cap.release()
cv2.destroyAllWindows()

################################ GUARDO LOS PARÁMETROS COMO .CSV ################################
PARAM = np.array([[tasa, time, fps, count]])
my_params = 'tasa,time,fps,count'

np.savetxt(archivo_param, PARAM, fmt='%d', delimiter=',', comments='', header=my_params)
