import os
import numpy as np
import cv2

path = os.path.expanduser('~/Escritorio/Data/Moneda/')
file = '1000_1h'

tasa = 1 # Voy a guardar 1 de cada 'tasa' frames
time = 1000 # Tiempo de LED prendido en un tirada (en milisegundos)
e = 10 # Tamaño del recorte donde voy a ver prendido/apagado

#################### CARGO EL VIDEO Y CREO SU CARPETA ####################
video_fname = path + file + '.mp4'
frames_dir = path + file + '/frames/'

cap = cv2.VideoCapture(video_fname)

try:
    os.makedirs(frames_dir)
    print("Se creó '%s'\n" % frames_dir)
except FileExistsError:
    print("Ya existe '%s'\n" % frames_dir)

#################### EXTRAIGO LOS FRAMES ####################
fps = cap.get(cv2.CAP_PROP_FPS)
frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
segundos = frames_count / fps

print("Este video '%s.mp4', grabado a %d fps, tiene %d frames y dura %d min." 
      % (file, fps, frames_count, segundos/60))
print('Extraigo 1 de cada %d frames...' % tasa)

i = 1
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    if i%tasa == 0:
        frame_fname = frames_dir + 'frame_' + str(i) + '.jpg'
        cv2.imwrite(frame_fname, frame)
    if i%1000 == 0:
        pc = (i/frames_count) * 100
        print('...%d%% completo' % pc)
    i+=1

cap.release()
cv2.destroyAllWindows()

#################### EXPORTO LOS METADATOS COMO .CSV ####################
meta = np.array([[0, 0, e, tasa, time, fps, frames_count, segundos]])
meta_fname = path + file + '/' + file + '_meta.csv'
meta_header = 'i,j,e,tasa,time,fps,frames_count,segundos'

np.savetxt(meta_fname, meta, header=meta_header, fmt='%d', delimiter=',', comments='')
