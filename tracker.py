import os, datetime
from datetime import date

import numpy as np
import pandas as pd

import cv2
import shutil

from importlib import reload

import aux_tracker
reload(aux_tracker)
from aux_tracker import evolucion_temporal

import tqdm
import time

today = date.today().strftime("%b-%d-%Y")

def mover_y_renombrar(origen, destino):
    """Si el usuario lo desea, los archivos alojados en 'origen'
    son movidos a 'destino' y renombrados.
    """
    if os.listdir(origen) == []:
        print(f'¡No hay archivos en {origen}!')
    else:
        print(f'Los archivos alojados en {origen} son...\n{os.listdir(origen)}')
        for file in os.listdir(origen):
            prompt_mv = input(f'¿Desea mover {file} a {destino}? [y/n]: ')
            if prompt_mv.lower() == 'y':
                print('Moviendo...')
                shutil.move(f'{origen}/{file}', destino)
                print('---------- Movido ----------')
                prompt_rn = input(f'¿Desea renombrar a {file}? [y/n]: ')
                if prompt_rn.lower() == 'y':
                    newname = input('Nuevo nombre: ').rstrip()
                    ext = file.split('.')[1]
                    while (f'{newname}.{ext}' in os.listdir(destino)) == True:
                        newname = input('Ese nombre ya está en uso, pruebe con otro: ')
                    os.rename(f'{destino}/{file}', f'{destino}/{newname}.{ext}')
                    print('---------- Renombrado ----------')

def tasa(file, step, stop, left, right):
    tasa = 1 # Voy a guardar uno de cada 'tasa' frames
    radio = 150 # Radio de la arena (en milímetros)

    ############### CREO LA CARPETA DE FRAMES ###############4
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    frames_dir = f'{repo_dir}/Media/{today}/{file}(frames)'
    try:
        os.makedirs(frames_dir)
        print(f'Se creó {frames_dir}')
    except FileExistsError:
        print(f'Ya existe {frames_dir}')

    ############### CARGO EL VIDEO Y EXTRAIGO SUS FRAMES ###############
    video_fname = f'{repo_dir}/Media/{today}/{file}.mp4'
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

    ############### CREO LA CARPETA DE CSV's ###############
    csv_dir = f'{repo_dir}/Data/{today}/{file}(csv)'
    try:
        os.makedirs(csv_dir)
        print(f'Se creó {csv_dir}')
    except FileExistsError:
        print(f'Ya existe {csv_dir}')

    ############### EXPORTO LOS METADATOS COMO .CSV ###############
    meta_dict = {
        'step': [step], 'stop': [stop], 'left': [left], 'right': [right],
        'tasa': [tasa], 'radio': [radio],
        'fps': [fps], 'frames_count': [frames_count], 'segundos': [segundos],
    }
    meta = pd.DataFrame(meta_dict)
    meta_fname = f'{csv_dir}/{file}_meta.csv'
    meta.to_csv(meta_fname, index=False)

def tasa_prompt():
    file = input('Escriba el nombre del archivo: ').rstrip()
    step = int(input('Escriba el step (en milisegundos): '))
    stop = int(input('Escriba el STOP (en milisegundos): '))
    left = int(input("Escriba la calibración 'left': "))
    right = int(input("Escriba la calibración 'right': "))

    prompt_ok = input(f'¿Los valores escritos arriba son correctos? [y/n]: ')

    if prompt_ok.lower() == 'y':
        print(f'\n¡Bien, entonces ejecuto tasa()!')
        tasa(file, step, stop, left, right)
    elif prompt_ok.lower() == 'n':
        print('\nOk, inténtelo de nuevo con los valores correctos...')
        tasa_prompt()

    return file

def graficador_circulos(img, circles):
    """Grafica en la imagen 'img' de tres canales, los centros 
    y perímetros de los círculos detectados en ella: estos son 
    de la forma 'circles' = [[[x1,y1,r1], [x2,y2,r2], ...]].
    Para una fácil identificación, el primer círculo lo dibuja
    de color verde; todos los demás, si los hay, de color rojo.
    """
    circles_int = np.uint16(np.around(circles))
    for c, v in enumerate(circles_int[0,:]):
        if c == 0:
            BGR = (0, 255, 0)
        else:
            BGR = (0, 0, 255)
        cv2.circle(img, (v[0],v[1]), v[2], BGR, 2) # Perímetro
        cv2.circle(img, (v[0],v[1]), 2, BGR, 3) # Centro
    cv2.imshow('circulos detectados', img)
    # Presione cualquier tecla para cerrar la ventana y continuar
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detector_ojal(file, frame, th, e, ojal_guess, i_l, j_l, i_r, j_r):
    frames_dir = f'{repo_dir}/Media/{today}/{file}(frames)'

    # Cargo la imagen de un frame en escala de grises
    img = cv2.imread(f'{frames_dir}/{frame}', 0)
    # Recorto entornos cuadrados alrededor de los centros de los ojalillos
    img_l = img[i_l-e:i_l+e+1, j_l-e:j_l+e+1]
    img_r = img[i_r-e:i_r+e+1, j_r-e:j_r+e+1]
    # Busco los ojalillos en esos recortes
    circles_l = cv2.HoughCircles(
        img_l, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=th, 
        minRadius=ojal_guess-1, maxRadius=ojal_guess+1)
    circles_r = cv2.HoughCircles(
        img_r, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=th, 
        minRadius=ojal_guess-1, maxRadius=ojal_guess+1)

    # Posiciones LOCALES (i0,j0) de los ojalillos en esos recortes
    i0_l, j0_l = circles_l[0, 0, 1], circles_l[0, 0, 0]
    i0_r, j0_r = circles_r[0, 0, 1], circles_r[0, 0, 0]
    # Recupero las posiciones GLOBALES (i,j) a partir de las locales
    i_l, j_l = i_l + (i0_l-e), j_l + (j0_l-e)
    i_r, j_r = i_r + (i0_r-e), j_r + (j0_r-e)

    return i_l, j_l, i_r, j_r, circles_l, circles_r

def clicks(img_frame):
    global clicks
    clicks = []

    def callback(event, x, y, flags, param):
        if event == 1:
            clicks.append((x,y))

    cv2.namedWindow('presione ESC para cerrar')
    cv2.setMouseCallback('presione ESC para cerrar', callback)
    while True:
        cv2.imshow('presione ESC para cerrar', img_frame)    
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

    i_l, j_l = clicks[0][1], clicks[0][0] # Centro del ojalillo izquierdo
    i_r, j_r = clicks[1][1], clicks[1][0] # Centro del ojalillo derecho

    return i_l, j_l, i_r, j_r

def confirmacion(file, frame, th, e, ojal_guess):

    frames_dir = f'{repo_dir}/Media/{today}/{file}(frames)'
    # Cargo la imagen de un frame en escala de grises
    img = cv2.imread(f'{frames_dir}/{frame}')

    i_l, j_l, i_r, j_r = clicks(img)
    i_l, j_l, i_r, j_r, circles_l, circles_r = detector_ojal(
        file, frame, th, e, ojal_guess, i_l, j_l, i_r, j_r)

    graficador_circulos(img, [[[j_l, i_l, ojal_guess]]])
    graficador_circulos(img, [[[j_r, i_r, ojal_guess]]])

def data(file):
    ############### PREPARO LOS FRAMES ###############
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    frames_dir = f'{repo_dir}/Media/{today}/{file}(frames)'
    frames_ = os.listdir(frames_dir)
    frames = sorted(frames_, key=lambda x: int(x[6:-4])) # Porque 'frame_###.jpg'
    N = len(frames)

    ############### IMPORTO Y ACTUALIZO LOS METADATOS ###############
    csv_dir = f'{repo_dir}/Data/{today}/{file}(csv)'
    meta_fname = f'{csv_dir}/{file}_meta.csv'
    meta = pd.read_csv(meta_fname)

    meta['ojal_guess'] = ojal_guess = 6 # Guess para el radio de los ojalillos (en píxeles)
    meta['e'] = e = 16 # 2*e = lado del cuadrado que voy a recortar (en píxeles)
    meta['th'] = th = 10 # Umbral 'param2' del método HoughCircles

    meta.to_csv(meta_fname, index=False)
    print(f'Los metadatos asociados a esta medición son...\n{meta}')

    ############### SELECCIONO LOS CENTROS ###############
    frame0_fname = f'{frames_dir}/{frames[0]}'
    img_frame0 = cv2.imread(frame0_fname)
    print(f'Seleccionemos los ojalillos.'
          f'Primero el izquierdo (más cerca del jumper) y luego el derecho.')
    i_l, j_l, i_r, j_r = clicks(img_frame0)

    ############### LEVANTO LA DATA ###############
    # Acá guardo la evolución temporal de las posiciones
    # de los centros de los ojalillos adheridos al Kilobot
    I_L, J_L = np.empty(N), np.empty(N)
    I_R, J_R = np.empty(N), np.empty(N)
    TIME = np.empty(N)

    for c, fr in enumerate(frames):
        print(f'\n--------------- {fr} ({(c+1)/N:.0%}) ---------------')
        
        i_l, j_l, i_r, j_r, circles_l, circles_r = detector_ojal(
            file, fr, th, e, ojal_guess, i_l, j_l, i_r, j_r)

        print(f'i_l = {i_l}, j_l = {j_l}        i_r = {i_r}, j_r = {j_r}')
        # Guardo las posiciones de los centros y el instante de tiempo
        I_L[c], J_L[c] = i_l, j_l
        I_R[c], J_R[c] = i_r, j_r
        TIME[c] = int(fr[6:-4]) / meta['fps']
        # Esto es necesario para poder recortar en la siguiente iteración (!)
        i_l, j_l = int(np.around(i_l)), int(np.around(j_l))
        i_r, j_r = int(np.around(i_r)), int(np.around(j_r))
        
        if (circles_l is None) or (circles_r is None):
            # Para identificar el error, muestro las imágenes de
            # los frames 'current' (el que falló) y 'last' (exitoso)
            print(f'¡Error en {fr}! No se identificó al menos un ojalillo.'
                  f'Use las imágenes que le muestro para identificar el error.')
            img_last = cv2.imread(f'{frames_dir}/{frames[c-1]}')
            img_current = cv2.imread(f'{frames_dir}/{frames[c]}')
            circles_last = [[[j_l, i_l, ojal_guess], [j_r, i_r, ojal_guess]]]
            graficador_circulos(img_last, circles_last)
            graficador_circulos(img_current, circles_last)
            exit()

    D = np.sqrt((I_L-I_R)**2 + (J_L-J_R)**2) # Distancia entre ojalillos

    print(f"\n--------------- Resumen para '{file}.mp4' ---------------")
    print(f'Cargamos {N} frames de dimensión matricial {img_frame0.shape}:\n'
          f'{frames[0]}, {frames[1]}, {frames[2]}, ... , {frames[-1]}')
    print('Distancias entre ojalillos...')
    print(f'...promedio = {np.mean(D):.2f} px')
    print(f'...mínima = {np.amin(D):.2f} px (en el {frames[np.argmin(D)]})')
    print(f'...máxima = {np.amax(D):.2f} px (en el {frames[np.argmax(D)]})')

    ############### EXPORTO LA DATA COMO .CSV ###############
    data_dict = {
        'TIME': TIME, 'I_L': I_L, 'J_L': J_L, 'I_R': I_R, 'J_R': J_R, 
    }
    data = pd.DataFrame(data_dict)
    data_fname = f'{csv_dir}/{file}_data.csv'
    data.to_csv(data_fname, index=False)
    print(f'\n¡Bien! La evolución temporal de los ojalillos se guardó en {data_fname}')

def detector_arena(file):
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    frames_dir = f'{repo_dir}/Media/{today}/{file}(frames)'
    frames_ = os.listdir(frames_dir)
    frame0_fname = f'{frames_dir}/{frames_[0]}'
    img_frame0 = cv2.imread(frame0_fname)
    img_frame0_gray = cv2.cvtColor(img_frame0, cv2.COLOR_BGR2GRAY)

    radio_guess = int(input('Escriba su guess para el radio de la arena (en píxeles): '))

    circles_a = cv2.HoughCircles(
        img_frame0_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=22, 
        minRadius=radio_guess-5, maxRadius=radio_guess+5)

    print(f'\nCandidatos para la arena...\n{circles_a}')
    graficador_circulos(img_frame0, circles_a)

    prompt = input('¿Detectó la arena correctamente y la pintó de verde? [y/n]: ')

    if prompt.lower() == 'y':
        i_a, j_a = circles_a[0, 0, 1], circles_a[0, 0, 0] # Centro de la arena
        r_a = circles_a[0, 0, 2] # Radio de la arena

        ############### IMPORTO Y ACTUALIZO LA DATA ###############
        csv_dir = f'{repo_dir}/Data/{today}/{file}(csv)'
        data_fname = f'{csv_dir}/{file}_data.csv'
        data = pd.read_csv(data_fname)

        data['i_a'], data['j_a'] = i_a, j_a
        data['r_a'] = r_a

        data.to_csv(data_fname, index=False)
        print(f'\n¡Bien! El centro y el radio de la arena se guardaron en {data_fname}')

    elif prompt.lower() == 'n':
        print('\nOk, inténtelo de nuevo con otro guess para el radio de la arena.')
        detector_arena(file)

repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
camara_dir = r'/run/user/1000/gvfs/mtp:host=Sony_E5606_YT911BA6SB/Almacenamiento interno/DCIM/OpenCamera/Kilobot'
media_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots/Media/')

#x = os.path.expanduser('~/Escritorio/origen')
#mover_y_renombrar(origen=x, destino=media_dir)
#y = '65_72_3000_100_cali'
#tasa_prompt()
#data(y)
#detector_arena(y)
#confirmacion(file=y, frame='frame_82.jpg', th=10, e=16, ojal_guess=6)


prompt_init = input('¿Voy a ejecutar mover+tasa+data? [y/n] ')

if prompt_init.lower() == 'y':
    mover_y_renombrar(origen=camara_dir, destino=media_dir)
    #si camara dir es vacio no seguir!
    x = tasa_prompt()
    data(x)
    detector_arena(x)
    evolucion_temporal(f'{repo_dir}/Data/{today}', [x],VAR='ALPHA_UN')

else:
    prompt = input('¿Que quiere ejecutar? '
        '1: tasa, 2: data+arena, 3:arena, 4:tasa+data')
    archivo = input('Archivo para ejecutar: ')
    if prompt.lower() == '1':
        tasa_prompt()
    elif prompt.lower() == '2':
        data(archivo)
        detector_arena(archivo)
    elif prompt.lower() == '3':
        #detector_arena(archivo)
        evolucion_temporal(f'{repo_dir}/Data/{today}', [archivo], VAR='ALPHA_UN')
    elif prompt.lower() == '4':
        tasa_prompt()
        data(archivo)





