import os
import numpy as np
import pandas as pd
import cv2
import shutil
from tqdm import tqdm

# Funciones auxiliares:

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
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    frames_dir = f'{repo_dir}/Media/{file}(frames)'

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

    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    # Cargo la imagen de un frame en escala de grises
    img = cv2.imread(f'{frames_dir}/{frame}')

    i_l, j_l, i_r, j_r = clicks(img)
    i_l, j_l, i_r, j_r, circles_l, circles_r = detector_ojal(
        file, frame, th, e, ojal_guess, i_l, j_l, i_r, j_r)

    graficador_circulos(img, [[[j_l, i_l, ojal_guess]]])
    graficador_circulos(img, [[[j_r, i_r, ojal_guess]]])

def prompt():
    filename = input('Escriba el nombre del archivo: ').rstrip()
    #step = int(input('Escriba el STEP (en ms): '))
    #stop = int(input('Escriba el STOP (en ms): '))
    #left = int(input("Escriba la calibración LEFT: "))
    #right = int(input("Escriba la calibración RIGHT: "))

    prompt_ok = input(f'¿Los valores escritos arriba son correctos? [y/n]: ')

    if prompt_ok.lower() == 'y':
        return filename
    else:
        print('Ok, inténtelo de nuevo con los valores correctos...')
        prompt()

# Funciones principales:

def mover_y_renombrar():
    """Si el usuario lo desea, los archivos alojados en 'origen'
    son movidos a 'destino' y renombrados.
    """
    origen = r'/run/user/1000/gvfs/mtp:host=Sony_E5606_YT911BA6SB/Almacenamiento interno/DCIM/OpenCamera/Kilobot'
    destino = os.path.expanduser(f'~/Escritorio/Repositorios/Kilobots/Media')
    origen_files = os.listdir(origen)
    destino_files = os.listdir(destino)

    if origen_files == []:
        print(f'¡No hay archivos en la cámara!')
    else:
        print(f'Hay {len(origen_files)} archivo(s) en la cámara:'
              f'\n{origen_files}'
              f'\nEl destino es {destino}')
        for file in origen_files:
            prompt_mv = input(f'¿Desea mover {file}? [y/n]: ')
            if prompt_mv.lower() == 'y':
                shutil.move(f'{origen}/{file}', destino)
                print('---------- Movido ----------')
                prompt_rn = input(f'¿Desea renombrar a {file}? [y/n]: ')
                if prompt_rn.lower() == 'y':
                    newname = input('Nuevo nombre: ').rstrip()
                    ext = file.split('.')[1]
                    while (f'{newname}.{ext}' in destino_files) == True:
                        newname = input('¡Pruebe con otro nombre!: ').rstrip()
                    os.rename(f'{destino}/{file}', f'{destino}/{newname}.{ext}')
                    print('---------- Renombrado ----------')

def frames(file):
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')

    ############### CREO LA CARPETA DE FRAMES ###############4
    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    try:
        os.makedirs(frames_dir)
        print(f'Se creó {frames_dir}')
    except FileExistsError:
        print(f'Ya existe {frames_dir}')

    ############### CARGO EL VIDEO Y EXTRAIGO SUS FRAMES ###############
    video_fname = f'{repo_dir}/Media/{file}.mp4'
    cap = cv2.VideoCapture(video_fname)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segundos = frames_count / fps

    print(f"Este video '{file}.mp4', grabado a {fps:.1f} fps, "
          f"tiene {frames_count} frames y dura {segundos/60:.1f} minutos")

    i = 0
    while cap.isOpened():
        for i in tqdm(range(frames_count), desc="Extrayendo frames"):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(f'{frames_dir}/frame_{i}.jpg', frame)      
            i += 1
        cap.release()

    ############### CREO LA CARPETA DE CSV's ###############
    csv_dir = f'{repo_dir}/Data/{file}(csv)'
    try:
        os.makedirs(csv_dir)
        print(f'Se creó {csv_dir}')
    except FileExistsError:
        print(f'Ya existe {csv_dir}')

    ############### EXPORTO LOS METADATOS COMO .CSV ###############
    meta_dict = {
        #'step': [step], 'stop': [stop], 'left': [left], 'right': [right],
        'radio': [150], 'fps': [fps], 'frames_count': [frames_count], 'segundos': [segundos],
    }
    meta = pd.DataFrame(meta_dict)
    meta_fname = f'{csv_dir}/{file}_meta.csv'
    meta.to_csv(meta_fname, index=False)

def data(file):

    ############### PREPARO LOS FRAMES ###############
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    frames_ = os.listdir(frames_dir)
    frames = sorted(frames_, key=lambda x: int(x[6:-4])) # Porque 'frame_###.jpg'
    N = len(frames)

    ############### IMPORTO Y ACTUALIZO LOS METADATOS ###############
    csv_dir = f'{repo_dir}/Data/{file}(csv)'
    meta_fname = f'{csv_dir}/{file}_meta.csv'
    meta = pd.read_csv(meta_fname)

    meta['ojal_guess'] = ojal_guess = 6 # Guess para el radio de los ojalillos (en px)
    meta['e'] = e = 16 # 2*e = lado del cuadrado que voy a recortar (en px)
    meta['th'] = th = 10 # Umbral 'param2' del método HoughCircles

    meta.to_csv(meta_fname, index=False)
    print(f'Los metadatos asociados a esta medición son:'
          f'\n{meta}')

    ############### SELECCIONO LOS CENTROS ###############
    frame0_fname = f'{frames_dir}/{frames[0]}'
    img_frame0 = cv2.imread(frame0_fname)
    print(f'Seleccionemos los ojalillos. '
          f'Primero el izquierdo (más cerca del jumper) y luego el derecho.')
    i_l, j_l, i_r, j_r = clicks(img_frame0)

    ############### LEVANTO LA DATA ###############
    # Acá guardo la evolución temporal de las posiciones
    # de los centros de los ojalillos adheridos al Kilobot
    I_L, J_L = np.empty(N), np.empty(N)
    I_R, J_R = np.empty(N), np.empty(N)
    TIME = np.empty(N)

    for c in tqdm(range(N), desc="Trackeando posiciones"):
        fr = frames[c]
        i_l, j_l, i_r, j_r, circles_l, circles_r = detector_ojal(
            file, fr, th, e, ojal_guess, i_l, j_l, i_r, j_r)
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
    print(f'Cargamos {N} frames de dimensión matricial {img_frame0.shape}:'
          f'\n{frames[0]}, {frames[1]}, {frames[2]}, ... , {frames[-1]}')
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

def arena(file):
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')

    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    frames_ = os.listdir(frames_dir)
    frame0_fname = f'{frames_dir}/{frames_[0]}'
    img_frame0 = cv2.imread(frame0_fname)
    img_frame0_gray = cv2.cvtColor(img_frame0, cv2.COLOR_BGR2GRAY)

    radio_guess = int(input('Escriba su guess para el radio de la arena (en px): '))

    circles = cv2.HoughCircles(
        img_frame0_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=22, 
        minRadius=radio_guess-5, maxRadius=radio_guess+5)

    print(f'\nCandidatos para la arena...\n{circles}')
    graficador_circulos(img_frame0, circles)

    prompt = input('¿Detectó la arena correctamente y la pintó de verde? [y/n]: ')

    if prompt.lower() == 'y':
        i, j = circles[0, 0, 1], circles[0, 0, 0] # Centro de la arena
        r = circles[0, 0, 2] # Radio de la arena

        ############### IMPORTO Y ACTUALIZO LA DATA ###############
        csv_dir = f'{repo_dir}/Data/{file}(csv)'
        data_fname = f'{csv_dir}/{file}_data.csv'
        data = pd.read_csv(data_fname)

        data['i_a'], data['j_a'] = i, j
        data['r_a'] = r

        data.to_csv(data_fname, index=False)
        print(f'\n¡Bien! El centro y el radio de la arena se guardaron en {data_fname}')

    elif prompt.lower() == 'n':
        print('\nOk, inténtelo de nuevo con otro guess para el radio de la arena.')
        detector_arena(file)


frames('65_74_2000_100_1hr_c')
data('65_74_2000_100_1hr_c')
detector_arena('65_74_2000_100_1hr_c')
"""
repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
media_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots/Media')
camara_dir = r'/run/user/1000/gvfs/mtp:host=Sony_E5606_YT911BA6SB/Almacenamiento interno/DCIM/OpenCamera/Kilobot'
dct = {
    '1': mover_y_renombrar,
    '2': tasa,
    '3': data,
    '4': detector_arena,
}
newline = "\n"
prompt_init = input(f'¿Qué desea ejecutar?:\n{newline.join(f"{key}: {value}" for key, value in dct.items())}\n')
fn = prompt()
for i in prompt_init:
    if i != '1':
        dct[f'{i}'](fn)
"""

"""
if prompt_init.lower() == 'y':
    mover_y_renombrar(origen=camara_dir, destino=media_dir)
    #si camara dir es vacio no seguir!
    x = tasa_prompt()
    data(x)
    detector_arena(x)
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
        detector_arena(archivo)
    elif prompt.lower() == '4':
        tasa_prompt()
        data(archivo)
"""

# Cosas de aux_tracker:
#my_path = '/home/tom/Escritorio/Repositorios/Kilobots/Data'
#my_file = '65_74_2000_100_cali'
#graficador_arena(my_path, my_file)
#evolucion_temporal(my_path, [my_file], VAR='ALPHA_UN')
#videos = sorted([v for v in os.listdir(carpeta) if os.path.isdir(carpeta+v)])
#histograma(videos, carpeta=folder, VAR='ALPHA')
#temporal2(videos, carpeta=folder, VAR='ALPHA')
#curvas_calibracion(videos, carpeta=folder)