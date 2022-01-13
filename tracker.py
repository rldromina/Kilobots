import os
import sys
import numpy as np
import pandas as pd
import cv2
import shutil
from tqdm import tqdm

class fc:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

# Funciones auxiliares:

def detector_ojal(img0, i_l, j_l, i_r, j_r, e, ojal, th):
    i_l, j_l = int(i_l), int(j_l)
    i_r, j_r = int(i_r), int(j_r)
    # Recorto entornos cuadrados alrededor de los centros de los ojalillos
    img_l = img0[i_l-e:i_l+e+1, j_l-e:j_l+e+1]
    img_r = img0[i_r-e:i_r+e+1, j_r-e:j_r+e+1]
    # Busco los ojalillos en esos recortes
    circles_l = cv2.HoughCircles(img_l, cv2.HOUGH_GRADIENT, 1, 20, param1=50, 
        param2=th, minRadius=ojal-1, maxRadius=ojal+1)
    circles_r = cv2.HoughCircles(img_r, cv2.HOUGH_GRADIENT, 1, 20, param1=50, 
        param2=th, minRadius=ojal-1, maxRadius=ojal+1)
    return circles_l, circles_r

def graficador_circulos(img, circles):
    """Grafica en la imagen 'img' de tres canales, los centros y
    los perímetros de los círculos 'circles', estos son de la forma
    [[[x1,y1,r1], [x2,y2,r2], ...]]. Para una fácil identificación,
    el primer círculo lo dibuja de verde; todos los demás, de rojo.
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

def clickear(img):
    global clicks
    clicks = []
    def callback(event, x, y, flags, param):
        if event == 1:
            clicks.append((x,y))
    cv2.namedWindow('presione ESC para cerrar')
    cv2.setMouseCallback('presione ESC para cerrar', callback)
    while True:
        cv2.imshow('presione ESC para cerrar', img)    
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    i_l, j_l = clicks[0][1], clicks[0][0] # Centro del ojalillo izquierdo
    i_r, j_r = clicks[1][1], clicks[1][0] # Centro del ojalillo derecho

    prompt = input('¿Clickeo correctamente? [y/n]: ')
    print('')
    if prompt.lower() == 'y':
        return i_l, j_l, i_r, j_r
    else:
        return clickear(img)

def prompt_valores():
    left  = input('¿Cuál es el valor de LEFT?: ')
    right = input('¿Cuál es el valor de RIGHT?: ')
    step  = input('¿Cuál es el STEP (en ms)?: ')
    stop  = input('¿Cuál es el STOP (en ms)?: ')

    prompt = input('\n¿Son correctos esos valores? [y/n]: ')
    print('')
    if prompt.lower() == 'y':
        return left, right, step, stop
    else:
        return prompt_valores()

# Funciones principales:

def mover_y_renombrar():
    print(f'\n{fc.BLUE}Vamos a mover y renombrar lo que hay en la cámara{fc.END}\n')

    origen = (r'/run/user/1000/gvfs/mtp:host=Sony_E5606_YT911BA6SB'
              r'/Almacenamiento interno/DCIM/OpenCamera/Kilobot')
    destino = os.path.expanduser(f'~/Escritorio/Repositorios/Kilobots/Media')
    origen_files = os.listdir(origen)

    if origen_files == []:
        print(f'{fc.YELLOW}No hay nada en la cámara{fc.END}')
    else:
        print(f'Hay {len(origen_files)} archivo(s) en la cámara:\n{origen_files}')
        for file in origen_files:
            prompt_mv = input(f'\n¿Desea mover a {file}? [y/n]: ')
            if prompt_mv.lower() == 'y':
                shutil.move(f'{origen}/{file}', destino)
                destino_files = os.listdir(destino)
                print(f'{fc.GREEN}Movido{fc.END}')

                prompt_rn = input(f'¿Desea renombrar a {file}? [y/n]: ')
                if prompt_rn.lower() == 'y':
                    newname = input('Nuevo nombre: ').rstrip()
                    ext = file.split('.')[1]
                    while (f'{newname}.{ext}' in destino_files) == True:
                        newname = input('Pruebe con otro nombre: ').rstrip()
                    os.rename(f'{destino}/{file}', f'{destino}/{newname}.{ext}')
                    print(f'{fc.GREEN}Renombrado{fc.END}')

def frames(file):
    print(f'\n{fc.BLUE}Vamos a extraer todos los frames de {fc.BOLD}{file}.mp4{fc.END}\n')

    left, right, step, stop = prompt_valores()

    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')

    ############### CREO LA CARPETA DE FRAMES ###############
    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    try:
        os.makedirs(frames_dir)
        print(f'La carpeta de frames {fc.BOLD}{frames_dir}{fc.END} se creó')
    except FileExistsError:
        print(f'La carpeta de frames {fc.BOLD}{frames_dir}{fc.END} ya existía')

    ############### CARGO EL VIDEO Y EXTRAIGO SUS FRAMES ###############
    video_filename = f'{repo_dir}/Media/{file}.mp4'
    cap = cv2.VideoCapture(video_filename)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segundos = frame_count / fps

    print(f'\nEste video, grabado a {fps:.2f} fps, tiene '
          f'{frame_count} frames y dura {segundos/60:.2f} minutos\n')

    while cap.isOpened():
        for i in tqdm(range(frame_count), desc='Extrayendo frames'):
            ret, frame = cap.read()
            if ret == False:
                print(f'{fc.RED}La iteración del frame nro. {i} falló. Saliendo...{fc.END}')
                sys.exit()
            cv2.imwrite(f'{frames_dir}/frame_{i}.jpg', frame)
        cap.release()

    ############### CREO LA CARPETA DE CSV's ###############
    csv_dir = f'{repo_dir}/Data/{file}(csv)'
    try:
        os.makedirs(csv_dir)
        print(f'\nLa carpeta de CSV {fc.BOLD}{csv_dir}{fc.END} se creó')
    except FileExistsError:
        print(f'\nLa carpeta de CSV {fc.BOLD}{csv_dir}{fc.END} ya existía')

    ############### EXPORTO LOS METADATOS COMO .CSV ###############
    meta_dict = {
        'left': [left], 'right': [right], 'step': [step], 'stop': [stop],
        'fps': [fps], 'frame_count': [frame_count], 'segundos': [segundos],
        'radio': [150],
    }
    meta = pd.DataFrame(meta_dict)
    meta_filename = f'{csv_dir}/{file}_meta.csv'
    meta.to_csv(meta_filename, index=False)

    print(f'{fc.GREEN}\n¡Bien! Se extrajeron todos los frames '
          f'y los metadatos se guardaron como {fc.BOLD}{meta_filename}{fc.END}')

def data(file):
    print(f'\n{fc.BLUE}Vamos a trackear la trayectoria en {fc.BOLD}{file}.mp4{fc.END}\n')

    e = 16 # 2*e = lado del cuadrado que voy a recortar (en px)
    ojal = 6 # Guess para el radio de los ojalillos (en px)
    th = 10 # Umbral 'param2' del método HoughCircles

    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')

    ############### LISTA DE FRAMES ###############
    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    frames_ = os.listdir(frames_dir)
    frames = sorted(frames_, key=lambda x: int(x[6:-4])) # Porque 'frame_###.jpg'
    N = len(frames)

    ############### IMPORTO LOS METADATOS ###############
    csv_dir = f'{repo_dir}/Data/{file}(csv)'
    meta_filename = f'{csv_dir}/{file}_meta.csv'
    meta = pd.read_csv(meta_filename)

    ############### CLICKEO LOS CENTROS ###############
    frame_filename = f'{frames_dir}/{frames[0]}'
    img = cv2.imread(frame_filename)
    print('Seleccione primero el ojalillo izquierdo (más cerca del jumper)\n')
    i_l, j_l, i_r, j_r = clickear(img)

    ############### LEVANTO LA DATA ###############
    # Acá guardo la evolución temporal de las posiciones
    # de los centros de los ojalillos adheridos al Kilobot
    I_L, J_L = np.zeros(N), np.zeros(N)
    I_R, J_R = np.zeros(N), np.zeros(N)
    TIME = np.zeros(N)

    for c in tqdm(range(N), desc='Trackeando posiciones'):
        fr = frames[c]
        img0 = cv2.imread(f'{frames_dir}/{fr}', 0)
        circles_l, circles_r = detector_ojal(img0, i_l, j_l, i_r, j_r, e, ojal, th)
        if (circles_l is None) or (circles_r is None):
            print(f'\n\n{fc.RED}Con e={e}, ojal={ojal} y th={th} no se '
                  f'detectó al menos un ojalillo en {fc.BOLD}{fr}{fc.END}'
                  f'{fc.RED}. Saliendo...{fc.END}')
            sys.exit()
        # Posiciones LOCALES (i0,j0) de los ojalillos en esos recortes
        i0_l, j0_l = circles_l[0, 0, 1], circles_l[0, 0, 0]
        i0_r, j0_r = circles_r[0, 0, 1], circles_r[0, 0, 0]
        # Recupero las posiciones GLOBALES (i,j) a partir de las locales
        i_l, j_l = i_l + (i0_l-e), j_l + (j0_l-e)
        i_r, j_r = i_r + (i0_r-e), j_r + (j0_r-e)
        # Guardo las posiciones de los centros y el instante de tiempo
        I_L[c], J_L[c] = i_l, j_l
        I_R[c], J_R[c] = i_r, j_r
        TIME[c] = int(fr[6:-4]) / meta['fps']

    D = np.sqrt((I_L-I_R)**2 + (J_L-J_R)**2) # Distancia entre ojalillos

    print(f'\nCargamos {N} frames de dimensión matricial {img0.shape}:'
          f'\n{frames[0]}, {frames[1]}, {frames[2]}, ... , {frames[-1]}')
    print('\nDistancias entre ojalillos:'
          f'\npromedio = {np.mean(D):.2f} px'
          f'\nmínima = {np.amin(D):.2f} px (en {frames[np.argmin(D)]})'
          f'\nmáxima = {np.amax(D):.2f} px (en {frames[np.argmax(D)]})')

    ############### ACTUALIZO LOS METADATOS ###############
    meta['e'], meta['ojal'], meta['th'] = e, ojal, th
    meta.to_csv(meta_filename, index=False)

    ############### EXPORTO LA DATA COMO .CSV ###############
    data_dict = {
        'TIME': TIME, 'I_L': I_L, 'J_L': J_L, 'I_R': I_R, 'J_R': J_R, 
    }
    data = pd.DataFrame(data_dict)
    data_filename = f'{csv_dir}/{file}_data.csv'
    data.to_csv(data_filename, index=False)

    print(f'{fc.GREEN}\n¡Bien! Se actualizaron los metadatos '
          f'y la data se guardó como {fc.BOLD}{data_filename}{fc.END}')

def arena(file):
    print(f'\n{fc.BLUE}Vamos a detectar la arena en {fc.BOLD}{file}.mp4{fc.END}\n')

    radio = int(input('Escriba su guess para el radio de la arena (en px): '))

    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    frames_ = os.listdir(frames_dir)
    frame_filename = f'{frames_dir}/{frames_[0]}'

    img = cv2.imread(frame_filename)
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img0, cv2.HOUGH_GRADIENT, 1, 20, 
        param1=50, param2=22, minRadius=radio-5, maxRadius=radio+5)

    if circles is not None:
        print(f'\nCandidatos para la arena:\n{circles}')
        graficador_circulos(img, circles)
        prompt = input('\n¿Dibujó correctamente la arena de verde? [y/n]: ')
        if prompt.lower() == 'y':
            ############### IMPORTO Y ACTUALIZO LA DATA ###############
            csv_dir = f'{repo_dir}/Data/{file}(csv)'
            data_filename = f'{csv_dir}/{file}_data.csv'
            data = pd.read_csv(data_filename)

            data['i_a'], data['j_a'] = circles[0, 0, 1], circles[0, 0, 0]
            data['r_a'] = circles[0, 0, 2]
            data.to_csv(data_filename, index=False)

            print(f'\n{fc.GREEN}¡Bien! Se actualizó la data con '
                  f'el centro y el radio de la arena{fc.END}')
        else:
            print(f'\n{fc.YELLOW}Ok, inténtelo nuevamente...{fc.END}')
            return arena(file)
    else:
        print(f'\n{fc.RED}No se detectó nada con ese guess. Saliendo...{fc.END}')
        sys,exit()

def prueba_th(file):
    print(f'\n{fc.BLUE}Vamos a detectar ojalillos en {fc.BOLD}{file}.mp4{fc.END}\n')

    i = input('¿Sobre qué número de frame quiere detectar?: ')

    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
    frames_dir = f'{repo_dir}/Media/{file}(frames)'
    img = cv2.imread(f'{frames_dir}/frame_{i}.jpg')
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('\nSeleccione primero el ojalillo izquierdo (más cerca del jumper)\n')
    i_l, j_l, i_r, j_r = clickear(img)

    e = 16
    ojal = 6
    th = int(input('¿Con qué umbral probamos?: '))

    circles_l, circles_r = detector_ojal(img0, i_l, j_l, i_r, j_r, e, ojal, th)
    while (circles_l is None) or (circles_r is None):
        print(f'{fc.RED}\nNo se detectó al menos un ojalillo. '
              f'Inténtelo nuevamente...\n{fc.END}')
        th = int(input('¿Con qué nuevo umbral probamos?: '))
        circles_l, circles_r = detector_ojal(img0, i_l, j_l, i_r, j_r, e, ojal, th)

    # Posiciones LOCALES (i0,j0) de los ojalillos en esos recortes
    i0_l, j0_l = circles_l[0, 0, 1], circles_l[0, 0, 0]
    i0_r, j0_r = circles_r[0, 0, 1], circles_r[0, 0, 0]
    # Radios
    r_l, r_r = circles_l[0, 0, 2], circles_r[0, 0, 2]
    # Recupero las posiciones GLOBALES (i,j) a partir de las locales
    i_l, j_l = i_l + (i0_l-e), j_l + (j0_l-e)
    i_r, j_r = i_r + (i0_r-e), j_r + (j0_r-e)
    graficador_circulos(img, [[[j_l, i_l, r_l], [j_r, i_r, r_r]]])

def main():
    funct = {
        '0': mover_y_renombrar,
        '1': frames,
        '2': data,
        '3': arena,
        '9': prueba_th,
    }

    for k, v in funct.items(): print(f'{k} : {v.__name__}')
    prompt = input('\n¿Qué funciones quiere ejecutar?: ')

    check = False
    for i in prompt:
        if i == '0':
            funct[i]
        elif (i != '0') and (check == False):
            file = input('\n¿Qué video quiere analizar?: ')
            funct[i](file)
            check = True
        else:
            funct[i](file)

if __name__ == "__main__":
    main()