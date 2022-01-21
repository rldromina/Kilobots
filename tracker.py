import os
import sys
import numpy as np
import pandas as pd
import cv2
import shutil
from tqdm import tqdm

class fc:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def prompt_valores():
    left  = int(input('\n¿Cuál es el valor de LEFT?: '))
    right = int(input('¿Cuál es el valor de RIGHT?: '))
    step  = int(input('¿Cuál es el STEP (en ms)?: '))
    stop  = int(input('¿Cuál es el STOP (en ms)?: '))
    prompt = input('\n¿Son correctos estos valores? [Y/n]: ')
    if prompt.lower() == 'y' or prompt == '':
        return left, right, step, stop
    else:
        return prompt_valores()

def arena(img):
    r = int(input(f'\n{fc.BOLD}Escriba su guess para el radio de la arena (en px): {fc.END}'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=50, param2=22, minRadius=r-5, maxRadius=r+5)
    if circles is None:
        print(f'\n{fc.YELLOW}No se detectó nada, inténtelo de nuevo{fc.END}')
        return arena(img)
    else:
        print(f'\nCandidatos para la arena:\n{circles}')
        graficador_circulos(img, circles)
        prompt = input('\n¿Detectó la arena y la pintó de verde? [Y/n]: ')
        if prompt.lower() == 'y' or prompt == '':
            i_a, j_a, r_a = circles[0,0,1], circles[0,0,0], circles[0,0,2]
            return i_a, j_a, r_a
        else:
            print(f'\n{fc.YELLOW}Ok, inténtelo de nuevo{fc.END}')
            return arena(img)

def graficador_circulos(img, circles):
    """Grafica en la imagen 'img' de tres canales, los centros y
    los perímetros de los círculos 'circles', estos son de la forma
    [[[x1,y1,r1], [x2,y2,r2], ...]]. Para una fácil identificación,
    el primer círculo lo dibuja de verde; todos los demás, de rojo.
    """
    img_ = np.copy(img)
    circles_int = np.uint16(np.around(circles))
    for c, v in enumerate(circles_int[0,:]):
        if c == 0:
            BGR = (0,255,0)
        else:
            BGR = (0,0,255)
        cv2.circle(img_, (v[0],v[1]), v[2], BGR, 1) # Perímetro
        cv2.circle(img_, (v[0],v[1]), 1, BGR, 1) # Centro
    cv2.namedWindow('circulos detectados', cv2.WINDOW_NORMAL)
    cv2.imshow('circulos detectados', img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clickear(img):
    print(f'\n{fc.BOLD}Clickee primero el ojalillo izquierdo '
          f'(más cerca del jumper) y luego, el derecho{fc.END}')
    clicks = []
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x,y))
    cv2.namedWindow('presione ESC para cerrar', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('presione ESC para cerrar', callback)
    while True:
        cv2.imshow('presione ESC para cerrar', img)    
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    print(f'\nPuntos clickeados:\n{clicks}')
    prompt = input(f'\n¿Son correctos estos clicks? [Y/n]: ')
    if prompt.lower() == 'y' or prompt == '':
        i_l, j_l = int(clicks[0][1]), int(clicks[0][0])
        i_r, j_r = int(clicks[1][1]), int(clicks[1][0])
        return i_l, j_l, i_r, j_r
    else:
        print(f'\n{fc.YELLOW}Ok, inténtelo de nuevo{fc.END}')
        return clickear(img)

def detector_ojal(img_gray, i, j, r, p1, p2):
    e = 2 * r # 2*e = lado del cuadrado que voy a recortar (en px)
    img_gray = img_gray[i-e:i+e+1, j-e:j+e+1]
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                               param1=p1, param2=p2, minRadius=r-1, maxRadius=r+1)
    return circles

# Funciones principales

def mover_y_renombrar():
    origen = (r'/run/user/1000/gvfs/mtp:host=Sony_E5606_YT911BA6SB'
              r'/Almacenamiento interno/DCIM/OpenCamera/Kilobot')
    destino = os.path.expanduser(f'~/Escritorio/Repositorios/Kilobots/Media')
    origen_files = os.listdir(origen)

    if origen_files == []:
        print(f'{fc.YELLOW}\nNo hay nada en la cámara{fc.END}')
    else:
        print(f'\nHay {len(origen_files)} archivo(s) en la cámara:\n{origen_files}')
        for file in origen_files:
            prompt_mv = input(f'\n¿Desea mover a {file}? [Y/n]: ')
            if prompt_mv.lower() == 'y' or prompt_mv == '':
                file_path = os.path.join(origen, file)
                shutil.move(file_path, destino)
                destino_files = os.listdir(destino)
                print(f'{fc.GREEN}\nMovido{fc.END}')
                prompt_rn = input(f'\n¿Desea renombrar a {file}? [Y/n]: ')
                if prompt_rn.lower() == 'y' or prompt_rn == '':
                    new = input('\nNuevo nombre (con extensión): ').rstrip()
                    new_path = os.path.join(destino, new)
                    while (new_path in destino_files) == True:
                        new = input('Pruebe con otro nombre: ').rstrip()
                        new_path = os.path.join(destino, new)
                    current_path = os.path.join(destino, file)
                    os.rename(current_path, new_path)
                    print(f'{fc.GREEN}\nRenombrado{fc.END}')

def TODO(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segundos = N / fps
    ret0, frame0 = cap.read()
    cap.release()

    print(f'\n{fc.BLUE}Este video fue grabado a {fps:.2f} fps, tiene {N} frames de '
          f'dimension matricial {frame0.shape} y dura {segundos/60:.2f} minutos{fc.END}')

    left, right, step, stop = prompt_valores()
    i_a, j_a, r_a = arena(frame0)
    i_l, j_l, i_r, j_r = clickear(frame0)
    ojal = 6 # Guess para el radio de los ojalillos (en px)
    p1, p2 = 50, 13 # Umbrales del método HoughCircles (más bajo, más sensible)

    I_L, J_L, I_R, J_R = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    TIME = np.zeros(N)

    print()
    cap = cv2.VideoCapture(video_path)
    for c in tqdm(range(N), desc=f'{fc.BOLD}Trackeando con p2={p2}{fc.END}'):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles_l = detector_ojal(frame_gray, i_l, j_l, ojal, p1, p2)
        circles_r = detector_ojal(frame_gray, i_r, j_r, ojal, p1, p2)
        if (circles_l is None) or (circles_r is None):
            print(f'\n\n{fc.RED}No se detectó al menos un ojalillo '
                  f'en el frame n° {c}. Saliendo...{fc.END}')
            sys.exit()
        # Posiciones LOCALES (i0,j0) de los ojalillos en esos recortes
        i0_l, j0_l = circles_l[0,0,1], circles_l[0,0,0]
        i0_r, j0_r = circles_r[0,0,1], circles_r[0,0,0]
        # Recupero las posiciones GLOBALES (i,j) a partir de las locales
        i_l, j_l = i_l + (i0_l-2*ojal), j_l + (j0_l-2*ojal)
        i_r, j_r = i_r + (i0_r-2*ojal), j_r + (j0_r-2*ojal)
        # Guardo las posiciones de los centros y el instante de tiempo
        I_L[c], J_L[c], I_R[c], J_R[c] = i_l, j_l, i_r, j_r
        TIME[c] = c / fps
        # Nuevos centros para los pŕoximos recortes
        i_l, j_l, i_r, j_r = int(i_l), int(j_l), int(i_r), int(j_r)
    cap.release()

    D = np.sqrt((I_L-I_R)**2 + (J_L-J_R)**2)
    print(f'\nDistancias entre ojalillos:\n'
          f'min = {np.amin(D):.2f} px (en {np.argmin(D)}) - '
          f'avg = {np.mean(D):.2f} px - '
          f'max = {np.amax(D):.2f} px (en {np.argmax(D)})')

    data = {'TIME': TIME, 'I_L': I_L, 'J_L': J_L, 'I_R': I_R, 'J_R': J_R}
    meta = {
        'left': left, 'right': right, 'step': step, 'stop': stop,
        'fps': fps, 'frame_count': N, 'segundos': segundos,
        'ojal': ojal, 'param1': p1, 'param2': p2,
        'radio': 150, 'i_a': i_a, 'j_a': j_a, 'r_a': r_a,
        }
    return data, meta

def test_umbral(video_path):
    c = int(input('\n¿Sobre qué número de frame quiere detectar?: '))
    cap = cv2.VideoCapture(video_path)
    cap.set(1, c)
    ret, frame = cap.read()
    cap.release()

    i_l, j_l, i_r, j_r = clickear(frame)
    ojal = 6
    p1, p2 = 50, int(input('\n¿Con qué umbral probamos?: '))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles_l = detector_ojal(frame_gray, i_l, j_l, ojal, p1, p2)
    circles_r = detector_ojal(frame_gray, i_r, j_r, ojal, p1, p2)
    while (circles_l is None) or (circles_r is None):
        print(f'{fc.YELLOW}\nCon p2={p2} no se detectó al menos un ojalillo. '
              f'Inténtelo de nuevo{fc.END}')
        p2 = int(input('\n¿Con qué nuevo umbral probamos?: '))
        circles_l = detector_ojal(frame_gray, i_l, j_l, ojal, p1, p2)
        circles_r = detector_ojal(frame_gray, i_r, j_r, ojal, p1, p2)
    # Posiciones LOCALES (i0,j0) [y radios] de los ojalillos en esos recortes
    i0_l, j0_l, r_l = circles_l[0,0,1], circles_l[0,0,0], circles_l[0,0,2]
    i0_r, j0_r, r_r = circles_r[0,0,1], circles_r[0,0,0], circles_r[0,0,2]
    # Recupero las posiciones GLOBALES (i,j) a partir de las locales
    i_l, j_l = i_l + (i0_l-2*ojal), j_l + (j0_l-2*ojal)
    i_r, j_r = i_r + (i0_r-2*ojal), j_r + (j0_r-2*ojal)

    circulos = [[[j_l, i_l, r_l], [j_r, i_r, r_r]]]
    print(f'\nCírculos detectados:\n{circulos}')
    graficador_circulos(frame, circulos)

def main():    
    print(f'\n0: mover_y_renombrar'
          f'\n1: TODO'
          f'\n2: test_umbral')

    prompt = input(f'\n{fc.BOLD}¿Qué función quiere ejecutar?: {fc.END}')

    if prompt not in ['0', '1', '2']:
        print(f'\n{fc.YELLOW}Por favor, seleccione solamente 1, 2 o 3{fc.END}')
        return main()

    if prompt == '0':
        mover_y_renombrar()
    else:
        file = input(f'\n{fc.CYAN}{fc.BOLD}¿Qué video quiere analizar?: {fc.END}').strip()
        repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')
        video_path = os.path.join(repo_dir, 'Media', f'{file}.mp4')
        if prompt == '1':
            data, meta = TODO(video_path)
            # Creo la carpeta de CSV's
            csv_dir = os.path.join(repo_dir, 'Data', file)
            os.makedirs(csv_dir, exist_ok=True)
            # Guardo el dataframe de 'data' como CSV
            data_df = pd.DataFrame(data)
            data_path = os.path.join(csv_dir, f'{file}_data.csv')
            data_df.to_csv(data_path, index=False)
            # Guardo el dataframe de 'metadatos' como CSV
            meta_df = pd.DataFrame(meta, index=[0])
            meta_path = os.path.join(csv_dir, f'{file}_meta.csv')
            meta_df.to_csv(meta_path, index=False)
            
            print(f'\n{fc.GREEN}¡Hecho! Se guardaron la data y '
                  f'los metadatos en {fc.BOLD}{csv_dir}{fc.END}')
        elif prompt == '2':
            test_umbral(video_path)

if __name__ == '__main__':
    main()
    # 1000_stop100_65_74