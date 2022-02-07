import os
import sys
import readline
import numpy as np
import pandas as pd
import cv2
import shutil
from tqdm import tqdm

from aux_tracker import evolucion_temporal, histograma_distancias, graficador_arena, ajuste

class fc:
    R = '\033[91m'
    G = '\033[92m'
    Y = '\033[93m'
    B = '\033[94m'
    M = '\033[95m'
    C = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

# Funciones secundarias

def prompt_valores():
    while True:
        try:
            left  = int(input('\n¿Cuál es el valor de LEFT?: '))
            right = int(input('¿Cuál es el valor de RIGHT?: '))
            step  = int(input('¿Cuál es el STEP (en ms)?: '))
            stop  = int(input('¿Cuál es el STOP (en ms)?: '))
        except ValueError:
            print(f'\n{fc.Y}Por favor, introduzca valores enteros{fc.END}')
            continue
        prompt = input('\n¿Son correctos estos valores? [Y/n]: ')
        if prompt.lower() in ['y', '']:
            break
        else:
            print(f'\n{fc.Y}Ok, inténtelo de nuevo{fc.END}')
            continue
    return left, right, step, stop

def graficador_circulos(img, circles):
    """Grafica los círculos 'circles' en la imagen 'img' de tres canales.
    Los 'circles' son de la forma [[[x1, y1, r1], [x2, y2, r2], ...]].
    """
    img_ = np.copy(img)
    circles_int = np.uint16(np.rint(circles))
    BGR = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for i, circle in enumerate(circles_int[0]):
        try:
            color = BGR[i]
        except:
            color = (0, 0, 255)
        x, y, r = circle
        cv2.circle(img_, center=(x, y), radius=r, color=color)
        cv2.circle(img_, center=(x, y), radius=0, color=color)
        cv2.putText(img_, f'{i}', org=(70*i, 50), fontFace=0,
                    fontScale=2, color=color, thickness=3)
    window_name = 'presione ESC para cerrar'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width=650, height=650)
    while True:
        cv2.imshow(window_name, img_)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def detector_arena(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    while True:
        try:
            r = int(input('\nGuess para el radio de la arena (en px): '))
        except ValueError:
            print(f'\n{fc.Y}Por favor, introduzca un valor entero{fc.END}')
            continue
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                                   param1=50, param2=22, minRadius=r-1, maxRadius=r+1)
        if circles is None:
            print(f'\n{fc.Y}No se detectó nada, inténtelo de nuevo{fc.END}')
            continue
        print(f'\nCandidatas para la arena:\n{circles}')
        graficador_circulos(img, circles)
        prompt = input(f'\n¿Alguna de estas arenas es la correcta? [n°/n]: ')
        try:
            i = int(prompt)
            arena = circles[0][i]
            break
        except:
            print(f'\n{fc.Y}Inténtelo de nuevo{fc.END}')
            continue
    return arena

def clickear(img):
    window_name = 'presione ESC para cerrar'
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
            clicks.append((x, y))
    while True:
        clicks = []
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width=650, height=650)
        cv2.setMouseCallback(window_name, callback)
        while True:
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        print(f'\nPuntos clickeados:\n{clicks}')
        prompt = input(f'\n¿Son correctos estos clicks? [Y/n]: ')
        if prompt.lower() in ['y', '']:
            break
        else:
            print(f'\n{fc.Y}Ok, inténtelo de nuevo{fc.END}')
            continue
    return clicks

def detector_ojal(img_gray, x, y, e, p1, p2, r):
    recorte = img_gray[y-e:y+e+1, x-e:x+e+1]
    circles = cv2.HoughCircles(recorte, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                               param1=p1, param2=p2, minRadius=r-1, maxRadius=r+1)
    return circles

def trackeador(video_path, x_l, y_l, x_r, y_r, e, p1, p2, ojal):
    cap = cv2.VideoCapture(video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    l, r = np.array([x_l, y_l]), np.array([x_r, y_r])

    X_L, Y_L = np.zeros(N), np.zeros(N)
    X_R, Y_R = np.zeros(N), np.zeros(N)
    TIME = np.zeros(N)
    for i in tqdm(range(N)):
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles_l = detector_ojal(frame_gray, *l, e, p1, p2, ojal)
        circles_r = detector_ojal(frame_gray, *r, e, p1, p2, ojal)
        try:
            l = l + np.array(circles_l[0][0][:2]) - e
            r = r + np.array(circles_r[0][0][:2]) - e
            # Guardo las posiciones de los ojalillos y el instante de tiempo
            X_L[i], Y_L[i] = l
            X_R[i], Y_R[i] = r
            TIME[i] = i / fps
            # Los centros para los próximos recortes (!)
            l, r = l.astype(int), r.astype(int)
        except:
            print(f'\n\n{fc.Y}Con p2={p2} no se detectó al menos '
                  f'un ojalillo en el frame n° {i}{fc.END}')
            cap.release()
            return None
    cap.release()
    return TIME, X_L, Y_L, X_R, Y_R

# Funciones principales

def mover_y_renombrar():
    origen = (r'/run/user/1000/gvfs/mtp:host=Sony_E5606_YT911BA6SB'
              r'/Almacenamiento interno/DCIM/OpenCamera/Kilobot')
    destino = os.path.expanduser(f'~/Escritorio/Repositorios/Kilobots/Media')

    origen_files = os.listdir(origen)

    if origen_files == []:
        sys.exit(f'{fc.R}\nNo hay nada en la cámara. Saliendo...{fc.END}')

    print(f'\nHay {len(origen_files)} archivo(s) en la cámara:')
    for file in origen_files:
        size = os.path.getsize(os.path.join(origen, file))
        print(f'{file} ({size/1e+6:.2f} MB)')

    for file in origen_files:
        prompt_mv = input(f'\n{fc.BOLD}¿Desea mover a {file}? [Y/n]: {fc.END}')
        if prompt_mv.lower() in ['y', '']:
            shutil.move(os.path.join(origen, file), destino)
            print(f'{fc.G}Movido{fc.END}')
            destino_files = os.listdir(destino)

            prompt_rn = input(f'{fc.BOLD}¿Desea renombrarlo? [Y/n]: {fc.END}')
            if prompt_rn.lower() in ['y', '']:
                new = input('Nuevo nombre (CON EXTENSIÓN): ').rstrip()
                while (new in destino_files) == True:
                    new = input('Ese nombre ya está en uso, pruebe con otro: ').rstrip()
                current_path = os.path.join(destino, file)
                new_path = os.path.join(destino, new)
                os.rename(current_path, new_path)
                print(f'{fc.G}Renombrado{fc.END}')

def TODO(video_path):
    ojal = 6 # Guess para el radio de los ojalillos (en px)
    e = 2 * ojal # 2*e = lado del cuadrado que voy a recortar (en px)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    segundos = frame_count / fps
    _, frame0 = cap.read()
    cap.release()

    print(f'\n{fc.B}Este video {os.path.basename(video_path)} fue grabado a '
          f'{fps:.2f} fps, tiene {frame_count} frames de dimensión matricial '
          f'{frame0.shape} y dura {segundos/60:.2f} minutos{fc.END}')

    print(f'\n{fc.BOLD}--- Introduzca la configuración ---{fc.END}')
    left, right, step, stop = prompt_valores()

    print(f'\n{fc.BOLD}--- Seleccione la arena ---{fc.END}')
    x_a, y_a, r_a = detector_arena(frame0)

    print(f'\n{fc.BOLD}--- Seleccione los ojalillos ---{fc.END}')
    clicks = clickear(frame0)

    umbrales = list(range(13, 9, -1))
    for p2 in umbrales:
        print(f'\n{fc.BOLD}--- Trackeando con un umbral p2={p2} ---{fc.END}\n')
        ret = trackeador(video_path, *clicks[0], *clicks[1], e, 50, p2, ojal)
        if ret is None and p2 != umbrales[-1]:
            continue
        elif ret is None and p2 == umbrales[-1]:
            sys.exit(f'\n{fc.R}Ninguno de los umbrales p2 en '
                     f'{umbrales} resultó. Saliendo...{fc.END}')
        else:
            TIME, X_L, Y_L, X_R, Y_R = ret
            break

    data = {
        'TIME': TIME, 'X_L': X_L, 'Y_L': Y_L, 'X_R': X_R, 'Y_R': Y_R,
        }
    meta = {
        'left': left, 'right': right, 'step': step, 'stop': stop,
        'frame_count': frame_count, 'fps': fps, 'segundos': segundos,
        'e': e, 'param2': p2, 'ojal': ojal,
        'x_a': x_a, 'y_a': y_a, 'r_a': r_a, 'radio': 150,
        'video_path': video_path,
        }
    return data, meta

def test_umbral(video_path):
    ojal = 6
    e = 2 * ojal

    cap = cv2.VideoCapture(video_path)
    i = int(input(f'\n{fc.BOLD}Número de frame para detectar: {fc.END}'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
    _, frame = cap.read()
    cap.release()

    print(f'\n{fc.BOLD}--- Seleccione los ojalillos ---{fc.END}')
    clicks = clickear(frame)
    l, r = np.array(clicks[0]), np.array(clicks[1])

    registro = []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while True:
        p2 = int(input(f'\n{fc.BOLD}¿Con qué umbral p2 probamos?: {fc.END}'))
        circles_l = detector_ojal(frame_gray, *l, e, 50, p2, ojal)
        circles_r = detector_ojal(frame_gray, *r, e, 50, p2, ojal)
        try:
            l_ = l + np.array(circles_l[0][0][:2]) - e
            r_ = r + np.array(circles_r[0][0][:2]) - e
            # Resultados de la detección
            circulos = [[[*l_, circles_l[0][0][2]], [*r_, circles_r[0][0][2]]]]
            registro.append((p2, circulos))
            print(f'\nCírculos detectados:\n{circulos}')
            graficador_circulos(frame, circulos)
        except:
            print(f'{fc.Y}\nCon p2={p2} no se detectó al menos '
                  f'un ojalillo. Inténtelo de nuevo{fc.END}')
            registro.append((p2, None))
            continue
        prompt = input('\n¿Quiere hacer otra prueba? [Y/n]: ')
        if prompt.lower() in ['y', '']:
            continue
        else:
            break
    
    print(f'\nSe hicieron {len(registro)} pruebas:\n{registro}')
    return registro

def main():
    repo_dir = os.path.expanduser('~/Escritorio/Repositorios/Kilobots')

    myDict = {
        '0': mover_y_renombrar,
        '1': TODO,
        '2': test_umbral,
        '3': evolucion_temporal,
        '4': histograma_distancias,
        '5': ajuste,
        '6': graficador_arena,
        }

    for k, v in myDict.items():
        print(f'{fc.M}{fc.BOLD}{k}: {v.__name__}{fc.END}')

    prompt = input(f'\n{fc.BOLD}¿Qué función quiere ejecutar?: {fc.END}')
    while prompt not in myDict.keys():
        print(f'\n{fc.Y}Por favor, seleccione sólo uno de esos números{fc.END}')
        prompt = input(f'\n¿Qué función quiere ejecutar?: ')

    if prompt == '0':
        mover_y_renombrar()
    elif prompt in list('12'):
        file = input(f'\n{fc.BOLD}¿Qué video quiere abrir?: {fc.END}')
        video_path = os.path.join(repo_dir, 'Media', f'{file}.mp4')
        if not os.path.isfile(video_path):
            sys.exit(f'\n{fc.R}{video_path} no existe{fc.END}')
        if prompt == '1':
            data, meta = myDict[prompt](video_path)
            # Creo la carpeta de CSVs
            csv_dir = os.path.join(repo_dir, 'Data', file)
            os.makedirs(csv_dir, exist_ok=True)
            # Guardo los dataframes de 'data' y 'metadatos' como CSVs
            data_df = pd.DataFrame(data)
            meta_df = pd.DataFrame(meta, index=[0])
            data_path = os.path.join(csv_dir, f'{file}_data.csv')
            meta_path = os.path.join(csv_dir, f'{file}_meta.csv')
            data_df.to_csv(data_path, index=False)
            meta_df.to_csv(meta_path, index=False)
            print(f'\n{fc.G}¡Listo! La data y los metadatos '
                  f'se guardaron en {csv_dir}{fc.END}')
        else:
            myDict[prompt](video_path)
    else:
        file = input(f'\n{fc.BOLD}¿Qué video quiere abrir?: {fc.END}')
        csv_dir = os.path.join(repo_dir, 'Data', file)
        if not os.path.exists(csv_dir):
            sys.exit(f'\n{fc.R}{csv_dir} no existe{fc.END}')
        myDict[prompt](csv_dir)

if __name__ == '__main__':
    main()
    # 1000_stop100_65_74
