import os
import numpy as np
import pandas as pd
import cv2

REPO = os.path.expanduser('~/Escritorio/Repositorios/Kilobots') 
file = '65_73'

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

#################### PREPARO LOS FRAMES ####################
frames_dir = f'{REPO}/Media/{file}(frames)'
frames_ = os.listdir(frames_dir)
frames = sorted(frames_, key=lambda x: int(x[6:-4])) # Porque 'frame_###.jpg'
N = len(frames)

#################### IMPORTO Y ACTUALIZO LOS METADATOS ####################
csv_dir = f'{REPO}/Data/{file}(csv)'
meta_fname = f'{csv_dir}/{file}_meta.csv'
meta = pd.read_csv(meta_fname)

meta['radio_guess'] = radio_guess = 350 # Guess para el radio de la arena (en píxeles)
meta['ojal_guess'] = ojal_guess = 6 # Guess para el radio de los ojalillos (en píxeles)
meta['e'] = e = 16 # 2*e = lado del cuadrado que voy a recortar (en píxeles)
meta['th'] = th = 13 # Umbral 'param2' del método HoughCircles

meta.to_csv(meta_fname, index=False)
print(f'Los metadatos asociados a esta medición son:\n{meta}')

#################### DETECTO LA ARENA ####################
frame0_fname = f'{frames_dir}/{frames[0]}'
img_frame0 = cv2.imread(frame0_fname)
img_frame0_gray = cv2.cvtColor(img_frame0, cv2.COLOR_BGR2GRAY)

circles_a = cv2.HoughCircles(
    img_frame0_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=22, 
    minRadius=radio_guess-6, maxRadius=radio_guess+6)

print(f'\nCandidatos para la arena:\n{circles_a}')

graficador_circulos(img_frame0, circles_a)

i_a, j_a = circles_a[0, 0, 1], circles_a[0, 0, 0] # Centro de la arena
r_a = circles_a[0, 0, 2] # Radio de la arena

prompt = input('¿Detectó la arena correctamente y la pintó de verde? [y/n] ')

if prompt.lower() == 'y':
    print(f'¡Bien! Entonces seleccionemos los ojalillos.')
    print(f'Primero el izquierdo (más cerca del jumper) y luego el derecho.')

    #################### SELECCIONO LOS CENTROS ####################
    global clicks
    clicks = []

    def callback(event, x, y, flags, param):
        if event == 1:
            clicks.append((x,y))

    cv2.namedWindow('presione ESC para cerrar')
    cv2.setMouseCallback('presione ESC para cerrar', callback)

    while True:
        cv2.imshow('presione ESC para cerrar', img_frame0)    
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

    i_l, j_l = clicks[0][1], clicks[0][0] # Centro del ojalillo izquierdo
    i_r, j_r = clicks[1][1], clicks[1][0] # Centro del ojalillo derecho

    #################### LEVANTO LA DATA ####################
    # Acá guardo la evolución temporal de las posiciones
    # de los centros de los ojalillos adheridos al Kilobot
    I_L, J_L = np.empty(N), np.empty(N)
    I_R, J_R = np.empty(N), np.empty(N)
    TIME = np.empty(N)

    for c, fr in enumerate(frames):
        print(f'\n--------------- {fr} ({(c+1)/N:.0%}) ---------------')
        # Cargo la imagen de un frame en escala de grises
        img = cv2.imread(f'{frames_dir}/{fr}', 0)
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

        if (circles_l is None) or (circles_r is None):
            # Para identificar el error, muestro las imágenes de
            # los frames 'current' (el que falló) y 'last' (exitoso)
            img_last = cv2.imread(f'{frames_dir}/{frames[c-1]}')
            img_current = cv2.imread(f'{frames_dir}/{frames[c]}')
            circles_last = [[[j_l, i_l, ojal_guess], [j_r, i_r, ojal_guess]]]
            graficador_circulos(img_last, circles_last)
            graficador_circulos(img_current, circles_last)
            exit()

        # Posiciones LOCALES (i0,j0) de los ojalillos en esos recortes
        i0_l, j0_l = circles_l[0, 0, 1], circles_l[0, 0, 0]
        i0_r, j0_r = circles_r[0, 0, 1], circles_r[0, 0, 0]
        # Recupero las posiciones GLOBALES (i,j) a partir de las locales
        i_l, j_l = i_l + (i0_l-e), j_l + (j0_l-e)
        i_r, j_r = i_r + (i0_r-e), j_r + (j0_r-e)
        print(f'i_l = {i_l}, j_l = {j_l}        i_r = {i_r}, j_r = {j_r}')
        # Guardo las posiciones de los centros y el instante de tiempo
        I_L[c], J_L[c] = i_l, j_l
        I_R[c], J_R[c] = i_r, j_r
        TIME[c] = int(fr[6:-4]) / meta['fps']
        # Esto es necesario para poder recortar en la siguiente iteración (!)
        i_l, j_l = int(np.around(i_l)), int(np.around(j_l))
        i_r, j_r = int(np.around(i_r)), int(np.around(j_r))

    D = np.sqrt((I_L-I_R)**2 + (J_L-J_R)**2) # Distancia entre ojalillos

    print(f"\n--------------- Resumen para '{file}.mp4' ---------------")
    print(f'Cargamos {N} frames de dimensión matricial {img.shape}:\n'
          f'{frames[0]}, {frames[1]}, {frames[2]}, ... , {frames[-1]}')

    print('Distancias entre ojalillos...')
    print(f'...promedio = {np.mean(D):.2f} px')
    print(f'...mínima = {np.amin(D):.2f} px (en el {frames[np.argmin(D)]})')
    print(f'...máxima = {np.amax(D):.2f} px (en el {frames[np.argmax(D)]})')

    #################### EXPORTO LA DATA COMO .CSV ####################
    data_dict = {
        'TIME': TIME, 'I_L': I_L, 'J_L': J_L, 'I_R': I_R, 'J_R': J_R, 
        'i_a': i_a, 'j_a': j_a, 'r_a': r_a,
    }
    data = pd.DataFrame(data_dict)
    data_fname = f'{csv_dir}/{file}_data.csv'
    data.to_csv(data_fname, index=False)

elif prompt.lower() == 'n':
    print("Ok, inténtelo de nuevo con otro valor de 'radio_guess'.")
    exit()