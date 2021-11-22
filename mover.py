import shutil, os
from pathlib import Path

#origen = r'/run/user/1000/gvfs/mtp:host=Sony_E5606_YT911BA6SB/Almacenamiento interno/DCIM/OpenCamera/Kilobot'


origen = os.path.expanduser('~/Escritorio/origen')
destino = os.path.expanduser('~/Escritorio/destino')
files = os.listdir(origen)

for f in files:
    shutil.move(f'{origen}/{f}', destino)
    #print(f)

print('Estos son los videos encontrados en la cámara:', files)

prompt = input('¿Querés cambiarle el nombre a los videos? [y/n] ')

if prompt.lower() == 'y':
    for current in files:
        print(f'Nombre actual: {current}')
        extension = current.split('.')[1]
        newname = input("Nuevo nombre: ")
        os.rename(f'{destino}/{current}', f'{destino}/{newname}.{extension}')
        print('----------')