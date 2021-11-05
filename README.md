# Kilobots

## Cómo instalar el software del controlador

0. Instalamos el IDE de Arduino desde https://www.arduino.cc/en/software: bajamos el .tar.xz, lo descomprimimos en el Escritorio y corrimos el install.sh. Dentro del IDE verificamos, en Herramientas, que la Placa (Arduino Nano), el Procesador (ver Nota 3) y el Puerto (ver Nota 2) sean los correctos. Para eso corrimos Archivos->Ejemplos->01.Basics->Blink y vimos que el IDE compiló y el LED del Arduino titiló.

1. Para el controlador, fuimos a https://github.com/acornejo/kilolib y descargamos como ZIP todos los archivos y los descomprimimos en el Escritorio. Se crea la carpeta “kilolib-master”.

2. Instalamos “avrdude” desde el escritorio con el comando “sudo apt-get install avr-libc gcc-avr avrdude”.

3. Desde “kilolib-master” ejecutamos “make ohc-arduino-16mhz”. Se crea una carpeta “build” con tres archivos: el más importante es “ohc-arduino-16mhz.hex”.

4. Desde “build” ejecutamos “sudo avrdude -v -patmega328p -carduino -P/dev/ttyUSB0 -b57600 -D -Uflash:w:ohc-arduino-16mhz.hex:i”. Ver qué puerto USB maneja el IDE de Arduino (ver Nota 2).

5. Desde https://github.com/acornejo/kilogui/releases/tag/v0.1 descargamos "kilogui_1.0-1_amd64.deb" y lo instalamos. Necesita Qt4 que se baja desde https://ubuntuhandbook.org/index.php/2020/07/install-qt4-ubuntu-20-04/. Se ejecuta la GUI como “sudo kilogui” (ver Nota 1) o se puede crear un lanzador en el Escritorio.

*Nota 1*: “sudo QT_X11_NO_MITSHM=1 kilogui” arregla posibles problemas de interfaz gráfica en gris debido a Qt4. (Compu de Romi con Ubuntu 18)

*Nota 2*: A VECES sirve “sudo chmod a+rw /dev/ttyUSB0” para acceder al puerto correcto. Ver en Linux qué puerto se usa, pero 'USB0' nos funciona.

*Nota 3*: Este Arduino Nano tiene el “ATmega328P (Old Bootloader)” (!)

*Nota 4*: Si el PPA falla al instalar Qt4 (nos pasó en Mint 20.3), fijarse las Fuentes de Software en el Gestor de Actualizaciones de Linux. Cambiar a otro mirror puede solucionarlo.

## Algunos comandos básicos de `git`

* Crear localmente una branch (clonando el main) y pasarme a ella:

    `git checkout -b tom`

* Creo remotamente (origin) esa branch local (tom) [Al parecer, desde cualquier branch funciona]

    `git push -u origin tom`

* Pasarme a la branch 'tom':

    `git checkout tom`

* Desde el workspace al index...

    `git add file`

    `git rm file`

    `git mv oldfile newfile`

* Traer todo del branch (tom) remoto (origin)

    `git pull origin tom`

* Traer un archivo (myfile.txt) desde una branch (main) [Después hay que commitear]

    `git checkout main myfile.txt`

* [Desde el main...] Elimino la branch (tom) remota (origin)

    `git push -d origin tom`

* [Desde el main...] Elimino la branch (tom) local

    `git branch -d tom`

---

[![Bichos](https://projects.iq.harvard.edu/files/styles/os_files_xxlarge/public/ssr/files/kilobot-stickers.png?m=1527790210&itok=_3fus5Gf)](https://www.youtube.com/watch?v=uc6f_2nPSX8)