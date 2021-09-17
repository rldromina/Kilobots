#include "kilolib.h"
// 20210602 Giros en sentido aleatorio. El tamaño del paso es proporcional a TIME

// Declaro constantes
#define EVEN 0
#define ODD 1
#define TIME 1000 // Tiempo de giro

// Declaro variables globales
int new_state = ODD;

void setup() {
}

// Función que define el movimiento
void set_led(int new_state) {
    if (new_state == EVEN) {
        set_color(RGB(1, 1, 1));
        }
    else if (new_state == ODD) {
        set_color(RGB(0, 0, 0));
        }
}

void loop() {
    int random_number = rand_hard(); // "Tiro la moneda"
    new_state = (random_number % 2);
    set_led(new_state);
    delay(TIME);
}

int main() {
    kilo_init();
    kilo_start(setup, loop);
    
    return 0;
}