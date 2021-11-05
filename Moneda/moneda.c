#include "kilolib.h"
// LED prendido aleatoriamente. El tamaño del paso es proporcional a TIME

#define EVEN 0
#define ODD 1
#define TIME 1000
//#define SEMILLA 5

int new_state = ODD;
//uint8_t my_seed = SEMILLA;

void setup() {
//    rand_seed(my_seed);
}

void set_LED(int new_state) {
    if (new_state == EVEN) {
        set_color(RGB(1, 1, 1));
        }
    else if (new_state == ODD) {
        set_color(RGB(0, 0, 0));
        }
}

void loop() {
    int random_number = rand_soft();
    new_state = (random_number % 2);
    set_LED(new_state);
    delay(TIME);
}

int main() {
    kilo_init();
    kilo_start(setup, loop);
    
    return 0;
}
