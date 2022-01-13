#include "kilolib.h"
// Giros en sentido aleatorio. El tamaño del paso es proporcional a TIME2
// y el tiempo de reposo, a TIME1.

#define CCW 0
#define CW 1
#define STP 3
#define TIME1 100
#define TIME2 3000
//#define SEMILLA 5

int new_motion = CW;
uint8_t my_seed;


void setup() {
    my_seed = rand_hard();
    rand_seed(my_seed);
}

void set_motion(int new_motion) {
    if (new_motion == CCW) {
        spinup_motors();
        set_motors(kilo_turn_left, 0);
        //set_color(RGB(1, 1, 1));
        }
    else if (new_motion == CW) {
        spinup_motors();
        set_motors(0, kilo_turn_right);
        //set_color(RGB(0, 0, 0));
        }
    else if (new_motion == STP) {
        set_motors(0, 0);
        }
}

void loop() {
    int random_number = rand_soft();
    new_motion = (random_number % 2);
    set_motion(new_motion);
    delay(TIME2);
    set_motion(STP);
    delay(TIME1);
}

int main() {
    kilo_init();
    kilo_start(setup, loop);
    
    return 0;
}
