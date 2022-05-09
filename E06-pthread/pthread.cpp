#include <iostream>
#include <thread>
#include "mySemaphore.hpp"


using namespace std;

my::mySemaphore *semPtr1, *semPtr2;

inline void print_Hello() {
    semPtr1->p();
    cout << "Hello " << flush;
    semPtr2->v();
}

inline void print_Wolrd() {
    semPtr2->p();
    cout << "World" << endl;
    semPtr1->v();
}

void func_Hello() {
    print_Hello();
    print_Hello();
}

void func_World() {
    print_Wolrd();
    print_Wolrd();
}

int main(int argc, char** argv) {
    semPtr1 = new my::mySemaphore(1);
    semPtr2 = new my::mySemaphore();

    thread tHello(func_Hello);
    thread tWorld(func_World);
    
    tHello.join();
    tWorld.join();
    
    return 0;
}