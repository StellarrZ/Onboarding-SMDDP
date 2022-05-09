#include <iostream>
#include <thread>
#include "mySemaphore.hpp"


using namespace std;

my::mySemaphore *semPtr1, *semPtr2;

void hello() {
    semPtr1->p();
    cout << "Hello " << flush;
    semPtr2->v();

    semPtr1->p();
    cout << "Hello " << flush;
    semPtr2->v();
}

void world() {
    semPtr2->p();
    cout << "World" << endl;
    semPtr1->v();
    
    semPtr2->p();
    cout << "World" << endl;
    semPtr1->v();
}

int main(int argc, char** argv) {
    semPtr1 = new my::mySemaphore(1);
    semPtr2 = new my::mySemaphore();

    thread tHello(hello);
    thread tWorld(world);
    
    tHello.join();
    tWorld.join();
    
    return 0;
}