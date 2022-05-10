#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
// #include <cpuid.h>
#include <immintrin.h>


const unsigned seed0 = 0x20220509;   // a nice day today
const unsigned seedr = (unsigned)time(NULL);

using namespace std;

inline __m256i initPackedRandInts_8() {
    return _mm256_set_epi32(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand());
}

inline __m256i initPackedRandInts_8(const int *buf) {
    const int m = 0x7FFFFFFF;
    __m256i mask = _mm256_set_epi32(m, m, m, m, m, m, m, m);
    __m256i arr = _mm256_set_epi32(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
    return _mm256_and_si256(mask, arr);
}

inline bool genRandInts_16(const int *buf) {
    ifstream urandom("/dev/urandom");
    if (urandom.fail()) {
        std::cerr << "Error: Cannot open /dev/urandom\n";
        return false;
    }
    urandom.read((char *)buf, 16 * sizeof(int));
    urandom.close();
    return true;
}

inline void printPackedUnsigneds_8(const __m256i *arr) {
    unsigned *ptr = (unsigned *)arr;
    printf("%10u, %10u, %10u, %10u, %10u, %10u, %10u, %10u\n", 
            ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}

/* use std::rand() as initiator */
inline void rgen(__m256i &inpArr1, __m256i &inpArr2) {
    cout << "RAND_MAX = " << RAND_MAX << "\n" << endl;
    srand(seed0);
    inpArr1 = initPackedRandInts_8();
    inpArr2 = initPackedRandInts_8();
}

/* use /dev/urandom as initiator */
inline void rgen(__m256i &inpArr1, __m256i &inpArr2, int *buf) {
    inpArr1 = initPackedRandInts_8(buf);
    inpArr2 = initPackedRandInts_8(&buf[8]);
}


int main(int argc, char** argv) {
    __m256i inpArr1, inpArr2;

    // rgen(inpArr1, inpArr2);

    int buf[16];
    if(!genRandInts_16(buf)) { return 1; }
    rgen(inpArr1, inpArr2, buf);

    __m256i outArr = _mm256_add_epi32(inpArr1, inpArr2);

    cout << "input:" << endl;
    printPackedUnsigneds_8(&inpArr1);
    printPackedUnsigneds_8(&inpArr2);
    cout << "output:" << endl;
    printPackedUnsigneds_8(&outArr);

    return 0;
}