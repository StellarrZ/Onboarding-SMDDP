#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
// #include <cpuid.h>
#include <immintrin.h>


const unsigned numArrs = 5;     // numArrs > 0
const unsigned numInts = numArrs << 3;
const unsigned numBytes = numInts << 2;
const unsigned seed0 = 0x20220509;   // a nice day today
const unsigned seedr = (unsigned)time(NULL);
const int m = 0x1FFFFFFF;       // (unsigned)m * numArrs <= 0xFFFFFFFF
const __m256i mask = _mm256_set_epi32(m, m, m, m, m, m, m, m);


using namespace std;

inline __m256i initPackedRandInts_8() {
    __m256i arr = _mm256_set_epi32(rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand()); 
    return _mm256_and_si256(mask, arr);
}

inline __m256i initPackedRandInts_8(const int *buf) {
    __m256i arr = _mm256_loadu_si256((const __m256i_u *)buf);
    // __m256i arr = _mm256_set_epi32(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
    return _mm256_and_si256(mask, arr);
}

inline bool genRandInts_N(const int *buf) {
    ifstream urandom("/dev/urandom");
    if (urandom.fail()) {
        std::cerr << "Error: Cannot open /dev/urandom\n";
        return false;
    }
    urandom.read((char *)buf, numBytes);
    urandom.close();
    return true;
}

inline void printPackedUnsigneds_8(const __m256i *arr) {
    unsigned *ptr = (unsigned *)arr;
    printf("  |%10u |%10u |%10u |%10u |%10u |%10u |%10u |%10u\n", 
            ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}

/* use std::rand() as initiator */
inline void rgen(__m256i *inpArrs) {
    cout << "RAND_MAX = |" << RAND_MAX << " & 0x" << hex << m 
         << " = | " << dec << (RAND_MAX & m) << "\n" << endl;
    srand(seed0);
    for (unsigned i = 0; i < numArrs; ++i) {
        inpArrs[i] = initPackedRandInts_8();
    }
}

/* use /dev/urandom as initiator */
inline void rgen(__m256i *inpArrs, const int *buf) {
    for (unsigned i = 0; i < numArrs; ++i) {
        inpArrs[i] = initPackedRandInts_8(&buf[i << 3]);
    }
}

inline __m256i calcAdds(const __m256i *inpArrs) {
    __m256i outArr = inpArrs[0];
    for (unsigned i = 1; i < numArrs; ++i) {
        outArr = _mm256_add_epi32(outArr, inpArrs[i]);
    }
    return outArr;
}


int main(int argc, char** argv) {
    __m256i inpArrs[numArrs];

    // rgen(inpArrs);

    int buf[numInts];
    if(!genRandInts_N(buf)) { return 1; }
    rgen(inpArrs, buf);

    __m256i outArr = calcAdds(inpArrs);

    cout << "input:" << endl;
    for (unsigned i = 0; i < numArrs; ++i) {
        printPackedUnsigneds_8(&inpArrs[i]);
    }
    cout << "output:" << endl;
    printPackedUnsigneds_8(&outArr);

    return 0;
}