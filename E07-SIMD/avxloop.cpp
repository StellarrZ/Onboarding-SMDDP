#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
// #include <cpuid.h>
#include <immintrin.h>


const unsigned n = 100;       // n <= 0xFFFFFFF8
const unsigned numInts = n + (8 * bool(n % 8) - n % 8);     // padding for AVX
const unsigned numBytes = numInts << 2;


using namespace std;

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

inline void printArr(const int *arr) {
    cout << " ";
    for (unsigned i = 0; i < n; ++i) {
        cout << " |" << setw(11) << arr[i];
    }
    cout << endl;
}

/* prevent int32 overflow */
inline void shiftRightArthArr(int *arr) {
    for (unsigned i = 0; i + 7 < numInts; i += 8) {
        _mm256_storeu_si256((__m256i_u *)&arr[i], 
                            _mm256_srai_epi32(_mm256_loadu_si256((__m256i_u *)&arr[i]), 1));
    }
}

inline void calcAdd(const int *inpArr1, const int *inpArr2, int *outArr) {
    for (unsigned i = 0; i + 7 < numInts; i += 8) {
        _mm256_storeu_si256((__m256i_u *)&outArr[i], 
                            _mm256_add_epi32(_mm256_loadu_si256((__m256i_u *)&inpArr1[i]), 
                                             _mm256_loadu_si256((__m256i_u *)&inpArr2[i])));
    }
}


int main(int argc, char** argv) {
    int inpArr1[numInts], inpArr2[numInts];
    int outArr[numInts];

    if(!genRandInts_N(inpArr1)) { return 1; }
    else { shiftRightArthArr(inpArr1); }
    if(!genRandInts_N(inpArr2)) { return 1; }
    else { shiftRightArthArr(inpArr2); }

    calcAdd(inpArr1, inpArr2, outArr);

    cout << "input:" << endl;
    printArr(inpArr1);
    printArr(inpArr2);
    cout << "output:" << endl;
    printArr(outArr);

    return 0;
}