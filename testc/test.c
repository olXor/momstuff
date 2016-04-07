#include <stdio.h>
#include <cmath>
#include <time.h>
#include <cstdlib>
#include </usr/include/fenv.h>
#include <algorithm>
#pragma STDC FENV_ACCESS ON

int main() {
    //feenableexcept(FE_ALL_EXCEPT);
    srand(time(NULL));

    printf("%f", fabs(-1));
}
