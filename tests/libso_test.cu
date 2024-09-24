#include "onebit_sparse_tensor.h"

int main(){
    void  * a;
    void  * b;
    void  * c;
    void  * d;
    void  * meta;

    int m=10;
    int n=10;
    int k=10;
    onebit_sparse_matmul(
        a,
        b,
        c,
        d,
        meta,
        m,
        n,
        k
    );
}