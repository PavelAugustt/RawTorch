#include <iostream>

# define N  100000

void add(int *a, int *b, int *c)
{
    int tid = 0;

    while(tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main(void)
{
    int a[N], b[N], c[N];

    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    add(a, b, c);

    for(int i = 0; i < N; i++)
    {
        printf("a @ %d", a[i]);
        printf("b @ %d", b[i]);
        printf("c @ %d", c[i]);

    }
    return 0;
}