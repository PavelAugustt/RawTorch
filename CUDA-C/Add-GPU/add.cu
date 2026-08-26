#include <iostream>

__global__ void add(int a, int b, int *c)
{
    // i think *c just references th actual int variable cus we're passing the memory in parameter
    // this is called dereferencing apparently, we basically get the memory address and dereference to write a + b to that location
    *c = a + b;
    printf("a + b: %d", *c);
}

int main()
{
    int c;
    // this is very interesting to think about initially we define our variables c and * device c is basically where
    // the variable to store on device (gpu)
    int *device_c;

    // so when we call cudamalloc we basically say allocate a void pointer to this variable on device with a size of integer
    // so that when we add we can store our stuff in that spot
    cudaMalloc( (void**)&device_c, sizeof(int) );

    // here we just call the function 1 block 10 threads i think? thats an estimate then we just pass parameters with including the spot
    // we're storing c at on device memory

    // this is apparently a race condition, here when 10 threads run our kernel essentially theyre all writing to device_c
    // basically the same spot in other words for our result everything gets overwritten
    // pretty interesting right it is not noticeable because we're computing hte same value but if we had randomized it
    // well we would've seen the fireworks lol, i beleive we could bypass this by just passing a fuckton more memory addresses
    // or printing from the kernel directly instead of printing final result
    add<<<1, 10>>>(2, 7, device_c);

    // after this we basically just copy the data back from memory from device to basically the memory address of our int c
    // (int c lives on our host or cpu) we basically say yeah get that shit from device and store it where int c should be with
    // with a memory size of integer, idk what the 4th param symbolizes

    // ahh i see the 4th parameter basically tells what direction ur copying data from
    // a brainfart had led me to see the entire cudamemcpy() function to device to host, no no no
    // it copies both from host to device or vice versa the only difference being we havent specified the direction
    cudaMemcpy( &c, device_c, sizeof(int), cudaMemcpyDeviceToHost );

    // we print our stuff
    printf("c:%d", c);

    // and we finally free our shit on device!
    cudaFree(device_c);
    return 0;
}