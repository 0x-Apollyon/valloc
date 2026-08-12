# Vectorized(V)-Alloc

Valloc is an O(1) C++ slab allocator that manages memory using AVX2 intrinsics.

A standard glibc malloc call requires about 240-260 CPU cycles (62-82 nanoseconds), while Valloc completes a single allocation in 51-55 CPU cycles (18-20 nanoseconds).

## How does Valloc work ?

Valloc is a slab allocator, ie it takes a big slab of memory and divides it into small chunks which can be requested by other processes.

Valloc tracks memory state using a static 3 level bitmask tree. One root chunk tracks 256 branches, and each branch tracks 256 leaves (This operation in base256 is primarily a result of AVX2 registers being 4 byte-256 bit wide. If used on server hardware which supports AVX512 you can even use a bitmask tree which works in base512). A single bit corresponds to a single memory block and locating free memory requires exactly three steps, all of which run in constant time.

Valloc allocates memory using a segregated free list. It divides virtual memory into fixed size pools and when a program requests memory, Valloc calculates the next power of two (smallest power of two which is larger or equal) and allocates a memory block of that size.

## Why is it faster than standard malloc in glibc ?

Malloc is a general purpose allocator and it has to handle a lot of scenarios other than fixed size allocations, which make slab allocators inherently fast, so comparing the numbers directly is like comparing apples to oranges. 

However the use of vectorized instructions and making the allocator really, really branchless also provides speed gains.

If you wanted a more detailed answer: 

General purpose allocators like malloc scan memory sequentially to fit variable sized requests. Malloc does this by maintaining a linked list of free memory blocks which it can allocate, however that is really bad for speed.

Linked lists destroy locality, introduce pointer chasing and other cause other complicated I dont fully understand yet. The way malloc functions also introduces branches (which further cause pipeline stalls) and cache misses (a result of the destroyed locality).

Valloc avoids sequential scanning entirely (aswell as using no linked lists, linked lists in performance sensitive contexts is a recipe for disaster). The CPU loads 256 memory states into an AVX2 register and tests the entire block for empty space in one clock cycle using _mm256_testc_si256. 

If the register shows available space, Valloc finds the specific block by counting trailing zeros with __builtin_ctzll.

The hot path contains zero loops, and just to be safe and help the branch predictor and avoid pipeline stalls, __builtin_expect is used.

## Drawbacks

There is no free lunch, Valloc makes some tradeoffs which can cause some issues in certain contexts. 

- It operates on a single thread, that means multithreaded applications require locking (which basically nukes the speed)
- Valloc maps a really large chunk of virtual memory when it starts, primarily to avoid dynamic tree scaling which would introduce jitter aswell as making the allocator not constant time in all cases. Systems configured with strict virtual memory overcommit limits (vm.overcommit_memory=2) will reject the initial mapping.

## How to use

Valloc is primarily a fun sideproject I made as an experiment to see how fast I could make a memory allocator work, kind of inspired by John Lakos's talk at CPPCon on Memory Allocators.

If you want to use it to replicate my results, a test script is included. You can run it using the following commands (or use the Makefile).

```bash
cd tests
g++ -std=c++17 -O3 -mavx2 churn_test.cpp -o churn
./churn
```

If you want to use it in your own projects, copy the valloc.cpp file directly and use the VallocRouter class as you would use malloc, or put it in a header file, it works either way

Make sure to run it on a system which supports AVX2.

## AI Use Disclosure

Claude and Gemini wrote the churn test, helped me with the three level bitmap logic and also helped me out with the AVX2 instrinsics syntax.

