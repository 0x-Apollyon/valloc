#include <immintrin.h>
#include <cstdint>
#include <sys/mman.h>
#include <cstddef>
#include <new>
#include <algorithm>
#include <cerrno>
#include <iostream>
#include <cstring>

#define POOL_CAPACITY 16777216ULL //256*256*256
#define ARCH_SIZE 64 //sizeof(size_t)

inline int find_first_free_bit(const uint64_t* chunk){
    __m256i reg = _mm256_load_si256(reinterpret_cast<const __m256i*>(chunk));
    __m256i all_ones = _mm256_set1_epi64x(-1LL);

    if (_mm256_testc_si256(reg, all_ones)){
        return -1;
    } else {
        int idx;
        #pragma GCC unroll 4
        for (int i = 0; i < 4; i++){
            const uint64_t inverted = ~chunk[i];
            if (inverted){
                idx = __builtin_ctzll(inverted);
                return i*64 + idx;
            }
        }
        return -1;
    }
}

class Valloc {
    private:
        size_t block_size;
        size_t block_size_power;
    
        alignas(32) uint64_t L2[4] = {0}; //root
        uint64_t* L1;
        uint64_t* L0; //leaves

        uint8_t* memory_pool;
        uint8_t* memory_pool_end;

    public:
        Valloc(size_t size) {
            block_size = size;
            block_size_power = ARCH_SIZE - __builtin_clzll(size) - 1;
            //1024 = 256*4
            //262144 = 65536 = 256*256

            L1 = (uint64_t*) mmap(NULL, 1024 * sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            L0 = (uint64_t*) mmap(NULL, 262144 * sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            
            //1 gb pool
            size_t pool_size = POOL_CAPACITY * block_size;
            memory_pool = (uint8_t*) mmap(NULL, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            memory_pool_end = memory_pool + (POOL_CAPACITY << block_size_power);

            if (L1 == MAP_FAILED || L0 == MAP_FAILED || memory_pool == MAP_FAILED){
                std::cerr << "mmap failed for block_size: " << block_size << "\n";
                std::cerr << "errno: " << errno << " (" << strerror(errno) << ")\n";

                if (L1 != MAP_FAILED){
                    munmap(L1, 1024 * sizeof(uint64_t));
                }

                if (L0 != MAP_FAILED){
                    munmap(L0, 262144 * sizeof(uint64_t));
                }

                if (memory_pool != MAP_FAILED){
                    munmap(memory_pool, pool_size);
                }
                
                throw std::bad_alloc{};
            }
                
        }

        Valloc(const Valloc&) = delete;
        Valloc& operator=(const Valloc&) = delete; 
        //no copy constructors
        //no nakal, no akal needed to solve double free :)

        void* allocate() {
            int l2_bit = find_first_free_bit(L2);
            if (l2_bit == -1){
                return nullptr;
            }

        
            int l1_bit = find_first_free_bit(&L1[l2_bit << 2]);
            
            int l0_bit = find_first_free_bit(&L0[((l2_bit << 8) | l1_bit) << 2]);

            uint64_t global_block_index = (l2_bit << 16) | (l1_bit << 8) | l0_bit;
            uint64_t l0_index_int = (global_block_index >> 6);
            uint64_t l0_inside_int = (global_block_index & 63);

            L0[l0_index_int] = L0[l0_index_int] | (1ULL << l0_inside_int);
            
            if (__builtin_expect(L0[l0_index_int] == ~0ULL , 0)) {
                __m256i all_ones = _mm256_set1_epi64x(-1LL);
                __m256i reg = _mm256_load_si256(reinterpret_cast<const __m256i*>(&L0[(global_block_index >> 8) << 2]));

                if (_mm256_testc_si256(reg, all_ones)){
                    uint64_t l1_index = global_block_index >> 8;
                    L1[l1_index >> 6] = L1[l1_index >> 6] | (1ULL << (l1_index & 63));

                    if (__builtin_expect(L1[l1_index >> 6] == ~0ULL, 0)){
                        reg = _mm256_load_si256(reinterpret_cast<const __m256i*>(&L1[l2_bit << 2]));

                        if (_mm256_testc_si256(reg, all_ones)){
                            L2[l2_bit >> 6] = L2[l2_bit >> 6] | (1ULL << (l2_bit & 63));
                        }
                    }
                }
            }

            return memory_pool + (global_block_index << block_size_power);
        }

        int free(void * ptr){
            if (ptr == nullptr){
                return 0;
            }

            uint8_t* p = static_cast<uint8_t*>(ptr);
            size_t max_pool_size = (POOL_CAPACITY << block_size_power);

            if (p < memory_pool || p >= memory_pool_end) {
                return EFAULT; 
            }

            size_t byte_offset = p - memory_pool;
            if ((byte_offset & (block_size - 1)) != 0) {
                return EINVAL;
            }

            uint64_t global_block_index = (byte_offset >> block_size_power);
            uint64_t l0_index_int = global_block_index >> 6;
            uint64_t l0_inside_int = global_block_index & 63;

            if (((L0[l0_index_int] >> l0_inside_int) & 1ULL) == 0) {
                return EALREADY; 
            }

            L0[l0_index_int] = L0[l0_index_int] & ~(1ULL << l0_inside_int);

            uint64_t l1_index = global_block_index >> 8;
            L1[l1_index >> 6] = L1[l1_index >> 6] & ~(1ULL << (l1_index & 63));

            uint64_t l2_index = global_block_index >> 16;
            L2[l2_index >> 6] = L2[l2_index >> 6] & ~(1ULL << (l2_index & 63));

            return 0;
        }

        ~Valloc() {
            if (L1 && L1 != MAP_FAILED){
                munmap(L1, 1024 * sizeof(uint64_t));
            }

            if (L0 && L0 != MAP_FAILED){
                munmap(L0, 262144 * sizeof(uint64_t));
            }

            if (memory_pool && memory_pool != MAP_FAILED){
                munmap(memory_pool, POOL_CAPACITY * block_size);
            }
        }

        uint8_t* get_pool_start(){
            return memory_pool;
        }

        uint8_t* get_pool_end(){
            return memory_pool_end;
        }
};

class VallocRouter{
    private:
        struct PoolBounds {
            uint8_t* start;
            uint8_t* end;
            int pool_index;
        };
        
        PoolBounds bounds[9];
        Valloc pools[9]; ; //16, 32, 64, 128, 256, 512, 1024, 2048, 4096

    public:
        VallocRouter() : pools{Valloc(16), Valloc(32), Valloc(64), Valloc(128),Valloc(256), Valloc(512), Valloc(1024), Valloc(2048), Valloc(4096)} {
            #pragma GCC unroll 9
            for (int i = 0; i < 9; ++i) {
                bounds[i].start = pools[i].get_pool_start();
                bounds[i].end = pools[i].get_pool_end();
                bounds[i].pool_index = i;
            }

            std::sort(bounds, bounds + 9, 
            [](const PoolBounds& a, const PoolBounds& b){return a.start < b.start;});
        }

        ~VallocRouter() = default; 

        void* malloc(size_t size) {
            if (size == 0 || size > 4096){
                return nullptr;
            }

            size_t adjusted_size = std::max(size, (size_t) 9);

            uint64_t idx = ARCH_SIZE - __builtin_clzll(adjusted_size - 1);
            return pools[idx - 4].allocate();
        }

        int free(void* ptr) {
            if (ptr == nullptr){
                return 0;
            }
            
            uint8_t* p = reinterpret_cast<uint8_t*>(ptr);

            int i = (p >= bounds[0].start)
                + (p >= bounds[1].start)
                + (p >= bounds[2].start)
                + (p >= bounds[3].start)
                + (p >= bounds[4].start)
                + (p >= bounds[5].start)
                + (p >= bounds[6].start)
                + (p >= bounds[7].start)
                + (p >= bounds[8].start)
                - 1;

            if ((unsigned)i < 9 && p < bounds[i].end){
                return pools[bounds[i].pool_index].free(ptr);
            }
                
            return EFAULT;
        }
};