// Compatibility prelude for cuda/metal/cmd/cuda2metal.
// Source of truth for kernel bodies: the audited CUDA dialect in cuda/*.cu.
//
// ABI: every original CUDA argument occupies the same numbered Metal buffer
// binding. Pointer arguments use setBuffer; scalar arguments use setBytes. A
// final uint32 binding records pointer presence because nil Metal buffers are
// represented by a safe dummy resource at dispatch.

#include <metal_stdlib>
using namespace metal;

#define REDUCE_BLOCKSIZE 512
#define PI     3.1415926535897932384626433
#define MU0    (4*PI*1e-7)
#define QE     1.60217646E-19
#define MUB    9.2740091523E-24
#define GAMMA0 1.7595e11
#define HBAR   1.05457173E-34

#define index(ix,iy,iz,Nx,Ny,Nz) ((((iz)*(Ny) + (iy)) * (Nx)) + (ix))
#define idx(ix,iy,iz) (index((ix),(iy),(iz),(Nx),(Ny),(Nz)))
#define MOD(n,M) (((n) % (M) + (M)) % (M))
#define PBCx (PBC & 1)
#define PBCy (PBC & 2)
#define PBCz (PBC & 4)
#define hclampx(ix) (PBCx ? MOD((ix), Nx) : min((ix), Nx-1))
#define lclampx(ix) (PBCx ? MOD((ix), Nx) : max((ix), 0))
#define hclampy(iy) (PBCy ? MOD((iy), Ny) : min((iy), Ny-1))
#define lclampy(iy) (PBCy ? MOD((iy), Ny) : max((iy), 0))
#define hclampz(iz) (PBCz ? MOD((iz), Nz) : min((iz), Nz-1))
#define lclampz(iz) (PBCz ? MOD((iz), Nz) : max((iz), 0))
#define symidx(i,j) ((j)<=(i) ? ((((i)*((i)+1))/2)+(j)) : ((((j)*((j)+1))/2)+(i)))
#define is0(m) (dot((m), (m)) == 0.0f)

inline float3 make_float3(float x, float y, float z) {
    return float3(x, y, z);
}

inline float len(float3 value) {
    return length(value);
}

inline float3 normalized(float3 value) {
    float valueLength = length(value);
    return valueLength != 0.0f ? value / valueLength : float3(0.0f);
}

inline float pow2(float value) {
    return value * value;
}

inline float pow3(float value) {
    return value * value * value;
}

inline float pow4(float value) {
    float square = value * value;
    return square * square;
}

// One Hopf-index kernel uses CUDA's double-complex helper even though all
// source arrays and outputs are float32. Apple GPUs do not expose fp64;
// preserve the complex algebra in the engine's native float32 precision.
inline float2 mumaxMakeComplex(float realPart, float imaginaryPart) {
    return float2(realPart, imaginaryPart);
}

inline float2 mumaxComplexAdd(float2 a, float2 b) {
    return a + b;
}

inline float2 mumaxComplexSub(float2 a, float2 b) {
    return a - b;
}

inline float2 mumaxComplexMul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline float mumaxComplexImag(float2 value) {
    return value.y;
}

inline bool mumaxPointerPresent(uint mask, uint argumentIndex) {
    return (mask & (1u << argumentIndex)) != 0u;
}

inline float amul(device const float* array, bool present,
                  float multiplier, int indexValue) {
    return present ? multiplier * array[indexValue] : multiplier;
}

inline float3 vmul(device const float* x, device const float* y, device const float* z,
                   bool xPresent, bool yPresent, bool zPresent,
                   float mx, float my, float mz, int indexValue) {
    return make_float3(amul(x, xPresent, mx, indexValue),
                       amul(y, yPresent, my, indexValue),
                       amul(z, zPresent, mz, indexValue));
}

inline float inv_Msat(device const float* ms, bool present,
                      float multiplier, int indexValue) {
    float value = amul(ms, present, multiplier, indexValue);
    return value == 0.0f ? 0.0f : 1.0f / value;
}

inline float sum(float a, float b) {
    return a + b;
}

// Native atomic float add. All Apple Silicon GPUs support Metal 3 which
// provides atomic<float> with hardware-accelerated fetch_add. This replaces
// the CAS loop used by the initial port.
inline void mumaxAtomicAdd(device float* address, float value) {
    device atomic<float>* target =
        reinterpret_cast<device atomic<float>*>(address);
    atomic_fetch_add_explicit(target, value, memory_order_relaxed);
}

inline void atomicFmaxabs(device float* address, float value) {
    device atomic_uint* target = reinterpret_cast<device atomic_uint*>(address);
    uint desired = as_type<uint>(fabs(value));
    uint expected = atomic_load_explicit(target, memory_order_relaxed);
    // CUDA's source performs signed atomicMax over the non-negative float bit
    // pattern. Retaining that comparison also preserves its NaN/Inf behavior.
    while (as_type<int>(expected) < as_type<int>(desired)) {
        if (atomic_compare_exchange_weak_explicit(
                target, &expected, desired,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
}

#define atomicAdd mumaxAtomicAdd

// SIMD-group optimized reduction. Uses simd_sum / simd_max to reduce within
// each 32-wide SIMD group (no barrier needed), then a single threadgroup
// barrier to exchange per-group results. This replaces the O(log N) barrier
// tree from the initial CUDA translation with a 2-level reduction:
//   Level 1: simd-level (hardware shuffle, zero overhead)
//   Level 2: threadgroup-level (one barrier, ≤16 values on 512 threads)
//
// The macro adapts to the caller's `op` by using a generic simd_shuffle_down
// tree for operations other than sum (e.g., max). The `simd_reduce_op` helper
// is defined per-op before each kernel that uses it.

inline float simd_reduce_sum(float val) {
    return simd_sum(val);
}

inline float simd_reduce_max(float val) {
    // Apple GPU SIMD width is 32; use explicit shuffle tree
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val = max(val, simd_shuffle_down(val, offset));
    }
    return val;
}

#define reduce(load, op, atomicOp)                                           \
    /* Apple GPU SIMD width is always 32 */                                  \
    const uint tid = threadIdx.x;                                           \
    const uint simd_lane = tid & 31u;                                       \
    const uint simd_group = tid >> 5u;                                      \
    const uint simd_groups_per_tg = blockDim.x >> 5u;                       \
    int i = int(blockIdx.x * blockDim.x + threadIdx.x);                     \
    float mine = initVal;                                                   \
    int stride = int(gridDim.x * blockDim.x);                               \
    while (i < n) {                                                         \
        mine = op(mine, load(i));                                           \
        i += stride;                                                        \
    }                                                                       \
    /* Level 1: reduce within each SIMD group via hardware shuffle */        \
    for (ushort offset = 16; offset > 0; offset >>= 1) {                    \
        mine = op(mine, simd_shuffle_down(mine, offset));                   \
    }                                                                       \
    /* Level 2: exchange across SIMD groups via threadgroup memory */        \
    threadgroup float sdata[16];                                            \
    if (simd_lane == 0) {                                                   \
        sdata[simd_group] = mine;                                           \
    }                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    /* Final reduce: only first SIMD group reduces across groups */          \
    if (simd_group == 0) {                                                  \
        mine = (tid < simd_groups_per_tg) ? sdata[tid] : initVal;          \
        for (ushort offset = 16; offset > 0; offset >>= 1) {                \
            mine = op(mine, simd_shuffle_down(mine, offset));               \
        }                                                                   \
    }                                                                       \
    if (tid == 0) {                                                         \
        atomicOp(dst, mine);                                                \
    }
