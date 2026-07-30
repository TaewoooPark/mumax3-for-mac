# mumax³ Apple Silicon (Metal) 포팅 계획

CUDA 전용인 mumax³를 Apple Silicon GPU(Metal)에서 네이티브로 구동하기 위한 조사 결과 및 실행 계획.

> 구현 업데이트: 이 문서의 조사 이후 실제 포트는 기존 `cuda` 공개 API를 유지하는
> `darwin/arm64` 빌드 태그 방식으로 구현했다. 65개 production 커널은 감사 가능한
> 생성기로 MSL로 변환하고, Metal 소스는 실행 시 컴파일·파이프라인 캐시한다
> (CI에서는 `xcrun metal`로 65개 전체를 별도 정적 컴파일). FFT는 외부 의존성을
> 추가하지 않는 macOS 14+ MPSGraph R2C/C2R을 채택했고, RNG는 Philox4x32-10,
> reduction은 명시적 threadgroup barrier와 uint CAS float atomic을 사용한다.
> 따라서 아래의 VkFFT 우선순위와 `.metallib` 상시 임베딩은 조사 단계의 후보안이며
> 최종 구현과 다른 부분이다. 최종 감사 결과는
> `cuda/metal/kernels/AUDIT.md`와 Apple Silicon CI를 기준으로 한다.

---

## 1. 현 코드베이스의 CUDA 의존성 실측

포팅 난이도는 "CUDA를 얼마나 깊게 쓰는가"로 결정된다. 실측 결과 mumax³는 **CUDA를
매우 얕게** 쓰고 있으며, 이것이 이 포팅을 현실적으로 만드는 핵심 근거다.

### 1.1 규모

| 항목 | 수치 |
|---|---|
| Go 소스 | 346 파일 |
| CUDA 커널 (`cuda/*.cu`) | **65 개** |
| 자동 생성 래퍼 (`*_wrapper.go`) | 65 개 (임베딩된 PTX 총 **5.8 MB**) |
| `cuda` 패키지 공개 API | 91 함수 |
| `engine/`에서 `cuda.` 호출 지점 | 381 곳 |
| 회귀 테스트 (`test/*.mx3`) | **176 개** |

### 1.2 사용 중인 CUDA 기능 (전수 조사)

| 기능 | 사용 여부 | 비고 |
|---|---|---|
| `__shared__` 메모리 | `reduce.h` 매크로 **한 곳뿐** | reduce 커널 6개만 사용 |
| warp intrinsics (`__shfl`, `__ballot`) | **없음** | |
| texture / surface / `__ldg` | **없음** | |
| double precision | **없음** (전부 fp32) | |
| dynamic parallelism | **없음** | |
| CUDA Graphs / multi-stream | **없음** (`stream0` 단일 스트림) | |
| float atomics | `atomicAdd` 2개 커널<br>`atomicMax`(int 재해석) 4개 커널 | Metal 대응 필요 (§4.4) |
| 커널 인자 타입 | `float`, `float*`, `int`, `uint8_t`, `uint8_t*` **5종뿐** | `cuda2go.go`의 `typemap` |
| 최대 포인터 인자 수 | 16 (`cubicanisotropy2.cu`) | Metal 31-buffer 한도 이내 ✅ |

**결론: 커널 65개 전부가 "포인터 몇 개 + 스칼라 몇 개를 받아 인덱스 계산 후
element-wise/stencil 연산"이라는 동일한 형태다.** 기계적 변환이 가능한 구조.

### 1.3 외부 CUDA 라이브러리 의존 (실제 사용 심볼만)

```
cuFFT   : Plan3d, R2C, C2R, SetStream, ExecR2C, ExecC2R          (4~6 심볼)
cuRAND  : CreateGenerator(PSEUDO_DEFAULT), SetSeed, GenerateNormal (3 심볼)
Driver  : Init, CtxCreate/SetCurrent, MemAlloc/Free, MemAllocHost/FreeHost,
          Memcpy{HtoD,DtoH,Async}, MemsetD8/D32(Async), ModuleLoadData,
          ModuleGetFunction, LaunchKernel, Stream*, Device*   (약 30 심볼)
```

cuBLAS·cuDNN·Thrust 등 무거운 의존이 **전혀 없다.** 이게 결정적이다.

### 1.4 아키텍처 구조 (교체 지점)

```
engine/ (물리 · 스크립트 · 솔버, 381 call sites)
   │  ← 이 경계는 그대로 유지
   ▼
cuda/  (91개 공개 함수: Add, Madd2, LLTorque2, NewDemag, Reduce* …)
   │
   ├─ *_wrapper.go   ← cuda2go.go가 .cu에서 자동 생성 (PTX 임베딩)
   ├─ cuda/cu/       ← CUDA Driver API의 얇은 Go 바인딩
   ├─ cuda/cufft/    ← cgo
   └─ cuda/curand/   ← cgo
```

**교체 대상은 점선 아래 전부, 유지 대상은 `engine/` 전부.** `cuda` 패키지의 공개
API 91개만 시그니처를 보존하면 상위 계층은 한 줄도 안 고쳐도 된다.

---

## 2. 조사한 선행 사례 및 접근법

### 2.1 소규모 스타트업/오픈소스 레포 조사 결과

요청하신 국내 스타트업(sionic-ai, Trillion Labs, altalt.io) 및 유사 팀들의
레포를 조사한 결과, **CUDA→Metal 변환 레이어를 직접 만드는 팀은 사실상 없었다.**
공통적으로 관찰된 패턴은 다음과 같다.

| 팀/레포 | 실제로 한 일 | 시사점 |
|---|---|---|
| `altalt-org/Lightning-SimulWhisper` | Whisper를 **MLX/CoreML로 네이티브 재작성**, 디코더 ~15× | 번역이 아니라 "핫패스 네이티브 재작성" |
| `sionic-ai/flasma` | 런타임 추상화(TF runtime)에 얹어 크로스 플랫폼 확보 | 백엔드 경계를 위로 올림 |
| `vllm-project/vllm-metal` | vLLM의 **하드웨어 플러그인**으로 Metal 백엔드 분리 | 안정 API 뒤에 백엔드 교체 |
| `manishklach/mlx-metal-kernels` | attention/RoPE 등 **커널만** MSL로 재작성 후 검증 | 커널 레이어만 갈아끼움 |

> **핵심 교훈:** 성공한 팀은 전부 *"상위 API는 고정, 커널 레이어만 네이티브
> 재작성 + 수치 동등성 검증"* 전략을 썼다. 소스 번역 레이어에 베팅한 사례는
> 없다. mumax³는 `cuda` 패키지가 이미 그 "안정 API 경계" 역할을 하고 있어
> 이 패턴에 정확히 들어맞는다.

### 2.2 CUDA→Apple GPU 실행 경로 (기술 스택별)

| # | 접근 | 파이프라인 | 성숙도 | mumax³ 적합성 |
|---|---|---|---|---|
| **A** | **네이티브 Metal 백엔드** | `.cu` → MSL 트랜스파일 → `.metallib`, Go↔Metal 바인딩 자체 구현 | 우리가 만듦 | ★★★★★ |
| B | CUDA 호환 런타임 | `Lulzx/cuda-metal`(CuMetal): CUDA/PTX → MSL, `libcuda.dylib` shim, cuFFT/cuRAND 부분 지원, Apache-2.0 | 실험적, 검증 부담 | ★★★☆☆ |
| C | HIP→SPIR-V→MoltenVK | OpenFPM 5.2.0이 채택. clang/HIP → SPIR-V → MoltenVK(SPIRV-Cross) → MSL. M3 Pro에서 CPU 대비 10× | 툴체인 무거움(LLVM 전체 빌드, clspv 버전 핀), 64-bit 연산 IR 해킹 필요 | ★★☆☆☆ |
| D | Slang 단일 소스 | Khronos 호스팅. 한 소스 → **CUDA + MSL + SPIR-V** 동시 생성, 식별자 보존된 가독성 있는 코드 출력 | 안정적, 그래픽스 중심 | ★★★★☆ |
| E | MLX / wgpu 재작성 | 배열 프레임워크 위에 물리 재구현 | 사실상 mumax⁺ 재작성 | ★☆☆☆☆ |

### 2.3 FFT (demag 컨볼루션 — 최대 성능 병목)

demag 필드는 3D R2C/C2R FFT를 스텝당 6회(3D) 수행한다. cuFFT 대체는 두 선택지:

| | VkFFT (Metal 백엔드) | MPSGraph FFT |
|---|---|---|
| 라이선스 | MIT | Apple 시스템 프레임워크 |
| R2C/C2R 3D | ✅ | ✅ (Metal.jl 1.10이 이걸로 rfft/irfft 구현) |
| 임의 크기 | ✅ radix 2/3/4/5/7/8/11/13 + Rader + Bluestein | 제약 불명확 |
| 바인딩 | C 헤더 + metal-cpp (cgo) | Obj-C (cgo) |
| **판정** | **1순위** — mumax는 zero-padding 후 임의 크기 FFT를 쓰므로 Bluestein 지원이 필수 | 2순위/교차검증용 |

### 2.4 Apple GPU 하드웨어 제약 (확인 완료)

- **버퍼 바인딩 31개 한도** — 최대 인자 커널이 16개이므로 여유 ✅
- **device float atomic 미지원** — `atomic_fetch_add(float)`가 MSL에 없음. reduce 재설계 필요 ⚠️
- **SIMD 폭 32** (CUDA warp와 동일), threadgroup = CUDA block, `simd_sum()` 제공 ✅
- **threadgroup 최대 1024** 이지만 실제 상한은 파이프라인별 `maxTotalThreadsPerThreadgroup`.
  현재 `make3DConf`의 32×32=1024는 레지스터 사용량 많은 커널(exchange, dmi, slonczewski)에서
  초과할 수 있어 **런타임 조회 후 클램프 필수** ⚠️
- **grid 차원 65535 한도 없음** — `make1DConf`의 nx/ny 분할 트릭 제거 가능 ✅
- **통합 메모리** — `MTLStorageModeShared`로 HtoD/DtoH 복사가 사실상 무료. 출력/테이블 저장 경로가 크게 빨라짐 ✅

---

## 3. 권장 방안: **A안 (네이티브 Metal 백엔드) + 기계적 트랜스파일러**

### 왜 A안인가

1. **커널이 이례적으로 단순하다.** shared/warp/texture/fp64 미사용, 인자 타입 5종.
   65개 전부가 동일 패턴 → `cuda2go.go`(약 200줄)를 확장해 **CUDA C → MSL 트랜스파일러**를
   만드는 것이 번역 레이어를 신뢰하는 것보다 확실하다.
2. **의존 라이브러리가 3개뿐**이고 사용 심볼이 각각 한 자릿수다. cuFFT는 VkFFT로,
   cuRAND는 Philox 커널 40줄로 대체 가능.
3. **번역 레이어(B·C)는 디버깅 불가능한 실패를 만든다.** 물리 시뮬레이션은 "돌아가는 것"이
   아니라 "수치가 맞는 것"이 요구사항이다. IR 해킹 계층이 낄수록 176개 회귀 테스트
   실패 시 원인 추적이 불가능해진다.
4. **스타트업 사례가 전부 이 패턴이다** (§2.1).
5. `cuda` 패키지 공개 API 91개를 보존하므로 **`engine/` 381개 호출 지점은 무수정.**

### CUDA 경로는 유지한다

Go build tag로 백엔드를 분리해 리눅스/윈도우 NVIDIA 빌드는 그대로 살린다.
업스트림 mumax³와의 머지 가능성을 보존하는 것이 장기적으로 중요하다.

```
cuda/backend_cuda.go   //go:build cuda   (기본)
cuda/backend_metal.go  //go:build metal  (darwin/arm64)
```

### 보험: D안(Slang) 병행 검토

트랜스파일러 작성이 예상보다 어려운 커널(reduce, kernmul 계열)이 나오면, 해당
커널만 Slang으로 재작성해 CUDA/MSL 양쪽을 동시 생성하는 방식으로 후퇴할 수 있다.
전면 채택은 하지 않되 탈출구로 확보해 둔다.

---

## 4. 구체 설계

### 4.1 패키지 구조

```
cuda/
├── mtl/                    ★신규: cu/ 와 1:1 대응하는 Metal Go 바인딩 (cgo + Obj-C)
│   ├── device.go/.m        MTLDevice, 이름/메모리 조회
│   ├── memory.go/.m        MemAlloc→newBufferWithLength(StorageModeShared)
│   ├── module.go/.m        ModuleLoadData→newLibraryWithData(.metallib)
│   ├── function.go/.m      GetFunction→newComputePipelineStateWithFunction
│   ├── execution.go/.m     LaunchKernel→dispatchThreadgroups (§4.2 배칭)
│   └── stream.go/.m        Stream→MTLCommandQueue + 누적 CommandBuffer
├── mtlfft/                 ★신규: VkFFT-Metal cgo 래퍼 (cufft/ 대체)
├── *.metal                 ★신규: 65개 MSL 커널 (.cu에서 생성)
├── *_wrapper_metal.go      ★신규: cuda2metal이 생성, .metallib 임베딩
└── cuda2metal.go           ★신규: cuda2go.go 확장 (CUDA C → MSL + Go 래퍼)
```

### 4.2 커맨드 버퍼 배칭 (성능의 핵심)

mumax³는 RK45 스텝 하나당 수십 개의 작은 커널을 던진다. **런치마다 `MTLCommandBuffer`를
새로 만들면 성능이 무너진다.** Metal.jl 1.10이 정확히 이 문제를 "하나의 command buffer를
열어두고 런치를 그 안에 누적, sync 지점에서만 commit"으로 해결했다.

```
CUDA 의미론                     Metal 매핑
─────────────────────────────────────────────────────────────
stream0 에 커널 enqueue    →   열려 있는 CommandBuffer 의 ComputeEncoder 에 dispatch
Sync() / StreamSynchronize →   encoder.endEncoding + commit + waitUntilCompleted
Memcpy (동기)              →   commit → wait → memcpy (통합 메모리라 사실상 memcpy만)
```
- 인코더는 dispatch 사이에 자동 배리어가 걸리므로(`MTLDispatchTypeSerial`) CUDA 스트림의
  순차 실행 의미론과 일치한다.
- 커맨드 버퍼당 dispatch 상한(수천)에 도달하면 자동 flush.

### 4.3 CUDA C → MSL 트랜스파일 규칙

65개 커널이 균일하므로 아래 매핑 테이블로 기계 변환한다.

| CUDA | MSL |
|---|---|
| `extern "C" __global__ void f(...)` | `kernel void f(...)` + 인자에 `[[buffer(n)]]` 부여 |
| `float* __restrict__ p` | `device float* p [[buffer(n)]]` |
| 스칼라 인자 (`float`, `int`, `uint8_t`) | 단일 `constant Params& P [[buffer(k)]]` 구조체로 묶음 |
| `blockIdx`, `blockDim`, `threadIdx` | `threadgroup_position_in_grid`, `threads_per_threadgroup`, `thread_position_in_threadgroup` |
| `gridDim` | `threadgroups_per_grid` |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__shared__ float s[N]` | `threadgroup float s[N]` (인자 또는 지역 선언) |
| `make_float3`, `float3` | MSL 내장 `float3` (연산자 오버로드 불필요 → `float3.h` 대폭 축소) |
| `sqrtf`, `atan2f`, `fabs`, `min/max` | `metal::sqrt`, `metal::atan2`, `metal::fabs`, `metal::min/max` |
| `rsqrtf` | `metal::rsqrt` |
| `inline __device__` | `inline` (또는 헤더 그대로) |

`float3.h`의 연산자 오버로드는 MSL이 네이티브로 제공하므로 삭제 가능 — 코드가 오히려 줄어든다.
`stencil.h`, `amul.h`, `constants.h`, `exchange.h`, `sum.h`는 순수 매크로/inline이라 거의 그대로 재사용된다.

빌드는 `xcrun metal -c *.metal → xcrun metallib`로 `.metallib`을 만들고, PTX처럼 Go 소스에
임베딩(현재 5.8 MB PTX 대비 훨씬 작을 것)한다.

### 4.4 reduce 커널 재설계 (유일한 비자명 부분)

Metal에는 device 메모리 float atomic이 없다. `reduce.h`의
`atomicAdd(dst, sdata[0])` / `atomicFmaxabs`를 그대로 옮길 수 없다.

- `atomicFmaxabs`는 이미 `atomicMax((int*)a, ...)` — **정수 atomic이므로 Metal에서 그대로 동작.**
  (IEEE-754 양수 float은 정수 비교 순서가 보존됨)
- `atomicAdd(float*)` (reducesum, reducedot 2개)만 문제. 두 가지 안:
  1. **2-pass reduction** (권장): 1단계에서 threadgroup별 부분합을 버퍼에 쓰고, 2단계 커널이
     합산. atomic 불필요 + **결정론적** (실행마다 동일 결과 → 회귀 테스트에 유리, 현 CUDA 버전보다 개선)
  2. uint CAS 루프 (`atomic_compare_exchange_weak_explicit`) — 원본 의미론 유지하나 비결정적

동시에 수동 warp-unroll(`volatile float* smem` 부분)을 `simd_sum()` / `simd_max()`로 교체하면
threadgroup 메모리 왕복이 줄어 원본보다 빨라진다.

### 4.5 cuRAND 대체

사용 심볼이 `CreateGenerator / SetSeed / GenerateNormal` 뿐이다.
**Philox4×32-10 counter-based RNG를 MSL 커널로 직접 구현** (약 40줄) 후 Box-Muller로 정규분포 생성.

> ⚠️ **주의:** cuRAND의 기본 XORWOW와 난수열이 다르므로 `Temp > 0` 시뮬레이션은
> CUDA 버전과 **비트 일치하지 않는다.** 이는 정상이며, 해당 테스트는 통계적
> 비교(평균/분산/자기상관)로 검증해야 한다. §6 참조.

### 4.6 런치 설정 조정

```go
// make3DConf: 32×32=1024 → 파이프라인별 maxTotalThreadsPerThreadgroup 조회 후 클램프
// make1DConf: Metal은 grid 65535 제한이 없으므로 nx/ny 분할 제거 → 1D grid 그대로
```
Apple GPU에 맞춰 타일을 32×8 등으로 튜닝하는 것은 정확성 확보 후 최적화 단계에서.

---

## 5. 단계별 로드맵

| 단계 | 내용 | 산출물 / 검증 |
|---|---|---|
| **P0. 골격** | `cuda/mtl` 최소 바인딩(Device/Buffer/Library/Pipeline/CommandQueue) + `madd2` 커널 하나만 수동 MSL 변환 | Go에서 Metal 커널 1개 실행 성공. 아키텍처 리스크 조기 제거 |
| **P1. 트랜스파일러** | `cuda2metal.go` 작성, 65개 `.cu` → `.metal` 자동 변환, `.metallib` 빌드 + 임베딩, 래퍼 생성 | 65개 전부 컴파일 통과 |
| **P2. 런타임** | 메모리(통합 메모리 최적화), 커맨드 버퍼 배칭, `Sync()` 의미론, `init.go`의 디바이스 조회 | `cuda` 패키지 단위 테스트 통과 |
| **P3. reduce** | 2-pass + `simd_sum` 재설계 | reduce 6종 CPU 참조값 대비 검증 |
| **P4. FFT** | VkFFT-Metal cgo 래퍼(`mtlfft`), `fft3dr2c/c2r` 교체 | `conv_selftest.go` 통과, cuFFT 대비 수치 비교 |
| **P5. RNG** | Philox MSL 커널 | 통계 검증 |
| **P6. 회귀** | `test/` 176개 `.mx3` 실행 | 허용 오차 내 통과 (§6) |
| **P7. 빌드/배포** | build tag 분리, macOS Makefile, CI(GitHub Actions macos-14 arm64 러너) | `make` 한 번으로 빌드 |
| **P8. 최적화** | 타일 크기 튜닝, 커널 퓨전, 통합 메모리 활용한 I/O 경로 단축 | 벤치마크 (`bench/`) |

**중요:** P0을 반드시 먼저, 독립적으로 끝낼 것. 여기서 커맨드 버퍼 배칭 성능이
확인되지 않으면 전체 계획의 전제가 무너진다.

---

## 6. 검증 전략

`test/` 176개 `.mx3` 회귀 테스트가 이 포팅의 안전망이다.

1. **결정론적 테스트 (`Temp == 0`)**: CUDA 결과와 상대오차 비교. fp32 연산 순서 차이로
   비트 일치는 기대하지 말고 허용 오차(예: 자기화 성분 1e-5, 에너지 1e-4)를 설정.
2. **확률적 테스트 (`Temp > 0`)**: 난수열이 다르므로 비트 비교 불가.
   → 통계량(평균 자기화, 요동 분산) 및 물리 법칙(FDT, 평형 분포) 검증으로 대체.
3. **컨볼루션 자체 검증**: 기존 `conv_selftest.go` / `oommf/` 비교 경로를 그대로 활용.
4. **커널 단위 검증**: 트랜스파일 단계에서 커널별로 무작위 입력 → CPU 참조 구현과 대조하는
   하네스를 만든다. 65개를 한 번에 통합한 뒤 디버깅하는 것은 피한다.

---

## 7. 주요 리스크

| 리스크 | 영향 | 완화 |
|---|---|---|
| 커맨드 버퍼 오버헤드로 소형 메시에서 CPU-bound | 높음 | P0에서 조기 측정, 배칭 필수화 |
| VkFFT-Metal이 mumax 특정 크기에서 실패/느림 | 중간 | MPSGraph FFT를 대안으로 확보 (§2.3) |
| threadgroup 크기 초과로 파이프라인 생성 실패 | 중간 | 런타임 조회 후 클램프 (§4.6) |
| fp32 연산 순서 차이로 장시간 시뮬레이션 궤적 발산 | 중간 | 물리량(에너지 보존 등) 기준 검증으로 전환 |
| cgo + Objective-C 빌드 복잡도 | 낮음 | `go-mtl`, `green-aloe/metal` 등 기존 바인딩 참고 |
| 업스트림 mumax³와 divergence | 낮음 | build tag로 CUDA 경로 보존 |

---

## 8. 참고 자료

- OpenFPM 5.2.0 Metal 백엔드 PR — HIP/clang → SPIR-V → MoltenVK 파이프라인:
  <https://github.com/mosaic-group/openfpm/pull/18>
- VkFFT (Vulkan/CUDA/HIP/OpenCL/Level Zero/**Metal** FFT, MIT): <https://github.com/DTolm/VkFFT>
- CuMetal — CUDA compiler/runtime for Apple Silicon (Apache-2.0): <https://github.com/Lulzx/cuda-metal>
- Metal.jl 1.10 — MPSGraph 기반 FFT, 커맨드 버퍼 배칭:
  <https://juliagpu.org/post/2026-07-01-metal-1.10/index.html>
- Slang (Khronos) 컴파일 타깃 — CUDA/Metal/SPIR-V 단일 소스: <https://shader-slang.org/slang/user-guide/targets>
- Metal Feature Set Tables (31-buffer 한도 등): <https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf>
- Metal Shading Language Specification: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
- go-mtl (Go ↔ Metal 바인딩 참고): <https://github.com/hupe1980/go-mtl>
- MoltenVK: <https://github.com/KhronosGroup/MoltenVK>
- altalt-org/Lightning-SimulWhisper (MLX/CoreML 네이티브 재작성 사례):
  <https://github.com/altalt-org/Lightning-SimulWhisper>
- vllm-project/vllm-metal (하드웨어 플러그인으로 백엔드 분리 사례):
  <https://github.com/vllm-project/vllm-metal>
- mumax⁺ (C++/CUDA 후속 프로젝트): <https://github.com/mumax/plus>
