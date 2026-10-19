#pragma once
// atlas: one-column PTQ1_0 mat-vec with coalesced weight loads (ATLAS_PTQ1_GEMV=1).
//
// The stock one-column paths give each lane its own 28-byte block, so a warp-wide 4-byte weight load
// touches 32 addresses spread over ~896 B (7 cache lines): ~7 L1 wavefronts per instruction where
// 1 would do. Here a warp owns one row and moves each run of 32 consecutive blocks (896 B = 56 x 16 B)
// with two 16-byte vector loads per lane, stages it in shared memory, and every lane then reads its
// own block from there (block stride 7 words, coprime with 32 banks: conflict-free). The dot product
// itself is the stock vec_dot_ptq1_0_q8_1_multi on the staged block, activations in the same
// warp-transposed SOA_ISUM layout, so results match the stock path up to summation order.
//
// All weight chunks of the row are loaded into registers BEFORE the programmatic-dependent-launch
// wait: weights are constant, so their DRAM latency overlaps the tail of the previous kernel.
// Activations (written by the previous kernel) are only read after ggml_cuda_pdl_sync().

#define ATLAS_PTQ1_GEMV_NW 4   // warps (rows) per CTA

template <int NCH, bool has_gate>
__launch_bounds__(32 * ATLAS_PTQ1_GEMV_NW, 1)
static __global__ void atlas_ptq1_gemv(
        const void * __restrict__ vx, const void * __restrict__ vgate, const block_q8_1 * __restrict__ y,
        float * __restrict__ dst, const int nblk, const int nrows, const int stride_row_x, const int glu_op,
        const float * __restrict__ x_bias) {
    constexpr int NW = ATLAS_PTQ1_GEMV_NW;
    constexpr int NM = has_gate ? 2 : 1;
    __shared__ __align__(16) uint4 sm[NW][NM][56];

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int row  = blockIdx.x * NW + warp;
    const bool row_ok = row < nrows;

    // 1. weights -> registers (constant data, safe before the dependency wait)
    uint4 wa[NM][NCH], wb[NM][NCH];
#pragma unroll
    for (int m = 0; m < NM; ++m) {
        const char * base = (const char *) (m == 0 ? vx : vgate) + (size_t) row * stride_row_x * sizeof(block_ptq1_0);
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const int nb = min(32, nblk - c * 32);
            const int nv = nb > 0 ? nb * (int) sizeof(block_ptq1_0) / 16 : 0;
            const uint4 * src = (const uint4 *) (base + (size_t) c * 32 * sizeof(block_ptq1_0));
            wa[m][c] = (row_ok && lane      < nv) ? src[lane]      : make_uint4(0, 0, 0, 0);
            wb[m][c] = (row_ok && lane + 32 < nv) ? src[lane + 32] : make_uint4(0, 0, 0, 0);
        }
    }

    // 2. wait for the producer of y
    ggml_cuda_pdl_sync();

    float acc = 0.0f, accg = 0.0f;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
        const int nb = min(32, nblk - c * 32);
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            sm[warp][m][lane] = wa[m][c];
            if (lane < 24) {
                sm[warp][m][lane + 32] = wb[m][c];
            }
        }
        __syncwarp();
        if (row_ok && lane < nb) {
            const int kb = c * 32 + lane;
            float d;
            vec_dot_ptq1_0_q8_1_multi<1>(&sm[warp][0][0], y, lane, kb, 0, &d);
            acc += d;
            if constexpr (has_gate) {
                vec_dot_ptq1_0_q8_1_multi<1>(&sm[warp][1][0], y, lane, kb, 0, &d);
                accg += d;
            }
        }
        __syncwarp();
    }

    acc = warp_reduce_sum<32>(acc);
    if constexpr (has_gate) {
        accg = warp_reduce_sum<32>(accg);
    }
    if (lane == 0 && row_ok) {
        float r = acc;
        if (x_bias) {
            r += x_bias[row];   // fused residual add (ggml_cuda mul_mat + add fusion)
        }
        if constexpr (has_gate) {
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU: r *= ggml_cuda_op_silu_single(accg); break;
                case GGML_GLU_OP_GEGLU:  r *= ggml_cuda_op_gelu_single(accg); break;
                default: break;
            }
        }
        dst[row] = r;
    }
}

static bool atlas_ptq1_gemv_enabled() {
    static const bool on = getenv("ATLAS_PTQ1_GEMV") != nullptr && atoi(getenv("ATLAS_PTQ1_GEMV")) != 0;
    return on;
}

// Returns false when the call shape is not covered (caller falls through to the stock path).
static bool atlas_ptq1_gemv_launch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, cudaStream_t stream) {
    {
        static const bool dbg = getenv("ATLAS_PTQ1_GEMV_DEBUG") != nullptr;
        static int n = 0;
        if (dbg && n++ < 40) {
            fprintf(stderr, "atlas_gemv: rows %d ncols_x %d stride_row_x %d gate %d bias %d scale %d\n", nrows_x, ncols_x, stride_row_x,
                    fusion.gate != nullptr, fusion.x_bias != nullptr || fusion.gate_bias != nullptr, fusion.x_scale != nullptr || fusion.gate_scale != nullptr);
        }
    }
    if (fusion.gate_bias || fusion.x_scale || fusion.gate_scale) {
        return false;
    }
    if (fusion.gate && fusion.glu_op != GGML_GLU_OP_SWIGLU && fusion.glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    const int nblk = ncols_x / QK_PTQ1_0;
    const int nch  = (nblk + 31) / 32;
    static const int max_nch = getenv("ATLAS_PTQ1_GEMV_MAXCH") ? atoi(getenv("ATLAS_PTQ1_GEMV_MAXCH")) : 2;
    if (nblk % 4 != 0 || nch < 1 || nch > 6 || nch > max_nch) {
        return false;   // 16-byte vector loads need nblk*28 % 16 == 0 per chunk
    }
    const dim3 grid((nrows_x + ATLAS_PTQ1_GEMV_NW - 1) / ATLAS_PTQ1_GEMV_NW, 1, 1);
    const dim3 block(32, ATLAS_PTQ1_GEMV_NW, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int glu = fusion.gate ? (int) fusion.glu_op : -1;
#define ATLAS_L(N) \
    if (fusion.gate) { ggml_cuda_kernel_launch(atlas_ptq1_gemv<N, true >, lp, vx, fusion.gate, y, dst, nblk, nrows_x, stride_row_x, glu, (const float *) fusion.x_bias); } \
    else             { ggml_cuda_kernel_launch(atlas_ptq1_gemv<N, false>, lp, vx, nullptr,     y, dst, nblk, nrows_x, stride_row_x, glu, (const float *) fusion.x_bias); }
    switch (nch) {
        case 1: ATLAS_L(1); break;
        case 2: ATLAS_L(2); break;
        case 3: ATLAS_L(3); break;
        case 4: ATLAS_L(4); break;
        case 5: ATLAS_L(5); break;
        case 6: ATLAS_L(6); break;
    }
#undef ATLAS_L
    return true;
}
