// llama-mtp-capture — dump (tokens, pos, h_trunk, target top-K logits, MTP draft
// logits) for a prompt, from a model with an embedded NextN/MTP head (qwen35).
//
// Purpose (Atlas MTP-head distillation, 2026-07):
//   1. TRAINING DATA: the target's post-output_norm hidden rows (t_h_nextn) and
//      its next-token top-K logits are the (input, teacher) pairs for training a
//      quant-matched MTP head.
//   2. PARITY REFERENCE: the MTP context is driven explicitly with the same
//      shifted-batch layout common_speculative's draft-mtp uses in non-shared
//      mode (h row i pairs with token i+1), so an external (PyTorch) replica of
//      graph_mtp can be validated bit-for-bit against the dumped draft logits.
//
// Usage:
//   llama-mtp-capture -m merged-mtp.gguf -p "prompt text" -o outdir
//                     [-k topk=64] [--nfull 4] [-c n_ctx=4096] [-ngl 99]
//
// Output (outdir): manifest.json + raw little-endian arrays:
//   tokens.bin  i32[N]        pos.bin        i32[N]
//   h.bin       f32[N,n_embd] (target t_h_nextn rows, post output_norm)
//   tgt_topk_idx.bin i32[N,K] tgt_topk_val.bin f32[N,K]
//   dft_topk_idx.bin i32[N,K] dft_topk_val.bin f32[N,K]
//   tgt_full.bin f32[nfull,n_vocab]  dft_full.bin f32[nfull,n_vocab]

#include "llama.h"
#include "ggml.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

static void die(const char * msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}

static void write_bin(const std::string & path, const void * data, size_t nbytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) die(("cannot open " + path).c_str());
    fwrite(data, 1, nbytes, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Shard-streaming capture (Phase 2 of MTP-head distillation):
// stream a text corpus through the TARGET context only, dumping per token:
//   h (t_h_nextn, f16), teacher top-K logits (idx i32 + val f32), logsumexp f32.
// Sequences are fixed seq_len blocks; KV cleared between sequences; trainer
// reconstructs the shifted draft input (row 0 of each sequence = zeros).
// Shard dir: shard_NNNNN/{tokens.bin,h.f16.bin,tk_idx.bin,tk_val.bin,lse.bin,meta.json}
// ---------------------------------------------------------------------------
struct shard_buf {
    std::vector<int32_t>      toks;
    std::vector<ggml_fp16_t>  h16;
    std::vector<int32_t>      kidx;
    std::vector<float>        kval;
    std::vector<float>        lse;
    // EAGLE-3 taps: int8 rows (n_taps * n_embd per token) + fp16 scale per (token, tap)
    std::vector<int8_t>       tap_q;
    std::vector<ggml_fp16_t>  tap_sc;
};

static void flush_shard(const std::string & outdir, int shard_id, shard_buf & sb,
                        int n_embd, int topk, int seq_len, int n_taps) {
    if (sb.toks.empty()) return;
    char name[64];
    snprintf(name, sizeof(name), "%s/shard_%05d", outdir.c_str(), shard_id);
    MKDIR(name);
    const std::string d = name;
    const size_t n = sb.toks.size();
    write_bin(d + "/tokens.bin", sb.toks.data(), n * 4);
    write_bin(d + "/h.f16.bin",  sb.h16.data(),  n * (size_t) n_embd * 2);
    write_bin(d + "/tk_idx.bin", sb.kidx.data(), n * (size_t) topk * 4);
    write_bin(d + "/tk_val.bin", sb.kval.data(), n * (size_t) topk * 4);
    write_bin(d + "/lse.bin",    sb.lse.data(),  n * 4);
    if (n_taps > 0) {
        write_bin(d + "/taps.i8.bin",   sb.tap_q.data(),  n * (size_t) n_taps * n_embd);
        write_bin(d + "/taps_sc.f16.bin", sb.tap_sc.data(), n * (size_t) n_taps * 2);
    }
    FILE * mf = fopen((d + "/meta.json").c_str(), "w");
    fprintf(mf, "{\"n_tokens\":%zu,\"n_embd\":%d,\"topk\":%d,\"seq_len\":%d,\"n_taps\":%d}\n",
            n, n_embd, topk, seq_len, n_taps);
    fclose(mf);
    fprintf(stderr, "flushed %s (%zu tokens)\n", name, n);
    sb = shard_buf{};
}

static void topk_lse_rows(const float * logits, int n_rows, int n_vocab, int topk,
                          int32_t * kidx, float * kval, float * lse) {
    const int n_threads = 8;
    std::vector<std::thread> ths;
    for (int t = 0; t < n_threads; t++) {
        ths.emplace_back([=]() {
            std::vector<int32_t> idx(n_vocab);
            for (int i = t; i < n_rows; i += n_threads) {
                const float * row = logits + (size_t) i * n_vocab;
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                                  [&](int a, int b) { return row[a] > row[b]; });
                float mx = row[idx[0]];
                double s = 0.0;
                for (int v = 0; v < n_vocab; v++) s += exp((double) row[v] - mx);
                lse[i] = mx + (float) log(s);
                for (int j = 0; j < topk; j++) {
                    kidx[(size_t) i * topk + j] = idx[j];
                    kval[(size_t) i * topk + j] = row[idx[j]];
                }
            }
        });
    }
    for (auto & th : ths) th.join();
}

int main(int argc, char ** argv) {
    std::string model_path, prompt = "The quick brown fox jumps over the lazy dog because", outdir = "mtp-capture-out";
    std::string corpus_path, taps_arg;
    int topk = 64, nfull = 4, n_ctx = 4096, ngl = 99;
    int seq_len = 2048, shard_tokens = 250000;
    long long max_tokens = 0, skip_tokens = 0;
    int shard_base = 0;

    for (int i = 1; i < argc; i++) {
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) die((std::string("missing value for ") + flag).c_str());
            return argv[++i];
        };
        if      (!strcmp(argv[i], "-m"))       model_path = need("-m");
        else if (!strcmp(argv[i], "-p"))       prompt     = need("-p");
        else if (!strcmp(argv[i], "-f"))       corpus_path = need("-f");
        else if (!strcmp(argv[i], "-o"))       outdir     = need("-o");
        else if (!strcmp(argv[i], "-k"))       topk       = atoi(need("-k"));
        else if (!strcmp(argv[i], "--nfull"))  nfull      = atoi(need("--nfull"));
        else if (!strcmp(argv[i], "-c"))       n_ctx      = atoi(need("-c"));
        else if (!strcmp(argv[i], "-ngl"))     ngl        = atoi(need("-ngl"));
        else if (!strcmp(argv[i], "--seq"))    seq_len    = atoi(need("--seq"));
        else if (!strcmp(argv[i], "--shard-tokens")) shard_tokens = atoi(need("--shard-tokens"));
        else if (!strcmp(argv[i], "--max-tokens"))   max_tokens   = atoll(need("--max-tokens"));
        else if (!strcmp(argv[i], "--skip-tokens"))  skip_tokens  = atoll(need("--skip-tokens"));
        else if (!strcmp(argv[i], "--shard-base"))   shard_base   = atoi(need("--shard-base"));
        else if (!strcmp(argv[i], "--taps"))         taps_arg     = need("--taps");
        else die((std::string("unknown arg ") + argv[i]).c_str());
    }
    if (model_path.empty()) die("need -m model.gguf");
    MKDIR(outdir.c_str());

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("model load failed");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);

    // ---- target context ----
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = n_ctx;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    llama_context * ctx_tgt = llama_init_from_model(model, cparams);
    if (!ctx_tgt) die("target ctx init failed");
    llama_set_embeddings_nextn(ctx_tgt, true, /*masked=*/ false);

    // ---- optional EAGLE-3 multi-layer tap capture ----
    std::vector<int32_t> tap_layers;
    if (!taps_arg.empty()) {
        char * s = (char *) taps_arg.c_str();
        while (*s) {
            tap_layers.push_back((int32_t) strtol(s, &s, 10));
            while (*s == ',' || *s == ' ') s++;
        }
        llama_set_capture_layers(ctx_tgt, tap_layers.data(), (uint32_t) tap_layers.size(), /*masked=*/ false);
        fprintf(stderr, "tap capture enabled on %zu layers: %s\n", tap_layers.size(), taps_arg.c_str());
    }
    const int n_taps = (int) tap_layers.size();

    // ---- shard-streaming corpus capture mode ----
    if (!corpus_path.empty()) {
        MKDIR(outdir.c_str());
        FILE * cf = fopen(corpus_path.c_str(), "rb");
        if (!cf) die("cannot open corpus");
        fseek(cf, 0, SEEK_END);
        const long fsz = ftell(cf);
        fseek(cf, 0, SEEK_SET);
        std::string text(fsz, '\0');
        if (fread(&text[0], 1, fsz, cf) != (size_t) fsz) die("corpus read failed");
        fclose(cf);

        std::vector<llama_token> all(fsz / 2 + 64);
        int n_all = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                   all.data(), (int32_t) all.size(), /*add_special*/ false, /*parse_special*/ false);
        if (n_all <= 0) die("corpus tokenize failed");
        all.resize(n_all);
        if (skip_tokens > 0) {
            if (skip_tokens >= (long long) all.size()) die("--skip-tokens beyond corpus");
            all.erase(all.begin(), all.begin() + skip_tokens);
        }
        if (max_tokens > 0 && (long long) all.size() > max_tokens) all.resize((size_t) max_tokens);
        const long long n_seqs = (long long) all.size() / seq_len;
        fprintf(stderr, "corpus: %zu tokens (skip %lld) -> %lld sequences of %d (dropping tail)\n",
                all.size(), skip_tokens, n_seqs, seq_len);

        const int UB = 512;
        if (seq_len % UB != 0) die("--seq must be a multiple of 512");
        llama_batch b = llama_batch_init(UB, 0, 1);

        shard_buf sb;
        int shard_id = shard_base;
        const int64_t t0 = ggml_time_us();

        for (long long s = 0; s < n_seqs; s++) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, -1, -1);
            const llama_token * st = all.data() + s * seq_len;

            for (int off = 0; off < seq_len; off += UB) {
                b.n_tokens = UB;
                for (int i = 0; i < UB; i++) {
                    b.token[i]     = st[off + i];
                    b.pos[i]       = off + i;
                    b.n_seq_id[i]  = 1;
                    b.seq_id[i][0] = 0;
                    b.logits[i]    = 1;
                }
                if (llama_decode(ctx_tgt, b) != 0) die("decode failed in shard capture");

                const float * h  = llama_get_embeddings_nextn(ctx_tgt);
                const float * lg = llama_get_logits(ctx_tgt);
                if (!h || !lg) die("missing h/logits in shard capture");

                const size_t base_n = sb.toks.size();
                sb.toks.insert(sb.toks.end(), st + off, st + off + UB);
                sb.h16.resize((base_n + UB) * (size_t) n_embd);
                ggml_fp32_to_fp16_row(h, sb.h16.data() + base_n * (size_t) n_embd, (size_t) UB * n_embd);
                sb.kidx.resize((base_n + UB) * (size_t) topk);
                sb.kval.resize((base_n + UB) * (size_t) topk);
                sb.lse.resize(base_n + UB);
                topk_lse_rows(lg, UB, n_vocab, topk,
                              sb.kidx.data() + base_n * (size_t) topk,
                              sb.kval.data() + base_n * (size_t) topk,
                              sb.lse.data()  + base_n);

                if (n_taps > 0) {
                    // capture rows: n_taps * n_embd floats per token; quantize each
                    // (token, tap) segment to int8 with its own fp16 absmax scale
                    const float * cap = llama_get_embeddings_capture(ctx_tgt);
                    if (!cap) die("tap capture enabled but no capture buffer");
                    const size_t seg = (size_t) n_embd;
                    sb.tap_q.resize((base_n + UB) * (size_t) n_taps * n_embd);
                    sb.tap_sc.resize((base_n + UB) * (size_t) n_taps);
                    for (int r = 0; r < UB; ++r) {
                        for (int tj = 0; tj < n_taps; ++tj) {
                            const float * src = cap + ((size_t) r * n_taps + tj) * seg;
                            float amax = 0.0f;
                            for (size_t e = 0; e < seg; ++e) amax = std::max(amax, fabsf(src[e]));
                            const float sc = amax > 0 ? amax / 127.0f : 1.0f;
                            const float inv = 1.0f / sc;
                            int8_t * dst = sb.tap_q.data() + ((base_n + r) * (size_t) n_taps + tj) * seg;
                            for (size_t e = 0; e < seg; ++e) {
                                int v = (int) lroundf(src[e] * inv);
                                dst[e] = (int8_t) std::max(-127, std::min(127, v));
                            }
                            sb.tap_sc[(base_n + r) * (size_t) n_taps + tj] = ggml_fp32_to_fp16(sc);
                        }
                    }
                }
            }

            if ((int) sb.toks.size() >= shard_tokens) {
                flush_shard(outdir, shard_id++, sb, n_embd, topk, seq_len, n_taps);
            }
            if ((s + 1) % 8 == 0) {
                const double el = (ggml_time_us() - t0) / 1e6;
                const double done = (double) (s + 1) * seq_len;
                fprintf(stderr, "seq %lld/%lld  %.0f tok/s  eta %.1f min\n",
                        s + 1, n_seqs, done / el,
                        (n_seqs * (double) seq_len - done) / (done / el) / 60.0);
            }
        }
        flush_shard(outdir, shard_id++, sb, n_embd, topk, seq_len, n_taps);
        llama_batch_free(b);
        llama_free(ctx_tgt);
        llama_model_free(model);
        llama_backend_free();
        fprintf(stderr, "shard capture complete -> %s\n", outdir.c_str());
        return 0;
    }

    // ---- tokenize ----
    std::vector<llama_token> toks(1024);
    int N = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                           toks.data(), (int32_t) toks.size(), /*add_special*/ true, /*parse_special*/ true);
    if (N <= 1) die("tokenize failed / prompt too short");
    if (N > 512) die("prompt longer than one ubatch (512); shorten it");
    toks.resize(N);
    fprintf(stderr, "prompt tokens: %d, n_embd=%d, n_vocab=%d\n", N, n_embd, n_vocab);

    // ---- target prefill, logits + h on every position ----
    llama_batch tb = llama_batch_init(N, 0, 1);
    tb.n_tokens = N;
    for (int i = 0; i < N; i++) {
        tb.token[i]    = toks[i];
        tb.pos[i]      = i;
        tb.n_seq_id[i] = 1;
        tb.seq_id[i][0] = 0;
        tb.logits[i]   = 1;
    }
    if (llama_decode(ctx_tgt, tb) != 0) die("target decode failed");

    const float * h = llama_get_embeddings_nextn(ctx_tgt);
    if (!h) die("no nextn embeddings from target (is this a nextn model?)");
    const float * tgt_logits = llama_get_logits(ctx_tgt);
    if (!tgt_logits) die("no target logits");

    // ---- MTP draft context: fresh KV, explicit shifted batch ----
    llama_context_params dparams = llama_context_default_params();
    dparams.n_ctx    = n_ctx;
    dparams.n_batch  = 512;
    dparams.n_ubatch = 512;
    dparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dparams.ctx_other = nullptr;         // self-contained KV: controlled parity experiment
    llama_context * ctx_dft = llama_init_from_model(model, dparams);
    if (!ctx_dft) die("mtp ctx init failed");
    llama_set_embeddings_nextn(ctx_dft, true, /*masked=*/ true);

    // batch layout mirrors common_speculative_impl_draft_mtp::process():
    //   token[i] = prompt token i, embd row i = h[i-1] (row 0 = zeros: no h before pos 0)
    llama_batch db = llama_batch_init(N, n_embd, 1);
    db.token = (llama_token *) malloc(sizeof(llama_token) * N); // init allocs only embd; MTP needs both
    db.n_tokens = N;
    memset(db.embd, 0, (size_t) N * n_embd * sizeof(float));
    memcpy(db.embd + (size_t) 1 * n_embd, h, (size_t) (N - 1) * n_embd * sizeof(float));
    for (int i = 0; i < N; i++) {
        db.token[i]    = toks[i];
        db.pos[i]      = i;
        db.n_seq_id[i] = 1;
        db.seq_id[i][0] = 0;
        db.logits[i]   = 1;
    }
    if (llama_decode(ctx_dft, db) != 0) die("mtp decode failed");
    const float * dft_logits = llama_get_logits(ctx_dft);
    if (!dft_logits) die("no draft logits");

    // ---- dumps ----
    std::vector<int32_t> pos(N);
    std::iota(pos.begin(), pos.end(), 0);
    write_bin(outdir + "/tokens.bin", toks.data(), (size_t) N * 4);
    write_bin(outdir + "/pos.bin",    pos.data(),  (size_t) N * 4);
    write_bin(outdir + "/h.bin",      h,           (size_t) N * n_embd * 4);

    auto dump_topk = [&](const float * logits, const std::string & prefix) {
        std::vector<int32_t> ki((size_t) N * topk);
        std::vector<float>   kv((size_t) N * topk);
        std::vector<int32_t> idx(n_vocab);
        for (int i = 0; i < N; i++) {
            const float * row = logits + (size_t) i * n_vocab;
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&](int a, int b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; j++) {
                ki[(size_t) i * topk + j] = idx[j];
                kv[(size_t) i * topk + j] = row[idx[j]];
            }
        }
        write_bin(outdir + "/" + prefix + "_topk_idx.bin", ki.data(), ki.size() * 4);
        write_bin(outdir + "/" + prefix + "_topk_val.bin", kv.data(), kv.size() * 4);
    };
    dump_topk(tgt_logits, "tgt");
    dump_topk(dft_logits, "dft");

    const int nf = std::min(nfull, N);
    write_bin(outdir + "/tgt_full.bin", tgt_logits, (size_t) nf * n_vocab * 4);
    write_bin(outdir + "/dft_full.bin", dft_logits, (size_t) nf * n_vocab * 4);

    FILE * mf = fopen((outdir + "/manifest.json").c_str(), "w");
    fprintf(mf, "{\"n_tokens\":%d,\"n_embd\":%d,\"n_vocab\":%d,\"topk\":%d,\"nfull\":%d}\n",
            N, n_embd, n_vocab, topk, nf);
    fclose(mf);

    fprintf(stderr, "capture complete -> %s\n", outdir.c_str());

    free(db.token); db.token = nullptr;
    llama_batch_free(db);
    llama_batch_free(tb);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
