// tlars_core/src/bench_tlars_vd_time_standalone.cpp
#include <armadillo>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>
#include <limits>
#include <filesystem>
#include <ctime>

// your existing headers
#include "TLARS_Solver.hpp"
#include "../../src/vd_lars.hpp"

#include <pthread.h>
#include <sys/qos.h> // Required for QOS_CLASS_USER_INTERACTIVE on some compilers

// ... existing includes ...

static void pin_to_core(int core_id) {
#if defined(__linux__)
    // LINUX: Strict pinning to a physical core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "\n[WARN] Failed to pin to core " << core_id << "\n";
    }
#elif defined(__APPLE__)
    // MACOS: "Pin" to Performance Cores via QoS
    // This guarantees the code runs on P-Cores (fastest), preventing 
    // the OS from throttling it on Efficiency (E) cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    
    (void)core_id; // Silence unused parameter warning
#endif
}

// ----------------------------------------------------
//  CONFIG
// ----------------------------------------------------
static const int N_FIXED   = 1000;
static const int T_SELECT  = 10;
static const int REPS      = 100;

static const int P_LIST[]  = {1000, 10000, 50000};
static const int NUM_P     = sizeof(P_LIST) / sizeof(P_LIST[0]);

static const int L_MULT[]  = {1};
static const int NUM_M     = sizeof(L_MULT) / sizeof(L_MULT[0]);

static const uint64_t BASE_SEED = 123;

static const bool   USE_SIGNAL   = true;   // set false for null
static const int    K_ACTIVE     = 10;     // number of active variables
static const double TARGET_SNR   = 1.0;    // currently always 1
// ----------------------------------------------------
//  MEMORY-CAP HELPERS
// ----------------------------------------------------
static double env_mem_cap_gb(const char* var) {
    const char* s = std::getenv(var);
    if (!s || !*s) return std::numeric_limits<double>::infinity();
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || v <= 0.0) return std::numeric_limits<double>::infinity();
    return v;
}

// bytes needed ~ n * (p + L) * 8   (TLARS)
//               ~ n * p * 8        (VD-LARS)
static long double tlars_bytes_needed(int n, int p, int L) {
    return 8.0L * (long double)n * (long double)(p + L);
}
static long double vdlars_bytes_needed(int n, int p) {
    return 8.0L * (long double)n * (long double)p;
}

// Peak RSS in bytes (child-local)
static inline uint64_t peak_rss_bytes() {
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__) && defined(__MACH__)
    // macOS: ru_maxrss is already in bytes
    return static_cast<uint64_t>(ru.ru_maxrss);
#else
    // Linux: ru_maxrss is in kilobytes
    return static_cast<uint64_t>(ru.ru_maxrss) * 1024ULL;
#endif
}

// ----------------------------------------------------
//  ENV CAPS
// ----------------------------------------------------
static void cap_blas_env() {
    setenv("OMP_NUM_THREADS",        "1",     1);
    setenv("OPENBLAS_NUM_THREADS",   "1",     1);
    setenv("MKL_NUM_THREADS",        "1",     1);
    setenv("VECLIB_MAXIMUM_THREADS", "1",     1);
    setenv("MKL_DYNAMIC",            "FALSE", 1);
    setenv("OMP_DYNAMIC",            "FALSE", 1);
    setenv("OMP_PROC_BIND",          "TRUE",  1);
}

static void progress(int current, int total,
                     int p, int m, int L, int rep)
{
    double pct = 100.0 * current / total;
    std::cerr << "\r[ " << int(pct) << "% ]  "
              << "p=" << p << ", m=" << m << ", L=" << L
              << ", rep=" << rep
              << std::flush;
}

// ----------------------------------------------------
//  SAFE PIPE I/O
// ----------------------------------------------------
struct ChildResult {
    double sec;      // runtime in seconds
    double peak_mb;  // peak RSS in MB
    int    ok;       // 1 = success, 0 = mem-cap or error
    int    code;     // 0 = ok, 1 = mem-cap, 2 = child error
};

static void write_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    size_t left = n;
    while (left) {
        ssize_t w = ::write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) continue;
            _exit(111);
        }
        left -= static_cast<size_t>(w);
        p    += w;
    }
}

// safe read: returns false if pipe closed early (child crash/OOM)
static bool read_all_safe(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    size_t left = n;
    while (left) {
        ssize_t r = ::read(fd, p, left);
        if (r <= 0) {
            return false;
        }
        left -= static_cast<size_t>(r);
        p    += r;
    }
    return true;
}

// ----------------------------------------------------
//  DATA GEN
// ----------------------------------------------------
static void gen_problem_null(int n, int p, uint64_t seed,
                             arma::mat& X, arma::vec& y)
{
    arma::arma_rng::set_seed(seed);
    X = arma::randn(n, p);
    for (arma::uword j=0; j<static_cast<arma::uword>(p); ++j) {
        arma::vec c = X.col(j);
        c -= arma::mean(c);
        double nrm = arma::norm(c,2);
        if (nrm > 1e-12) c /= nrm;
        X.col(j) = c;
    }
    y = arma::randn(n);
    y -= arma::mean(y);
}

static void gen_problem_signal(int n, int p, uint64_t seed,
                               int k_active, double target_snr,
                               arma::mat& X, arma::vec& y)
{
    // 1) generate and standardize X as before
    arma::arma_rng::set_seed(seed);
    X = arma::randn(n, p);
    for (arma::uword j=0; j<static_cast<arma::uword>(p); ++j) {
        arma::vec c = X.col(j);
        c -= arma::mean(c);
        double nrm = arma::norm(c,2);
        if (nrm > 1e-12) c /= nrm;
        X.col(j) = c;
    }

    // 2) build sparse beta with k_active ones
    arma::vec beta(p, arma::fill::zeros);
    k_active = std::min(k_active, p);
    for (int j = 0; j < k_active; ++j) {
        beta(j) = 1.0;   // or choose random indices if you prefer
    }

    arma::vec mu = X * beta;                // signal part
    double signal_var = arma::dot(mu, mu) / static_cast<double>(n);

    // fallback in degenerate case
    if (signal_var <= 0.0) {
        signal_var = 1.0;
    }

    double noise_var = signal_var / target_snr;
    double noise_sd  = std::sqrt(noise_var);

    // 3) noise with matching variance
    arma::arma_rng::set_seed(seed + 1);     // different seed for noise
    arma::vec eps = arma::randn(n) * noise_sd;

    y = mu + eps;
    y -= arma::mean(y);                     // optional: keep centered
}

// ----------------------------------------------------
//  CHILD WORKERS WITH MEMORY CAP + PEAK RSS
// ----------------------------------------------------
static void child_vd(int n, int p, int L, int T,
                     uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    ChildResult out{};
    out.sec     = 0.0;
    out.peak_mb = 0.0;
    out.ok      = 1;
    out.code    = 0;

    // MEMORY CHECK (VD)
    long double need_b = vdlars_bytes_needed(n,p);
    long double cap_b  = env_mem_cap_gb("VD_MEM_CAP_GB") * 1024LL*1024LL*1024LL;
    if (need_b > cap_b) {
        out.ok      = 0;
        out.code    = 1;  // mem-capped
        out.peak_mb = static_cast<double>(peak_rss_bytes()) / (1024.0*1024.0);
        write_all(out_fd,&out,sizeof(out));
        _exit(0);
    }

    arma::mat X; arma::vec y;
    if (USE_SIGNAL) {
        gen_problem_signal(n, p, seed, K_ACTIVE, TARGET_SNR, X, y);
    } else {
        gen_problem_null(n, p, seed, X, y);
    }

    using MatC = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    using Vec  = Eigen::VectorXd;
    Eigen::Map<const MatC> Xeig(X.memptr(),n,p);
    Eigen::Map<const Vec>  Yeig(y.memptr(),n);

    VDOptions opt;
    opt.T_stop      = T;
    opt.max_vd_proj = n;
    opt.standardize = false;

    VD_LARS vd(Xeig.data(),n,p,Yeig.data(),n,L,opt);

    auto t0 = std::chrono::high_resolution_clock::now();
    (void)vd.run(T);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.sec     = std::chrono::duration<double>(t1-t0).count();
    out.peak_mb = static_cast<double>(peak_rss_bytes()) / (1024.0*1024.0);

    write_all(out_fd,&out,sizeof(out));
    _exit(0);
}

static void child_tlars(int n, int p, int L, int T,
                        uint64_t seed, int out_fd) {
    pin_to_core(2);
    cap_blas_env();
    ChildResult out{};
    out.sec     = 0.0;
    out.peak_mb = 0.0;
    out.ok      = 1;
    out.code    = 0;

    // MEMORY CHECK (TLARS) — we now assume X_aug is the main cost
    long double need_b = tlars_bytes_needed(n, p, L); // 8 * n * (p + L)
    long double cap_b  = env_mem_cap_gb("TLARS_MEM_CAP_GB") * 1024LL*1024LL*1024LL;

    // Safety factor for workspaces + allocator overhead
    static const long double MEM_SAFETY = 1.3L;
    if ((need_b * MEM_SAFETY) > cap_b) {
        out.ok      = 0;
        out.code    = 1;  // mem-capped (predicted)
        out.peak_mb = static_cast<double>(peak_rss_bytes()) / (1024.0*1024.0);
        write_all(out_fd, &out, sizeof(out));
        _exit(0);
    }

    // 1) Allocate the full augmented design: n x (p + L)
    arma::mat X_aug(n, p + L);

    // Left block view: X (n x p), zero-copy wrapper
    arma::mat X_view(X_aug.memptr(), n, p, false, true);

    arma::vec y;

    // 2) Generate X (and possibly y with signal) directly into X_view
    if (USE_SIGNAL) {
        arma::arma_rng::set_seed(seed);
        X_view.randn();

        // Standardize columns of X_view
        for (arma::uword j = 0; j < static_cast<arma::uword>(p); ++j) {
            arma::vec c = X_view.col(j);
            c -= arma::mean(c);
            double nrm = arma::norm(c, 2);
            if (nrm > 1e-12) c /= nrm;
            X_view.col(j) = c;
        }

        // Build sparse beta with K_ACTIVE ones
        arma::vec beta(p, arma::fill::zeros);
        int k = std::min(K_ACTIVE, p);
        for (int j = 0; j < k; ++j)
            beta(j) = 1.0;

        arma::vec mu = X_view * beta;
        double signal_var = arma::dot(mu, mu) / static_cast<double>(n);
        if (signal_var <= 0.0)
            signal_var = 1.0;

        double noise_sd = std::sqrt(signal_var / TARGET_SNR);

        arma::arma_rng::set_seed(seed + 1);
        arma::vec eps = arma::randn(n) * noise_sd;

        y = mu + eps;
        y -= arma::mean(y);
    } else {
        // Null case: X and y independent Gaussians, centered
        arma::arma_rng::set_seed(seed);
        X_view.randn();
        for (arma::uword j = 0; j < static_cast<arma::uword>(p); ++j) {
            arma::vec c = X_view.col(j);
            c -= arma::mean(c);
            double nrm = arma::norm(c, 2);
            if (nrm > 1e-12) c /= nrm;
            X_view.col(j) = c;
        }

        arma::arma_rng::set_seed(seed + 1);
        y = arma::randn(n);
        y -= arma::mean(y);
    }

    // 3) Generate dummies directly into the right block: D (n x L)
    arma::mat D_view(X_aug.colptr(p), n, L, false, true);

    arma::arma_rng::set_seed(seed + 2);
    D_view.randn();
    for (arma::uword j = 0; j < static_cast<arma::uword>(L); ++j) {
        arma::vec c = D_view.col(j);
        c -= arma::mean(c);
        double nrm = arma::norm(c, 2);
        if (nrm > 1e-12) c /= nrm;
        D_view.col(j) = c;
    }

    // 4) Solve TLARS on X_aug
    TLARS_Solver tlars(X_aug, y, L, false, false, false);

    auto t0 = std::chrono::high_resolution_clock::now();
    tlars.executeStep(T, true);
    auto t1 = std::chrono::high_resolution_clock::now();

    out.sec     = std::chrono::duration<double>(t1 - t0).count();
    out.peak_mb = static_cast<double>(peak_rss_bytes()) / (1024.0 * 1024.0);

    write_all(out_fd, &out, sizeof(out));
    _exit(0);
}


// ----------------------------------------------------
//  A LA PYTHON: _measure(fn,...)
// ----------------------------------------------------
template <typename Fn>
static ChildResult run_in_child(Fn fn, int n,int p,int L,int T,uint64_t seed) {
    int fds[2];
    if (pipe(fds)!=0) std::abort();
    pid_t pid = fork();
    if (pid<0) std::abort();

    if (pid==0) { // child
        ::close(fds[0]);
        int out_fd = fds[1];
        if (out_fd!=3) {
            if (dup2(out_fd,3) < 0) _exit(113);
            ::close(out_fd);
            out_fd=3;
        }
        fn(n,p,L,T,seed,out_fd);
        _exit(0);
    }

    ::close(fds[1]);

    ChildResult r{};
    bool okread = read_all_safe(fds[0],&r,sizeof(r));
    ::close(fds[0]);

    int st=0;
    (void)waitpid(pid,&st,0);

    if (!okread) {
        // child crashed / OOM before sending a full struct
        r.sec     = 0.0;
        r.peak_mb = 0.0;
        r.ok      = 0;
        r.code    = 2; // child error
        return r;
    }

    if (!WIFEXITED(st) || WEXITSTATUS(st)!=0) {
        r.ok = 0;
        if (r.code==0) r.code=2; // child error
    }
    return r;
}

// ----------------------------------------------------
//  HELPER: timestamp + dirs + meta JSON
// ----------------------------------------------------
static std::string current_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

// ----------------------------------------------------
//  MAIN
// ----------------------------------------------------
int main() {
    // prepare output directory results/bench_p_<timestamp>/
    const std::string root = "experiments/results";
    const std::string ts   = current_timestamp();

    std::string mode = USE_SIGNAL ? "signal" : "null";

    // add sample size immediately after mode:  signal_n1000   or   null_n1000
    mode += "_n" + std::to_string(N_FIXED);

    const std::string run_name = "fig7_memory_runtime_benchmark" + mode;

    std::filesystem::path outdir = std::filesystem::path(root) / run_name;
    std::error_code ec;
    std::filesystem::create_directories(outdir, ec);
    if (ec) {
        std::cerr << "Failed to create directory " << outdir << ": " << ec.message() << "\n";
        return 1;
    }

    std::filesystem::path raw_path  = outdir / "raw.csv";
    std::filesystem::path meta_path = outdir / "meta.json";

    std::ofstream raw(raw_path);
    if (!raw) {
        std::cerr << "Failed to open raw.csv for writing at " << raw_path << "\n";
        return 1;
    }

    // CSV header
    raw << "n,p,m,L,rep,"
        << "time_vd,time_tl,"
        << "mem_vd_mb,mem_tl_mb,"
        << "ok_vd,ok_tl\n";

    int total = NUM_P * NUM_M * REPS;
    int cnt   = 0;

    // main sweep
    for (int pi=0; pi<NUM_P; ++pi) {
        for (int mi=0; mi<NUM_M; ++mi) {
            for (int rep=0; rep<REPS; ++rep) {

                int p  = P_LIST[pi];
                int m  = L_MULT[mi];
                int L  = m * p;
                uint64_t seed = BASE_SEED + 97*p + 7*m + 10000*rep;

                ChildResult vd = run_in_child(child_vd,    N_FIXED,p,L,T_SELECT,seed);
                ChildResult tl = run_in_child(child_tlars,N_FIXED,p,L,T_SELECT,seed);

                raw
                    << N_FIXED << ',' << p << ',' << m << ',' << L << ',' << rep << ','
                    << vd.sec      << ',' << tl.sec      << ','
                    << vd.peak_mb  << ',' << tl.peak_mb  << ','
                    << vd.ok       << ',' << tl.ok       << '\n';

                progress(++cnt, total, p, m, L, rep);
            }
        }
    }
    raw.close();
    std::cerr << "\nWrote raw CSV to: " << raw_path << "\n";

    // write meta.json
    std::ofstream meta(meta_path);
    if (!meta) {
        std::cerr << "Failed to open meta.json for writing at " << meta_path << "\n";
        return 1;
    }

    double vd_cap_gb    = env_mem_cap_gb("VD_MEM_CAP_GB");
    double tlars_cap_gb = env_mem_cap_gb("TLARS_MEM_CAP_GB");

    meta << "{\n";
    meta << "  \"run_name\": \"" << run_name << "\",\n";
    meta << "  \"timestamp\": \"" << ts << "\",\n";
    meta << "  \"N_FIXED\": " << N_FIXED << ",\n";
    meta << "  \"T_SELECT\": " << T_SELECT << ",\n";
    meta << "  \"REPS\": " << REPS << ",\n";
    meta << "  \"BASE_SEED\": " << BASE_SEED << ",\n";

    meta << "  \"P_LIST\": [";
    for (int i=0; i<NUM_P; ++i) {
        meta << P_LIST[i];
        if (i+1 < NUM_P) meta << ", ";
    }
    meta << "],\n";

    meta << "  \"L_MULT\": [";
    for (int i=0; i<NUM_M; ++i) {
        meta << L_MULT[i];
        if (i+1 < NUM_M) meta << ", ";
    }
    meta << "],\n";

    meta << "  \"VD_MEM_CAP_GB\": ";
    if (std::isinf(vd_cap_gb)) meta << "null";
    else meta << vd_cap_gb;
    meta << ",\n";

    meta << "  \"TLARS_MEM_CAP_GB\": ";
    if (std::isinf(tlars_cap_gb)) meta << "null";
    else meta << tlars_cap_gb;
    meta << "\n";

    meta << "}\n";
    meta.close();

    std::cerr << "Wrote meta.json to: " << meta_path << "\n";
    return 0;
}
