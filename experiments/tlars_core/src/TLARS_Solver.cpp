//TLARS_Solver.cpp
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <iostream>

#include "TLARS_Solver.hpp"

#ifdef TLARS_HAVE_OPENMP
  #include <omp.h>
#else
  // Fallbacks so code compiles without OpenMP
  static inline int omp_get_thread_num()   { return 0; }
  static inline int omp_get_num_threads()  { return 1; }
#endif
// ============================================================================
// Constructors
// ============================================================================

TLARS_Solver::TLARS_Solver(
    arma::mat& X,
    arma::vec& y,
    std::size_t num_dummies,
    bool normalize,
    bool intercept,
    bool verbose,
    const std::vector<std::size_t>& dropped_indices)
    : X_(&X), y_(y), currentStep_(0), dropped_indices_(dropped_indices),
        algorithm_("TLARS"), normalize_(normalize), intercept_(intercept),
        verbose_(verbose), is_connected_(true), num_dummies_(num_dummies),
        count_active_dummies_(0)
{
    std::size_t nrows = X.n_rows;
    std::size_t pcols = X.n_cols;

    // Consistency checks
    if (pcols < num_dummies_) {
        throw std::invalid_argument(
            "TLARS_Solver: num_dummies ("+ std::to_string(num_dummies_) +
            ") cannot exceed total columns (" + std::to_string(pcols) + ")"
        );
    }
    if (num_dummies_ == 0) {
        throw std::invalid_argument(
            "TLARS_Solver: num_dummies must be > 0 for TLARS. "
            "Use base LARS for standard regression and variable selection."
        );
    }

    // Calculuate original predictor count and dummy index start
    p_original_ = pcols - num_dummies_;
    dummy_start_idx_ = p_original_;

    // Effective sample size
    effective_n_ = nrows - (intercept_ ? 1 : 0);

    // Cache maximum active set size = min(available predictors, effective dof)
    max_actives_ = std::min(pcols - dropped_indices_.size(), effective_n_);

    // max_steps logic
    maxSteps_ = std::max(std::size_t(8 * max_actives_), std::size_t(128));
    if (maxSteps_> 10 * max_actives_) {
        logWarning(concatMsg("Auto maxSteps=", maxSteps_,
            " is very large (>", 10 * max_actives_,
            "). Numerical instability possible if p is huge."));
    }


    // CRITICAL: Apply preprocessing in-place if requested
    // WARNING: Input X is modified
    if (normalize_ || intercept_) {
        // Allocates and initializes meansx_, normsx_
        preprocess();
    } // else they remain uninitialized just to save memory

    // Initialize state variables
    r_ = y_;
    ssy_ = arma::dot(r_, r_);
    initializeInactives();

    // Reserve memory
    lambda_.reserve(maxSteps_ + 1);
    actions_.reserve(maxSteps_);
    RSS_.reserve(maxSteps_ + 1);
    R2_.reserve(maxSteps_ + 1);
    actives_.reserve(max_actives_);

    // Initialize diagnostics for smooth start at zero
    beta_ = arma::zeros<arma::vec>(pcols);
    betaPath_.set_size(pcols, 1);
    betaPath_.col(0).zeros();
    lambda_.emplace_back(0.0);
    RSS_.emplace_back(ssy_);
    R2_.emplace_back(0.0);
    DoF_.emplace_back(intercept_ ? 1 : 0);

    // Dummy tracking
    dummies_at_step_.reserve(maxSteps_ + 1);
    dummies_at_step_.push_back(0);

    // Initialize correlations
    initializeCorrelations();
}


void TLARS_Solver::initializeInactives() {
    inactives_.clear();
    inactives_.reserve(X_->n_cols - dropped_indices_.size());
    constexpr std::size_t HASH_THRESHOLD{10};

    if (dropped_indices_.size() < HASH_THRESHOLD) {
        // Linear search for small sets
        for (std::size_t j{0}; j < X_->n_cols; ++j) {
            if (std::find(dropped_indices_.begin(), dropped_indices_.end(), j)
                == dropped_indices_.end()) {
                    inactives_.push_back(j);
            }
        }
    } else {
        // Hash set with O(1) average lookup
        std::unordered_set<std::size_t> dropped_set(
            dropped_indices_.begin(),
            dropped_indices_.end()
        );

        for (std::size_t j{0}; j < X_->n_cols; ++j) {
            if (dropped_set.find(j) == dropped_set.end()) {
                inactives_.push_back(j);
            }
        }
    }
}


void TLARS_Solver::preprocess() {
    std::size_t nrows = X_->n_rows;
    std::size_t pcols = X_->n_cols;

    // 1. Intercept (true/false)
    if (intercept_) {
        meansx_.set_size(pcols);

        // Single parallel region
        #pragma omp parallel
        {
            // Phase 1: Compute means
            #pragma omp for schedule(static)
            for (std::size_t j = 0; j < pcols; ++j) {
                meansx_[j] = arma::mean(X_->col(j));
            }
            // Implicit barrier ensures means computed before centering

            // Phase 2: Center columns
            #pragma omp for schedule(static)
            for (std::size_t j = 0; j < pcols; ++j) {
                X_->col(j) -= meansx_[j];
            }
        }

        // Center y
        mu_y_ = arma::mean(y_);
        y_ -= mu_y_;
    }

    // 2. Normalization (L2 scaling)
    if (normalize_) {
        normsx_.set_size(pcols);
        std::vector<std::vector<std::size_t>> thread_dropped;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();

            #pragma omp single
            {
                thread_dropped.resize(nthreads);
            }

            #pragma omp for schedule(static) nowait
            for (std::size_t j = 0; j < pcols; ++j) {
                // Compute L2 norm
                double norm_j = arma::norm(X_->col(j), 2);

                if (norm_j / std::sqrt(static_cast<double>(nrows)) < eps_) {
                    normsx_[j] = eps_ * std::sqrt(static_cast<double>(nrows));

                    // check if already dropped
                    if (std::find(dropped_indices_.begin(),
                                  dropped_indices_.end(), j) == dropped_indices_.end()) {
                        thread_dropped[tid].push_back(j);
                    }

                } else {
                    normsx_[j] = norm_j;
                }

                // scale column j
                X_->col(j) /= normsx_[j];
            }
        }

        for (const auto& local : thread_dropped) {
            for (std::size_t j : local) {
                dropped_indices_.push_back(j);
                logWarning(concatMsg("Column ", j, " has variance < eps_; dropped"));
            }
        }
    }
}


void TLARS_Solver::restore(arma::mat& X, arma::vec& y) const {
    std::size_t pcols = X.n_cols;

    if (normalize_ && normsx_.n_elem != pcols) {
        throw std::runtime_error(
            "TLARS_Solver::restore: normsx_ dimension mismatch"
        );
    }

    if (intercept_ && meansx_.n_elem != pcols) {
        throw std::runtime_error(
            "TLARS_Solver::restore: meansx_ dimension mismatch"
        );
    }

    if (y.n_elem != X.n_rows) {
        throw std::runtime_error(
            "TLARS_Solver::restore: y length mismatch with X rows"
        );
    }

    if (normalize_) {
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < pcols; ++j) {
            X.col(j) *= normsx_[j];
        }
    }

    if (intercept_) {
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < pcols; ++j) {
            X.col(j) += meansx_[j];
        }
        y += mu_y_;
    }
}

// ============================================================================
// Validation Helper
// ============================================================================

void TLARS_Solver::validateConnected() const {
    if (!is_connected_ || X_ == nullptr) {
        throw std::runtime_error(
            "TLARS_Solver: Solver is not connected to design matrix. "
            "Call reconnect(X) after loading from file."
        );
    }

    // Additional sanity checks
    if (X_->n_cols != beta_.n_elem) {
        throw std::runtime_error(
            "TLARS_Solver: Internal state corrupted - X columns != beta size"
        );
    }

    if (X_->n_rows == 0) {
        throw std::runtime_error(
            "TLARS_Solver: Design matrix has zero rows"
        );
    }
}

// ============================================================================
// Logger/Status information
// ============================================================================

void TLARS_Solver::logMsg(const std::string& msg) const {
    if (verbose_) {
        std::cout << msg << std::endl;
    }
}


void TLARS_Solver::logWarning(const std::string& msg) const {
    logMsg("[WARNING] " + msg);
}


void TLARS_Solver::logInfo(const std::string& msg) const {
    logMsg("[Info] " + msg);
}


// ============================================================================
// Core Algorithm: T-LARS Path
// ============================================================================

void TLARS_Solver::executeStep(std::size_t T_stop, bool early_stop)
{
    validateConnected();

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (count_active_dummies_ < T_stop || !early_stop)
    ) {

        // 1. Find variable with maximum correlation
        std::size_t j_star = findMaxAbsCorrelation();
        double C = std::abs(correlations_[j_star]);

        if (C < 100 * eps_) {
            logInfo("T-LARS terminated: Correlation < threshold.");
            break;
        }

        // 2. Add variable to active set
        addToActiveSet(j_star);
        if (isDummy(j_star)) {
            ++count_active_dummies_;
            logInfo(concatMsg(
                "Dummy ", j_star - dummy_start_idx_, " entered (",
                count_active_dummies_, "/", num_dummies_, ", index: ", j_star, ")"));
        }
        updateCholesky();

        // 3. Compute sign vector for actives
        arma::vec signs_A = computeSignVector(correlations_, actives_);

        // 4. Solve for equiangular direction
        arma::vec w_A = solveEquiangularDirection(signs_A);

        // 5. Construct equiangular vector in y space
        arma::vec u = computeEquiangularVector(w_A);

        // 6. Compute LARS step size
        double gamma = computeGamma(correlations_, u, C);

        // 7. Update beta (only actives) and residual
        updateBeta(gamma, w_A);
        updateResidual(gamma, u);

        // 8. Update correlations incrementally
        updateCorrelations(gamma, u);

        // 9. Update diagnostics
        betaPath_.insert_cols(betaPath_.n_cols, beta_);
        lambda_.push_back(C);

        double rss = arma::dot(r_, r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / ssy_);

        updateDummyTracking();
        DoF_.push_back((intercept_ ? 1: 0) + actives_.size() - count_active_dummies_);
        currentStep_++;
    }

    if (verbose_) {
        if (currentStep_ >= maxSteps_) {
            logWarning("Emergency path cap reached.");
        } else if (inactives_.empty()) {
            logInfo("No inactive predictors remain.");
        } else if (actives_.size() >= effective_n_) {
            logInfo("Reached effective rank limit.");
        }  else if ((count_active_dummies_ >= T_stop) && early_stop) {
            logInfo("T-LARS T/FDR stop triggered.");
        }
    }
}


// ============================================================================
// Helper Implementation
// ============================================================================

void TLARS_Solver::initializeCorrelations() {

    std::size_t ncols = X_->n_cols;
    correlations_ = arma::zeros<arma::vec>(ncols);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_[j] = arma::dot(X_->col(j), r_);
    }

}


void TLARS_Solver::updateCorrelations(double gamma, const arma::vec& u) {

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_[j] -= gamma * arma::dot(X_->col(j), u);
    }

}


std::size_t TLARS_Solver::findMaxAbsCorrelation() const {

    if (inactives_.empty())
        throw std::runtime_error("findMaxAbsCorrelation: No inactive variables remaining");
    double max_corr = 0.0;
    std::size_t max_idx = inactives_[0];
    for (std::size_t j : inactives_) {
        double abs_corr = std::abs(correlations_[j]);
        if (abs_corr > max_corr) {
            max_corr = abs_corr;
            max_idx = j;
        }
    }
    return max_idx;

}


arma::vec TLARS_Solver::computeSignVector(const arma::vec& correlations,
    const std::vector<std::size_t>& actives) const {

    arma::vec s(actives.size());
    for (std::size_t k = 0; k < actives.size(); ++k) {
        s[k] = (correlations[actives[k]] >= 0 ? 1.0 : -1.0);
    }
    return s;

}


arma::vec TLARS_Solver::solveEquiangularDirection(const arma::vec& s_A) {

    std::size_t m = cholG_.n_rows;

    if (m == 1) {
        // Scalar solve
        arma::vec GAi_s(1);
        double cholG_00 = cholG_.at(0, 0);
        GAi_s[0] = s_A[0] / (cholG_00 * cholG_00);
        A_A_ = 1.0 / std::sqrt(s_A[0] * GAi_s[0]);
        return A_A_ * GAi_s;

    } else {
        // Normal solve
        arma::vec GAi_s = arma::solve(arma::trimatu(cholG_),
            arma::solve(arma::trimatl(cholG_.t()), s_A));
        A_A_ = 1.0 / std::sqrt(arma::as_scalar(s_A.t() * GAi_s));
        return A_A_ * GAi_s;
    }

}


double TLARS_Solver::computeGamma(const arma::vec& c, const arma::vec& u, double C) const {

    // Check if active set is exhausted
    if (actives_.size() >= max_actives_) {
        return C / A_A_;
    }

    // Find next step to variable
    double gamma = std::numeric_limits<double>::max();

    for (std::size_t j : inactives_) {
        double a_j = arma::dot(X_->col(j), u);

        // Two candidate gamma values
        double g1 = (C - c[j]) / (A_A_ - a_j);
        double g2 = (C + c[j]) / (A_A_ + a_j);

        // Take minimum positive, finite value
        if (g1 > eps_ && std::isfinite(g1) && g1 < gamma)
            gamma = g1;
        if (g2 > eps_ && std::isfinite(g2) && g2 < gamma)
            gamma = g2;
    }

    if (gamma == std::numeric_limits<double>::max()) {
        logWarning("No valid gamma found, using fallback");
        return C / A_A_;
    }

    return gamma;

}


void TLARS_Solver::updateBeta(double gamma, const arma::vec& w_A) {

    std::size_t m = actives_.size();
    #pragma omp parallel for schedule(static) if(m > 100)
    for (std::size_t k = 0; k < m; ++k) {
        beta_(actives_[k]) += gamma * w_A[k];
    }

}


void TLARS_Solver::updateResidual(double gamma, const arma::vec& u) {
    r_ -= gamma * u;
}


void TLARS_Solver::addToActiveSet(std::size_t j_star) {

    auto it = std::find(actives_.begin(), actives_.end(), j_star);
    if (it != actives_.end()) return;

    actives_.emplace_back(j_star);
    actions_.emplace_back(static_cast<int>(j_star));
    ++num_additions_;
    inactives_.erase(std::remove(inactives_.begin(), inactives_.end(), j_star),
                     inactives_.end());
}


arma::vec TLARS_Solver::computeEquiangularVector(const arma::vec& w_A) const {
    arma::vec u(X_->n_rows, arma::fill::zeros);
    for (std::size_t k = 0; k < actives_.size(); ++k)
        u += w_A[k] * X_->col(actives_[k]);
    return u;
}


arma::mat TLARS_Solver::computeGramMatrix() const {
    std::size_t m{actives_.size()};
    arma::mat G(m, m);

    #pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i; j < m; ++j) {
            // Symmetric, compute upper triangle
            G.at(i, j) = arma::dot(X_->col(actives_[i]), X_->col(actives_[j]));
            // Mirror
            if (i != j) G.at(j, i) = G.at(i, j);
        }
    }
    return G;
}


void TLARS_Solver::updateCholesky() {
    std::size_t m = actives_.size();

    if (m == 1) {
        // First variable
        double norm_sq = arma::dot(X_->col(actives_[0]), X_->col(actives_[0]));
        cholG_.set_size(1, 1);
        cholG_(0, 0) = std::sqrt(norm_sq);
        return;
    }

    // New variable index (last on added)
    std::size_t j_new = actives_.back();

    // Step 1: Compute cross-products with existing active variables
    // X_new^T * X_A -> size: (m-1 x m-1)
    arma::vec cross_prod(m - 1);
    for (std::size_t k = 0; k < m - 1; ++k) {
        cross_prod[k] = arma::dot(X_->col(j_new), X_->col(actives_[k]));
    }

    // Step 2: Solve R_old^T * r = cross_prod for r (forward substitution)
    // cholG_ is upper triangular, so R^T is lower triangular
    arma::vec r = arma::solve(arma::trimatl(cholG_.t()), cross_prod);

    // Step 3: Compute new diagonal element
    double norm_sq_new = arma::dot(X_->col(j_new), X_->col(j_new));
    double r_pp_sq = norm_sq_new - arma::dot(r, r);

    // Numerical tolerance for positive definiteness check
    const double tol = 1e-10;

    // Check for numerical singularity
    if (r_pp_sq < tol) {
        // Variable j_new is nearly collinear with active set
        r_pp_sq = std::max(r_pp_sq, tol);

        logWarning(concatMsg("Variable ", j_new, " is collinear (r_pp_sq = ",
                    r_pp_sq, "). Dropping permanently."));

        actives_.pop_back();

        dropped_indices_.push_back(j_new);

        actions_.push_back(-static_cast<int>(j_new));

        return;
    }
    double r_pp = std::sqrt(r_pp_sq);

    // Step 4: Extend Cholesky factor
    arma::mat new_cholG(m, m, arma::fill::zeros);
    // Copy previous upper-triangular part
    new_cholG.submat(0, 0, m - 2, m - 2) = cholG_;
    // Fill new column in upper triangular
    for (std::size_t k = 0; k < m - 1; ++k) {
        new_cholG(k, m - 1) = r[k];
    }
    // Fill new diagonal
    new_cholG(m - 1, m - 1) = r_pp;
    cholG_ = std::move(new_cholG);
}


std::size_t TLARS_Solver::removeFromActiveSet(std::size_t exit_pos) {
    throw std::logic_error(
        "TLARS_Solver::removeFromActiveSet() not supported for TLARS "
        "(use TLASSO for lasso-style removal)"
    );
    return 0;
}


// ============================================================================
// Getters and Dagnostics
// ============================================================================

arma::vec TLARS_Solver::getBeta() const {

    if (normalize_ && normsx_.n_elem > 0) {
        return beta_ / normsx_;
    }
    return beta_;

}


arma::mat TLARS_Solver::getBetaPath() const {

    if (normalize_ && normsx_.n_elem > 0) {
        arma::mat beta_orig = betaPath_;
        for (std::size_t k = 0; k < beta_orig.n_cols; ++k) {
            beta_orig.col(k) /= normsx_;
        }
        return beta_orig;
    }
    return betaPath_;

}


std::vector<double> TLARS_Solver::getRSS() const { return RSS_; }


std::vector<double> TLARS_Solver::getR2() const { return R2_; }


std::vector<double> TLARS_Solver::getCp() const {
    // Recompute Cp if path statistics are valid
    if (!RSS_.empty()) {
        const_cast<TLARS_Solver*>(this)->updateCp();
    }
    return Cp_;
}


void TLARS_Solver::updateDummyTracking() {
    dummies_at_step_.push_back(count_active_dummies_);
}


std::vector<std::size_t> TLARS_Solver::getDoF() const {
    // DoF[i] = intercept + (actives - dummies) per step
    return DoF_;
}


arma::vec TLARS_Solver::getResiduals() const { return r_; }


std::vector<std::size_t> TLARS_Solver::getActivePredictorIndices() const {
    std::vector<std::size_t> res;
    res.reserve(actives_.size());
    for (std::size_t idx : actives_) {
        if (idx < dummy_start_idx_) {
            res.push_back(idx);
        }
    }
    return res;
}


std::vector<std::size_t> TLARS_Solver::getActiveDummyIndices() const {
    std::vector<std::size_t> res;
    res.reserve(count_active_dummies_);
    for (std::size_t idx : actives_) {
        if (idx >= dummy_start_idx_) {
            // dummies are always at or after dummy_start_idx_
            res.push_back(idx);
        }
    }
    return res;
}


arma::vec TLARS_Solver::predict(const arma::mat& X_new) const {
    // Predict function: for TLARS, typically use only real predictors
    // Accepts new data with same number of columns as original predictors
    std::size_t p_new = X_new.n_cols;
    if (p_new != p_original_) {
        throw std::runtime_error(
            "TLARS_Solver::predict: X_new dimension mismatch: must have n_cols = p_original_"
        );
    }

    arma::vec y_hat = arma::zeros<arma::vec>(X_new.n_rows);
    arma::vec beta_real = beta_.head(p_original_);

    // Replicate normalization logic as appropriate
    for (std::size_t j = 0; j < p_original_; ++j) {
        double beta_coef = beta_real[j];
        if (std::abs(beta_coef) > eps_) {
            if (normalize_ && intercept_)
                y_hat += beta_coef * ((X_new.col(j) - meansx_[j]) / normsx_[j]);
            else if (normalize_)
                y_hat += beta_coef * (X_new.col(j) / normsx_[j]);
            else if (intercept_)
                y_hat += beta_coef * (X_new.col(j) - meansx_[j]);
            else
                y_hat += beta_coef * X_new.col(j);
        }
    }

    // Adjustment for intercept
    if (intercept_) {
        double correction = mu_y_;
        for (std::size_t j = 0; j < p_original_; ++j) {
            double norm_fac = (normalize_ ? normsx_[j] : 1.0);
            correction -= meansx_[j] * beta_real[j] / norm_fac;
        }
        y_hat += correction;
    }
    return y_hat;
}


// ============================================================================
// Cp statistics (dummy-aware for TLARS)
// ============================================================================

void TLARS_Solver::updateCp(const std::vector<double>& rss_path,
                            std::vector<double>& cp_out) const
{
    cp_out.clear();
    const std::vector<std::size_t>& dof_adj = DoF_; // dummy aware
    if (rss_path.empty() || dof_adj.empty()) {
        return;
    }

    const std::size_t n = X_->n_rows;
    double rss_final = rss_path.back(); // Estimate S^2 from final model
    std::size_t df_final = dof_adj.back();

    // Check validaty conditions to avoid zero division or negative variance
    if (df_final >= n || rss_final <= 0) {
        cp_out.assign(rss_path.size(), std::numeric_limits<double>::quiet_NaN());
        return;
    }

    // Estimate residual variance from final model
    double resvar = rss_final / static_cast<double>(n - df_final);

    // Compute Cp for every step in the provided path
    cp_out.reserve(rss_path.size());
    for (std::size_t k = 0; k < rss_path.size(); ++k) {
        double cp_val = rss_path[k] / resvar - double(n) + 2.0 * double(dof_adj[k]);
        cp_out.emplace_back(cp_val);
    }
}


void TLARS_Solver::updateCp() {
    updateCp(RSS_, Cp_);
}


void TLARS_Solver::updateCp(double sigma_hat_sq, std::vector<double>& cp_out) const {
    cp_out.clear();
    const std::vector<std::size_t>& dof_adj = DoF_;
    std::size_t n = X_->n_rows;
    for (std::size_t k = 0; k < RSS_.size(); ++k) {
        double cp_k = RSS_[k] / sigma_hat_sq - static_cast<double>(n) +
                        2.0 * static_cast<double>(dof_adj[k]);
        cp_out.push_back(cp_k);
    }
}


// ============================================================================
// Serialization
// ============================================================================

void TLARS_Solver::save(const std::string& filename) const {
    std::ofstream os(filename, std::ios::binary);

    if (!os.is_open()) {
        throw std::runtime_error("TLARS_Solver::save: Can't open file " + filename);
    }
    try {
        cereal::PortableBinaryOutputArchive oarchive(os);
        oarchive(*this);
    } catch (const std::exception& e){
        throw std::runtime_error("TLARS_Solver::save: Serialization failed - " +
            std::string(e.what())
        );
    }
}


TLARS_Solver TLARS_Solver::load(const std::string& filename, arma::mat& X) {
    TLARS_Solver tlars; // new obj via no-args constructor (is_connected = false)

    // 1. Deserialize from file with error handling
    {
        std::ifstream is(filename, std::ios::binary);
        if (!is.is_open())
            throw std::runtime_error("TLARS_Solver::load: Can't open file " + filename);
        try {
            cereal::PortableBinaryInputArchive iarchive(is);
            iarchive(tlars);
        } catch (const std::exception& e) {
            throw std::runtime_error("TLARS_Solver::load: Deserialization failed - " +
                std::string(e.what())
            );
        }
    }

    // 2. Validate loaded integrity
    std::size_t expected_cols = tlars.p_original_ + tlars.num_dummies_;
    if (X.n_cols != expected_cols) {
        throw std::runtime_error("TLARS_Solver::load: X dimension mismatch");
    }
    if (tlars.beta_.n_elem == 0) {
        throw std::runtime_error("TLARS_Solver::load: Corrupted state (empty beta_)");
    }
    if (tlars.algorithm_.empty()) {
        throw std::runtime_error("TLARS_Solver::load: Corrupted state (empty algorithm_)");
    }

    // 3. Reconnect with error handling
    {
        tlars.reconnect(X);
        tlars.logInfo(tlars.concatMsg("TLARS_Solver loaded from '", filename, "'"));
    }

    return tlars;
}


void TLARS_Solver::reconnect(arma::mat& X) {
    if (X.n_cols != beta_.n_elem)
        throw std::runtime_error("TLARS_Solver::reconnect: Column mismatch");
    X_ = &X;
    is_connected_ = true;
}
