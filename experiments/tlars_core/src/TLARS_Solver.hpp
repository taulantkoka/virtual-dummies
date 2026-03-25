//TLARS_Solver.hpp
#ifndef TLARS_SOLVER_HPP
#define TLARS_SOLVER_HPP

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <armadillo>
#include <cereal/archives/portable_binary.hpp>
#include "arma_cereal.hpp"

/**
 * @brief Terminating Least Angle Regression (T-LARS) solver with dummy variable support
 *
 * @details Implements T-LARS algorithm as selector for T-Rex based FDR-controlled variable
 *  selection:
 *   - Augmented design matrix: X_augmented = [X_original | X_dummies]
 *   - Early stopping: Terminate when T_stop dummy variables enter active set
 *   - Adjusted DoF: Exclude dummies from degrees of freedom
 *   - Incremental correlation tracking O(n) updates
 *   - Rank-1 Cholesky updates for Gram matrix factorization
 *   - Full solution path tracking with model selection diagnostics
 *   - Warm-start capability via serialization
 */
class TLARS_Solver {
protected:
    // ============================================================================
    // Protected Members
    // ============================================================================

    // Data references
    arma::mat* X_{nullptr};             // Pointer to augmented matrix X = [X_orig | X_dummies]
    arma::vec y_{};                     // Response (centered if intercept=true)

    // Algorithm state
    arma::vec r_{};                     // Current residual vector
    arma::vec beta_{};                  // Current coefficient vector
    arma::vec correlations_{};          // Correlation between X and residual

    /**
     * @brief Monotonic counter of variable additions and never decremented
     *
     * @details Tracks the total number of additions across the algorithm's lifetime.
     *  - LARS: num_additions == actives_.size() (just semantic)
     *  - LASSO: num_additions >= actives_.size() (possible removals)
     */
    std::size_t num_additions_{0};

    // Solution path
    arma::mat betaPath_{};              // Each column is beta at step k
    std::vector<double> lambda_{};      // Max correlation at each step
    std::vector<int> actions_{};        // Indices of entering variables
    std::vector<double> RSS_{};         // Residual sum of squares for each path step
    std::vector<double> R2_{};          // R² at each step
    std::vector<double> Cp_{};          // Mallow's Cp at each step (computed at path end)
    std::vector<std::size_t> DoF_{};    // Degrees of freedom at each step (excluding dummies)

    // Preprocessing state
    bool normalize_{true};              // L2 normalize columns
    bool intercept_{true};              // Center data
    bool verbose_{false};               // Status logging
    arma::vec meansx_{};                // Column means
    arma::vec normsx_{};                // Column L2 norms
    double mu_y_{0.0};                  // y mean

    // Active set administration
    std::vector<std::size_t> actives_{};         // Currently active predictors and dummies
    std::vector<std::size_t> inactives_{};       // Currently inactive predictors and dummies
    std::vector<std::size_t> dropped_indices_{}; // Excluded predictors

    // Cholesky decomposition
    arma::mat cholG_{};           // Upper triangular Cholesky factor
    double A_A_{};                // Equiangular normalization factor

    // Configuration
    std::size_t currentStep_{0};     // Current step in solution path
    std::size_t max_actives_{0};     // Maximum active variables
    std::size_t maxSteps_{0};        // Max steps
    double ssy_{0.0};                // Total sum of squares of y
    double resvar_{0.0};             // Residual variance
    std::string algorithm_{"TLARS"}; // Algorithm identifier
    const double eps_{std::numeric_limits<double>::epsilon()};  // eps as machine precision
    std::size_t effective_n_{0};     // n - (intercept ? 1 : 0)

    // Serialization state
    bool is_connected_{false};     // X_ connection status

    // ============================================================================
    // T-LARS Specific Members
    // ============================================================================

    std::size_t num_dummies_{0};                 // Total dummy variables in X
    std::size_t count_active_dummies_{0};        // Currently active dummies
    std::size_t p_original_{0};                  // Original predictor count (before augmentation)
    std::size_t dummy_start_idx_{0};             // First dummy index = p_original
    std::vector<std::size_t> dummies_at_step_{}; // Dummy count at each step

    // ============================================================================
    // Internal Helpers
    // ============================================================================

    /**
     * @brief Compute sign vector for active correlations
     *
     * @param correlations Current correlation vector X^T * r
     * @param actives Indices of active variables
     *
     * @return Sign vector: s_j = sign(corr[j]) for j in actives
     */
    arma::vec computeSignVector(const arma::vec& correlations,
                                const std::vector<std::size_t>& actives) const;


    /**
     * @brief Solve for equiangular direction w_A = A_A * G_A^{-1} * s_A
     *
     * @param s_A Sign vector for active set
     *
     * @return Equiangular direction in coefficient space
     *
     * @note Updates A_A_ as side effect
     */
    arma::vec solveEquiangularDirection(const arma::vec& s_A);


    /**
     * @brief Compute LARS step size to next variable entry
     *
     * @param c Current correlation vector
     * @param u Equiangular vector in response space
     * @param C Maximum absolute correlation
     *
     * @return Step size gamma
     */
    double computeGamma(const arma::vec& c, const arma::vec& u, double C) const;


    /**
     * @brief Rank-1 Cholesky update for new active variable
     *
     * @details Updates cholG_ from (k-1)×(k-1) to k×k when variable enters
     * Complexity: O(k^2)
     */
    virtual void updateCholesky();


    /**
     * @brief Initialize inactive set (all non-dropped variables)
     *
     * @details Uses hash set for O(1) lookup when dropped_indices is large
     */
    void initializeInactives();


    /**
     * @brief Update coefficient vector: beta += gamma * w_A
     *
     * @param gamma Step size
     *
     * @param w_A Equiangular direction
     */
    void updateBeta(double gamma, const arma::vec& w_A);


    /**
     * @brief Update residual: r -= gamma * u
     *
     * @param gamma Step size
     * @param u Equiangular vector in response space
     */
    void updateResidual(double gamma, const arma::vec& u);


    /**
     * @brief Add variable to active set
     *
     * @param j_star Index of variable to activate
     *
     * @details Updates actives_, inactives_, actions_, and num_additions_
     */
    void addToActiveSet(std::size_t j_star);


    /**
     * @brief Remove variable from active set (LASSO only)
     *
     * @param exit_pos Position in actives_ vector to remove
     *
     * @return Index of removed variable
     *
     * @note Base LARS throws error; overridden in LASSO for Cholesky downdate
     */
    virtual std::size_t removeFromActiveSet(std::size_t exit_pos);

    // ============================================================================
    // Correlation management
    // ============================================================================

    /**
     * @brief Initialize correlations: c = X^T * r
     *
     * @details Only computes for inactive variables
     */
    void initializeCorrelations();


    /**
     * @brief Incrementally update correlations: c -= gamma * X^T * u
     * @param gamma Step size
     * @param u Equiangular vector
     * @details O(n*p_inactive) update instead of O(n*p) recomputation
     */
    void updateCorrelations(double gamma, const arma::vec& u);


    /**
     * @brief Find variable with maximum absolute correlation
     *
     * @return Index j* = argmax_j |c_j| for j in inactives
     *
     * @throws std::runtime_error if inactives is empty
     */
    std::size_t findMaxAbsCorrelation() const;

    // ============================================================================
    // Computational helper
    // ============================================================================

    /**
     * @brief Compute equiangular vector: u = X_A * w_A
     *
     * @param w_A Equiangular direction in coefficient space
     *
     * @return Equiangular vector in response space (length n)
     */
    arma::vec computeEquiangularVector(const arma::vec& w_A) const;


    /**
     * @brief Compute Gram matrix for active set: G_A = X_A^T X_A
     *
     * @details
     *  - Provided for completness and potential debug diagnostics only
     *  - The method is NEVER used in the pathwise algorithm
     *  - Standard LARS/LASSO/ElasticNet operate on Cholesky and Givens rotations
     *
     * @return Symmetric (k × k) matrix (k = current active set size)
     *
     * @warning Only use for diagnostics, debugging, or other analysis.
     */
    virtual arma::mat computeGramMatrix() const;


    /** @brief Internal preprocessing applied during construction
     *
     * @details Modifies X_ and y_ in-place if normalize_ and intercept_ flags are true.
     */
    void preprocess();


    /**
     * @brief Compute Mallow's Cp for all steps (big model)
     *
     * @param rss_path Vector for rss path history
     * @param cp_out Mallows Cp computed based on full model
     *
     * @details Uses final model residual variance as sigma^2 estimate
     */
    void updateCp(const std::vector<double>& rss_path, std::vector<double>& cp_out) const;


    /**
     * @brief Validate that solver is connected and ready for computation
     *
     * @throws std::runtime_error if X_ is null or dimension mismatch
     */
    void validateConnected() const;


    /**
     * @brief Update dummy tracking after adding a variable to active set
     */
    virtual void updateDummyTracking();


    // ============================================================================
    // Logging
    // ============================================================================

    /**
     * @brief Verbosity-controlled logging method
     *
     * @param msg Message to log (if verbose_ is true)
     *
     * @details Virtual to allow overrides for derived classes or wrappers
     *  to override output behavior (e.g., Python logger, R Rcout)
     *
     */
    virtual void logMsg(const std::string& msg) const;

    /**
     * @brief Log warning message with [WARNING] prefix added
     *
     * @param msg Warning message
     */
    void logWarning(const std::string& msg) const;

    /**
     * @brief Log info message with [INFO] prefix added
     *
     * @param msg Info message
     */
    void logInfo(const std::string& msg) const;

    /**
     * @brief Concatenate multiple arguments into a single string
     *
     * @tparam Args Variadic template for any printable types
     *  (int, double, string, etc.)
     * @param args Arguments to concatenate using string insertion operator<<
     *
     * @return Concatenated string
     *
     * @note Uses Cpp17 fold expression for efficient stream-based concatenation.
     *  All arguments must be printable to std::ostream.
     *
     * @example Example: concatMsg("Step ", 5, " RSS = ", 123.45)
     */
    template<typename... Args>
    std::string concatMsg(Args&&... args) const {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }

    // ============================================================================
    // Protected constructor for derived classes -> sets algorithm type
    // ============================================================================

    /**
     * @brief Protected constructor for inheritance delegation through hierarchy
     *
     * @param algorithm Algorithm name: "TLARS", "TLASSO", "TENET"
     *
     * @details Used by derived classes (TLASSO, TENET) to route initialization
     *  through the inheritance chain while ensuring consistent member setup.
     */
    explicit TLARS_Solver(const std::string& algorithm)
        : X_(nullptr), currentStep_(0), maxSteps_(0),
            algorithm_(algorithm), effective_n_(0), is_connected_(false),
            num_dummies_{0}, count_active_dummies_{0}, p_original_{0},
            dummy_start_idx_{0} {}


public:
    // ============================================================================
    // Constructors & Destructor
    // ============================================================================

    /**
     * @brief Default constructor for deserialization and container use.
     */
    TLARS_Solver() : TLARS_Solver("TLARS") {}

    /**
     * @brief Main T-LARS constructor
     *
     * @param X Augmented design matrix (n x (p + num_dummies))
     *          Structure: [X_original (n x p) | X_dummies (n x num_dummies)]
     * @param y Response vector (n x 1)
     * @param num_dummies Number of dummy columns at the end of X
     * @param normalize If true, L2 normalize all columns (including dummies). Default true.
     * @param intercept If true, center X (including dummies) and y. Default true.
     * @param verbose If true, print diagnostic messages. Default false.
     * @param dropped_indices Pre-identified zero-variance columns to exclude
     *
     * @throws std::invalid_argument if num_dummies = 0 or configuration invalid
     *
     * @warning X must be pre-augmented before calling constructor
     */
    TLARS_Solver(arma::mat& X,
                 arma::vec& y,
                 std::size_t num_dummies,
                 bool normalize = true,
                 bool intercept = true,
                 bool verbose = false,
                 const std::vector<std::size_t>& dropped_indices = {});

    /** @brief Virtual destructor */
    virtual ~TLARS_Solver() = default;

    // ============================================================================
    // Core Algorithm Execution
    // ============================================================================


    /**
     * @brief Execute T-LARS with early stopping based on dummy variable threshold
     *
     * @param T_stop Dummy threshold for early stopping
     *               - If T_stop > 0 and early_stop=true: Stop after T_stop dummies enter
     *               - If T_stop = 0 or early_stop=false: Full path until natural termination
     * @param early_stop If true, terminate when T_stop reached; if false, ignore T_stop
     *
     * @details Three execution modes:
     *   1. **Early stopping mode**: T_stop > 0, early_stop = true
     *      -> Terminate when count_active_dummies_ >= T_stop
     *   2. **Full path mode**: T_stop = 0 or early_stop = false
     *      -> Run until all variables exhausted or correlation < tolerance
     *   3. **Hybrid mode**: T_stop > 0, early_stop = false
     *      -> Track dummy count but don't stop early (for diagnostics)
     *
     * @note Uses incremental correlation updates and rank-1 Cholesky updates for efficiency
     */
    virtual void executeStep(std::size_t T_stop = 0, bool early_stop = true);


    // ============================================================================
    // Prediction
    // ============================================================================

    /**
     * @brief Predict response for new data using current T-LARS coefficients
     *
     * @param X_new New design matrix (n_new × p_original)
     *              Must contain ONLY original predictors (no dummies)
     *
     * @return Predicted values (n_new × 1)
     *
     * @throws std::runtime_error if X_new.n_cols != p_original_
     *
     * @note Automatically applies centering/normalization matching model fit
     */
    arma::vec predict(const arma::mat& X_new) const;

    // ============================================================================
    // Data Restoration
    // ============================================================================

    /**
     * @brief Restore augmented X and y to original scale
     *
     * @param X Augmented design matrix to restore (in-place)
     * @param y Response vector to restore (in-place)
     *
     * @details Applies inverse transformations:
     *  - If normalized: X.col(j) *= normsx_[j]
     *  - If intercept: X.col(j) += meansx_[j], y += mu_y_
     *
     * @throws std::runtime_error if dimensions mismatch
     *
     * @note Call only after algorithm completion
     */
    void restore(arma::mat& X, arma::vec& y) const;


    /**
     * @brief Public wrapper to compute Cp using internal state.
     *
     * @details Standard interface for LARS and LASSO solvers.
     */
    void updateCp();

    /**
     * @brief Public wrapper to compute Cp for model comparison, given
     *  a reference estimate for the residual variance.
     *
     * @param sigma_hat_sq Reference estimate for residual variance
     * @param cp_out Reference vector to store new result
     */
    void updateCp(double sigma_hat_sq, std::vector<double>& cp_out) const;

    // ============================================================================
    // Getters: Solution Path and Diagnostics
    // ============================================================================

    /**
     * @brief Get current coefficient vector (unscaled to original X, including dummy coefficients)
     *
     * @return Coefficient vector (p + num_dummies) × 1
     *
     * @note To extract only original predictor coefficients, use:
     *       beta.head(p_original_) or beta.subvec(0, p_original_-1)
     */
    virtual arma::vec getBeta() const;

    /**
     * @brief Get full solution path (unscaled to original X, including dummy coefficients)
     *
     * @return Coefficient matrix ((p + num_dummies) × num_steps)
     */
    virtual arma::mat getBetaPath() const;

    // ============================================================================
    // Model Selection Diagnostics
    // ============================================================================

    /**
     * @brief Get residual sum of squares for each step
     *
     * @return Vector of RSS values
     */
    virtual std::vector<double> getRSS() const;

    /**
     * @brief Get R-squared for each step
     *
     * @return Vector of R^2 values
     */
    virtual std::vector<double> getR2() const;

    /**
     * @brief Get Mallow's Cp for each step (lazy computation)
     *
     * @return Vector of Cp values
     */
    virtual std::vector<double> getCp() const;

    /**
     * @brief Get degrees of freedom (EXCLUDES dummies)
     *
     * @return DoF vector where DoF[k] = intercept + (active_predictors - active_dummies)
     */
    virtual std::vector<std::size_t> getDoF() const;

    /**
     * @brief Get current residual vector
     *
     * @return Vector of residuals
     */
    virtual arma::vec getResiduals() const;

    // ============================================================================
    // Non-overridable getters
    // ============================================================================

    /** @brief Get lambda sequence (max absolute correlation at each step) */
    inline const std::vector<double>& getLambda() const noexcept { return lambda_; }

    /** @brief Get action history (+j: add variable, -j: remove) */
    inline const std::vector<int>& getActions() const noexcept { return actions_; }

    /** @brief Get current active set indices */
    inline const std::vector<std::size_t>& getActives() const noexcept { return actives_; }

    /** @brief Get current inactive set indices */
    inline const std::vector<std::size_t>& getInactives() const noexcept { return inactives_; }

    /** @brief Get dropped (zero-variance) column indices */
    inline const std::vector<std::size_t>& getDroppedIndices() const noexcept {
        return dropped_indices_;
    }

    /** @brief Get algorithm type identifier */
    inline const std::string& getAlgorithmType() const noexcept { return algorithm_; }

    // ============================================================================
    // Summary statistics
    // ============================================================================

    /** @brief Get current active set size */
    inline std::size_t getNumActives() const noexcept { return actives_.size(); }

    /** @brief Get current inactive set size */
    inline std::size_t getNumInactives() const noexcept { return inactives_.size(); }

    /** @brief Get current step number */
    inline std::size_t getNumSteps() const noexcept { return currentStep_; }

    /** @brief Get maximum steps allowed */
    inline std::size_t getMaxSteps() const noexcept { return maxSteps_; }

    /** @brief Get total variable additions (≥ actives for LASSO) */
    inline std::size_t getNumAdditions() const noexcept { return num_additions_; }

    /** @brief Get total sum of squares of y */
    inline double getSSY() const noexcept { return ssy_; }

    /** @brief Get residual variance estimate */
    inline double getResVar() const noexcept { return resvar_; }

    /** @brief Check if verbosity logging is enabled */
    inline bool getVerbose() const noexcept { return verbose_; }


    // ============================================================================
    // Preprocessing Parameter (Optional if used)
    // ============================================================================

    /** @brief Check if normalization was applied */
    inline bool getNormalize() const noexcept { return normalize_; }

    /** @brief Check if intercept was fit */
    inline bool getIntercept() const noexcept { return intercept_; }

    /** @brief Get response mean */
    inline double getMuY() const noexcept { return mu_y_; }

    /**
     * @brief Get column means (if intercept=true)
     * @throws std::runtime_error if intercept=false
     */
    inline const arma::vec& getMeansx() const {
        if (!intercept_) {
            throw std::runtime_error(
                "TLARS_Solver::getMeansx: meansx_ is uninitialized (intercept=false)"
            );
        }
        return meansx_;
    }

    /**
     * @brief Get column L2 norms (if normalize=true)
     * @throws std::runtime_error if normalize=false
     */
    inline const arma::vec& getNormsx() const {
        if (!normalize_) {
            throw std::runtime_error(
                "TLARS_Solver::getNormsx: normsx_ is uninitialized (normalize=false)"
            );
        }
        return normsx_;
    }

    // ============================================================================
    // T-LARS Specific Getters
    // ============================================================================

    /**
     * @brief Number of original predictors (excludes dummies) in active set
     */
    inline std::size_t getNumActivePredictors() const noexcept {
        return actives_.size() - count_active_dummies_;
    }

    /**
     * @brief Number of dummies currently in active set
     */
    inline std::size_t getNumActiveDummies() const noexcept {
        return count_active_dummies_;
    }

    /**
     * @brief Total number of dummy variables in augmented X
     */
    inline std::size_t getNumDummies() const noexcept {
        return num_dummies_;
    }

    /**
     * @brief Original predictor count (before augmentation)
     */
    inline std::size_t getNumOriginalPredictors() const noexcept {
        return p_original_;
    }

    /**
     * @brief First dummy column index in X
     */
    inline std::size_t getDummyStartIndex() const noexcept {
        return dummy_start_idx_;
    }

    /**
     * @brief Check if variable index corresponds to a dummy
     */
    inline bool isDummy(std::size_t idx) const noexcept {
        return idx >= dummy_start_idx_;
    }

    /**
     * @brief Get indices of original predictors in active set (excludes dummies)
     */
    std::vector<std::size_t> getActivePredictorIndices() const;

    /**
     * @brief Get indices of dummies in active set
     */
    std::vector<std::size_t> getActiveDummyIndices() const;

    /**
     * @brief Get count of dummies in active set at each step
     */
    inline const std::vector<std::size_t>& getDummiesAtStep() const noexcept {
        return dummies_at_step_;
    }

    // ============================================================================
    // Serialization
    // ============================================================================

    /**
     * @brief Serialize solver state to archive (excludes X_ pointer)
     *
     * @details Saves all algorithm state except the design matrix pointer.
     * After deserialization, call reconnect(X) to restore pointer.
     *
     * @tparam Archive Cereal archive type
     * @param archive Cereal archive instance
     */
    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            // Step tracking
            CEREAL_NVP(currentStep_),
            CEREAL_NVP(maxSteps_),
            CEREAL_NVP(num_additions_),

            // Solution path
            CEREAL_NVP(beta_),
            CEREAL_NVP(betaPath_),
            CEREAL_NVP(lambda_),
            CEREAL_NVP(actions_),
            CEREAL_NVP(RSS_),
            CEREAL_NVP(R2_),
            CEREAL_NVP(Cp_),
            CEREAL_NVP(DoF_),

            // Active set
            CEREAL_NVP(actives_),
            CEREAL_NVP(inactives_),
            CEREAL_NVP(dropped_indices_),

            // Cholesky
            CEREAL_NVP(cholG_),
            CEREAL_NVP(A_A_),

            // Residuals and correlations
            CEREAL_NVP(y_),
            CEREAL_NVP(r_),
            CEREAL_NVP(correlations_),

            // Precomputed values
            CEREAL_NVP(ssy_),
            CEREAL_NVP(resvar_),
            CEREAL_NVP(effective_n_),
            CEREAL_NVP(max_actives_),

            // Preprocessing
            CEREAL_NVP(normalize_),
            CEREAL_NVP(intercept_),
            CEREAL_NVP(verbose_),
            CEREAL_NVP(meansx_),
            CEREAL_NVP(normsx_),
            CEREAL_NVP(mu_y_),

            // Algorithm type
            CEREAL_NVP(algorithm_),

            // T-LARS specific
            CEREAL_NVP(num_dummies_),
            CEREAL_NVP(count_active_dummies_),
            CEREAL_NVP(dummy_start_idx_),
            CEREAL_NVP(p_original_),
            CEREAL_NVP(dummies_at_step_)
        );
    }

    /**
     * @brief Save solver state to binary file
     *
     * @param filename Path to output file (e.g., "checkpoint.bin")
     *
     * @throws std::runtime_error if file cannot be opened
     */
    void save(const std::string& filename) const;

    /**
     * @brief Load solver state from binary file and reconnect to design matrix
     *
     * @param filename Path to checkpoint file
     * @param X Design matrix to reconnect (must be the original X)
     *
     * @return Loaded solver, ready for warm-start computation
     *
     * @throws std::runtime_error if:
     *  - File cannot be opened
     *  - Deserialization fails
     *  - X dimensions mismatch saved state
     *
     * @note Static method: use `auto lars = TLARS_Solver::load("file.bin", X);`
     */
    static TLARS_Solver load(const std::string& filename, arma::mat& X);

    /**
     * @brief Reconnect design matrix pointer after deserialization
     *
     * @param X Design matrix to connect
     *
     * @throws std::runtime_error if:
     *  - X.n_cols != beta_.n_elem
     *  - X.n_rows != r_.n_elem
     *  - X.n_rows inconsistent with effective_n
     *
     * @details Validates dimensions and sets X_ pointer + is_connected_ flag.
     */
    void reconnect(arma::mat& X);

    /**
     * @brief Check if solver is connected to design matrix
     *
     * @return true if X_ is valid and solver can execute steps
     */
    inline bool isConnected() const noexcept { return is_connected_; }

};

#endif /* TLARS_SOLVER_HPP */
