# ============================================================================
# types.jl — Core data structures for ddnests.jl
# ============================================================================

"""
    DDNLData

User-facing input container for the DDNL estimation pipeline.

# Fields
- `Y::Vector{Float64}`: Dependent variable — log share ratio `log(s_j/s_0)`.
- `X::Matrix{Float64}`: Exogenous product characteristics (J*M × P).
- `prices::Vector{Float64}`: Observed prices (J*M × 1).
- `shares::Matrix{Float64}`: Market shares matrix (J × M).
- `product_id::Vector{Int}`: Product identifier for each observation.
- `market_id::Vector{Int}`: Market identifier for each observation.
- `Z::Matrix{Float64}`: Instrument matrix for price endogeneity (user-supplied; e.g., GH, Hausman, BLP-style).
- `J::Int`: Number of products.
- `M::Int`: Number of markets.
- `P::Int`: Number of exogenous characteristics (columns in X).
"""
mutable struct DDNLData
    Y::Vector{Float64}
    X::Matrix{Float64}
    prices::Vector{Float64}
    shares::Matrix{Float64}
    product_id::Vector{Int}
    market_id::Vector{Int}
    Z::Matrix{Float64}
    J::Int
    M::Int
    P::Int
end

"""
    DDNLOptions

Configuration for the DDNL estimation pipeline.

# Fields
- `K_range::UnitRange{Int}`: Range of group counts to evaluate (default `2:10`).
- `tol::Float64`: EM convergence tolerance (default `1e-5`).
- `max_iter::Int`: Maximum EM iterations per run (default `1000`).
- `n_starts::Int`: Number of random restarts for classification (default `2000`).
- `n_kmeans::Int`: Number of k-means warm-starts (default `0`).
- `k_selection::Symbol`: K selection method — `:in_sample_elbow` or `:out_of_sample_elbow` (default `:in_sample_elbow`).
- `cv_folds::Int`: Number of cross-validation folds for out-of-sample K selection (default `10`).
- `method::Symbol`: Classification method — `:kmeans` or `:em`.
- `intercept_type::Symbol`: Intercept specification — `:market`, `:baseline`, or `:none`.
- `group_dummies_utility::Bool`: Include K-1 group dummies in utility (default `false`).
- `verbose::Bool`: Print progress information (default `true`).
"""
mutable struct DDNLOptions
    K_range::UnitRange{Int}
    tol::Float64
    max_iter::Int
    n_starts::Int
    n_kmeans::Int
    k_selection::Symbol
    cv_folds::Int
    method::Symbol
    intercept_type::Symbol
    group_dummies_utility::Bool
    verbose::Bool
end

function DDNLOptions(;
    K_range = 2:10,
    tol = 1e-5,
    max_iter = 1000,
    n_starts = 2000,
    n_kmeans = 0,
    k_selection = :in_sample_elbow,
    cv_folds = 10,
    method = :kmeans,
    intercept_type = :market,
    group_dummies_utility = false,
    verbose = true
)
    @assert k_selection in (:in_sample_elbow, :out_of_sample_elbow) "k_selection must be :in_sample_elbow or :out_of_sample_elbow"
    @assert method in (:kmeans, :em) "method must be :kmeans or :em"
    @assert intercept_type in (:market, :baseline, :none) "intercept_type must be :market, :baseline, or :none"
    DDNLOptions(K_range, tol, max_iter, n_starts, n_kmeans, k_selection, cv_folds,
                method, intercept_type, group_dummies_utility, verbose)
end

"""
    ClassificationResult

Output from classification for a single value of K.

# Fields
- `k::Int`: Number of groups.
- `groups::Matrix{Int}`: Group indicator matrix (J × K), one-hot encoding.
- `group_labels::Vector{Int}`: Group assignment vector (J × 1), integer labels 1…K.
- `parameters::Vector{Float64}`: Estimated EM parameters at convergence.
- `ssr::Float64`: Sum of squared residuals at convergence.
- `converged::Bool`: Whether EM converged within tolerance.
- `n_iterations::Int`: Number of EM iterations used.
"""
mutable struct ClassificationResult
    k::Int
    groups::Matrix{Int}
    group_labels::Vector{Int}
    parameters::Vector{Float64}
    ssr::Float64
    converged::Bool
    n_iterations::Int
end

"""
    ElbowResult

Output from the distance-elbow k-selection procedure.

# Fields
- `k_chord::Int`: Optimal k via maximum perpendicular distance to chord.
- `k_secondderiv::Int`: Optimal k via maximum second-derivative (deceleration).
- `k_star::Int`: Recommended k (validated against IV sigma constraints).
- `ssr_by_k::Vector{Float64}`: SSR values indexed by k.
- `k_values::Vector{Int}`: Corresponding k values.
- `distances::Vector{Float64}`: Perpendicular distances for chord method.
"""
mutable struct ElbowResult
    k_chord::Int
    k_secondderiv::Int
    k_star::Int
    ssr_by_k::Vector{Float64}
    k_values::Vector{Int}
    distances::Vector{Float64}
end

"""
    EstimationResult

Output from second-step nested logit estimation.

# Fields
- `k::Int`: Number of groups used.
- `beta_OLS::Vector{Float64}`: OLS coefficient estimates.
- `se_OLS::Vector{Float64}`: OLS heteroskedasticity-robust standard errors.
- `beta_IV::Vector{Float64}`: IV (2SLS) coefficient estimates.
- `se_IV::Vector{Float64}`: IV heteroskedasticity-robust standard errors.
- `sigma::Vector{Float64}`: Nesting parameters (one per group), transformed as `1 - coef`.
- `sigma_se::Vector{Float64}`: Standard errors of nesting parameters.
- `coef_names::Vector{String}`: Names of coefficients.
- `iv_spec::Symbol`: IV specification used (`:price`, `:group`, `:full`).
"""
mutable struct EstimationResult
    k::Int
    beta_OLS::Vector{Float64}
    se_OLS::Vector{Float64}
    beta_IV::Vector{Float64}
    se_IV::Vector{Float64}
    sigma::Vector{Float64}
    sigma_se::Vector{Float64}
    coef_names::Vector{String}
    iv_spec::Symbol
end

"""
    DDNLResult

Complete output from the `ddnl()` pipeline.

# Fields
- `data::DDNLData`: Input data (with instruments attached).
- `options::DDNLOptions`: Configuration used.
- `classifications::Vector{ClassificationResult}`: Classification results for each k in K_range.
- `elbow::ElbowResult`: K-selection results.
- `estimation::EstimationResult`: Second-step estimation at optimal k*.
- `elasticity_matrix::Matrix{Float64}`: J×J mean elasticity matrix.
- `elasticity_stats::DataFrames.DataFrame`: Per-group elasticity statistics.
- `optimal_k::Int`: Selected number of groups.
- `group_labels::Vector{Int}`: Final product-to-group assignments (J × 1).
"""
mutable struct DDNLResult
    data::DDNLData
    options::DDNLOptions
    classifications::Vector{ClassificationResult}
    elbow::ElbowResult
    estimation::EstimationResult
    elasticity_matrix::Matrix{Float64}
    elasticity_stats::DataFrames.DataFrame
    optimal_k::Int
    group_labels::Vector{Int}
end
