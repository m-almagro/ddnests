# ============================================================================
# validate_k1k2.jl — Cross-validation: K=1 and K=2 SSR against paper
# ============================================================================
#
# Paper's reference SSR values (from logs):
#   K=1: SSR = 520,235.72  (computed via FixedEffectModels IV regression)
#   K=2: SSR = 296,117.69  (computed via EM classification)
#
# This script runs the package for K=2 only (K=1 in the paper is not from EM)
# and also computes K=1 SSR from the EM framework for comparison.
#
# TO RUN:
#   cd ddnests
#   julia --project=. --threads=auto examples/validate_k1k2.jl
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ddnests
using DataFrames, CSV, Statistics, LinearAlgebra, Random, Printf, SparseArrays

# --------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------
const DATA_DIR = joinpath(homedir(), "Dropbox",
    "data_driven_nests_data", "Nielsen",
    "data_replication")

println("="^65)
println("VALIDATION: K=1 and K=2 SSR cross-check against paper")
println("="^65)

# --------------------------------------------------------------------------
# 1. Load data (identical to beer_market_example.jl)
# --------------------------------------------------------------------------
println("\n[1/3] Loading data...")

panel_path = joinpath(DATA_DIR, "panel_beer_structural_estimation.csv")
isfile(panel_path) || error("Data not found: $panel_path")

df_panel = DataFrame(CSV.File(panel_path))
sort!(df_panel, [:product, :market])

J = length(unique(df_panel.product))
M = length(unique(df_panel.market))
N = nrow(df_panel)

println("  J=$J, M=$M, N=$N")

product_ids = Vector{Int}(df_panel.product)
market_ids  = Vector{Int}(df_panel.market)
Y           = Vector{Float64}(df_panel.y)
X           = Matrix{Float64}(hcat(df_panel.log_unit_size, df_panel.log_upc_quantity))
prices      = Vector{Float64}(df_panel.log_price)
Z           = Matrix{Float64}(hcat(df_panel.gh_iv_upc_quantity, df_panel.gh_iv_unit_size))

shares = zeros(J, M)
for row in 1:N
    shares[product_ids[row], market_ids[row]] = df_panel.market_share[row]
end

data = DDNLData(Y, X, prices, shares,
    product_ids, market_ids, Z, J, M, size(X, 2))

# --------------------------------------------------------------------------
# 2. Reproduce K=1 SSR (paper uses IV via FixedEffectModels, EM uses OLS)
# --------------------------------------------------------------------------
println("\n[2/3] Computing K=1 SSR from EM framework...")

# For K=1, all products are in one group — trivial classification
gi_k1 = ones(Int, J, 1)

# Build X_hat exactly as classify() does: [MFE(M), exo, p_hat]
mfe = kron(ones(J), Matrix{Float64}(I, M, M))
X_fs = hcat(mfe, X, prices)
Z_fs = hcat(mfe, X, Z)
p_hat = ddnests.price_first_stage(X_fs, Z_fs, J, M)
X_hat = hcat(mfe, X, p_hat)

# Run group_estimates for K=1
par_k1, ssr_k1, _ = ddnests.group_estimates(gi_k1, Y, X_hat, J, M, 1)

println("  EM-OLS SSR (K=1)       : $(round(ssr_k1, digits=3))")
println("  Paper IV SSR (K=1)     : 520,235.721")
println("  Note: Paper K=1 uses IV regression, EM uses OLS → values will differ")

# --------------------------------------------------------------------------
# 3. Run K=2 classification and compare SSR
# --------------------------------------------------------------------------
println("\n[3/3] Running K=2 classification...")

opts = DDNLOptions(
    K_range=2:2,
    tol=1e-5,
    max_iter=1000,
    n_starts=2000,
    n_kmeans=0,
    k_selection=:in_sample_elbow,
    method=:kmeans,
    intercept_type=:market,
    group_dummies_utility=false,
    verbose=true
)

Random.seed!(1)
classifications = ddnests.classify_all(data, opts)
cr_k2 = classifications[1]

println("\n" * "="^65)
println("RESULTS")
println("="^65)
println()
@printf("  K=1 SSR (EM-OLS)   : %12.3f\n", ssr_k1)
@printf("  K=1 SSR (paper IV) : %12.3f\n", 520235.721)
println()
@printf("  K=2 SSR (package)  : %12.3f\n", cr_k2.ssr)
@printf("  K=2 SSR (paper)    : %12.3f\n", 296117.691)
println()

pct_diff = abs(cr_k2.ssr - 296117.691) / 296117.691 * 100
@printf("  K=2 %% difference   : %.2f%%\n", pct_diff)

# Group sizes
group_sizes = [sum(cr_k2.group_labels .== k) for k in 1:2]
println("\n  Group sizes: $group_sizes")
println("  Paper K=2 group sizes: [816, 1990]  (from log)")

println("\n" * "="^65)
if pct_diff < 1.0
    println("✓ K=2 SSR matches paper within 1% — PASS")
elseif pct_diff < 5.0
    println("~ K=2 SSR within 5% — close but check random seed / restarts")
else
    println("✗ K=2 SSR differs by >5% — investigate")
end
println("="^65)
