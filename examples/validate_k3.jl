# ============================================================================
# validate_k3.jl — Cross-validation: K=3 estimation against paper's Table A4
# ============================================================================
#
# Paper's Table A4 (K=3, IV specification):
#   log unit size    :  0.681 (0.040)
#   log upc quantity :  0.824 (0.042)
#   log price        : -1.038 (0.056)
#   σ1               :  0.968 (0.024)
#   σ2               :  0.649 (0.022)
#   σ3               :  0.391 (0.022)
#   Mean own-price   : -2.010 (0.002)
#   2806 products, 64497 observations
#
# TO RUN:
#   cd ddnests
#   julia --project=. --threads=auto examples/validate_k3.jl
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ddnests
using DataFrames, CSV, Statistics, LinearAlgebra, Random, Printf

# --------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------
const DATA_DIR = joinpath(homedir(), "Dropbox",
    "data_driven_nests_data", "Nielsen",
    "data_replication")

println("="^70)
println("VALIDATION: K=3 estimation vs paper Table A4")
println("="^70)

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
println("\n[1/4] Loading data...")

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
# 2. Classification for K=3
# --------------------------------------------------------------------------
println("\n[2/4] Running K=3 classification...")

opts = DDNLOptions(
    K_range=3:3,
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
cr = classifications[1]

println("\n  K=3 SSR: $(round(cr.ssr, digits=3))")
println("  Paper K=3 SSR: ~252,349")
group_sizes = [sum(cr.group_labels .== k) for k in 1:3]
println("  Group sizes: $group_sizes")

# --------------------------------------------------------------------------
# 3. Second-step estimation
# --------------------------------------------------------------------------
println("\n[3/4] Running second-step estimation...")

est = ddnests.second_step_estimate(data, cr.groups, 3, opts)

# --------------------------------------------------------------------------
# 4. Display results and compare
# --------------------------------------------------------------------------
println("\n" * "="^70)
println("RESULTS — Package vs Paper (Table A4)")
println("="^70)
println()

# Find coefficient indices
n_exo = size(X, 2)  # 2 exo chars
# With MFE: coefs are [MFE(48), exo(2), price(1), σ(3)]
# The exo chars start at position M+1 in the coefficient vector
mfe_cols = M  # 48 MFE columns

# Paper Table A4 values
paper = Dict(
    "log_unit_size"    => (0.681, 0.040),
    "log_upc_quantity" => (0.824, 0.042),
    "log_price"        => (-1.038, 0.056),
    "σ1"               => (0.968, 0.024),
    "σ2"               => (0.649, 0.022),
    "σ3"               => (0.391, 0.022),
)

# Extract package values
# Coefficient order: [MFE(48), log_unit_size, log_upc_quantity, price, σ1, σ2, σ3]
idx_lus = mfe_cols + 1
idx_luq = mfe_cols + 2
idx_price = mfe_cols + 3
idx_sigma = (mfe_cols + 4):(mfe_cols + 6)

println("  OLS specification:")
println("  ", "-"^65)
@printf("  %-22s %12s %12s\n", "", "Package", "Paper")
println("  ", "-"^65)
@printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "log unit size",
    est.beta_OLS[idx_lus], est.se_OLS[idx_lus], 0.059, 0.006)
@printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "log upc quantity",
    est.beta_OLS[idx_luq], est.se_OLS[idx_luq], 0.067, 0.004)
@printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "log price",
    est.beta_OLS[idx_price], est.se_OLS[idx_price], -0.067, 0.005)
for k in 1:3
    @printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "σ$k",
        est.beta_OLS[idx_sigma[k]], est.se_OLS[idx_sigma[k]],
        [0.272, 0.047, -0.204][k], [0.001, 0.001, 0.001][k])
end
println("  ", "-"^65)

println()
println("  IV (Full) specification:")
println("  ", "-"^65)
@printf("  %-22s %12s %12s\n", "", "Package", "Paper")
println("  ", "-"^65)
@printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "log unit size",
    est.beta_IV[idx_lus], est.se_IV[idx_lus], 0.681, 0.040)
@printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "log upc quantity",
    est.beta_IV[idx_luq], est.se_IV[idx_luq], 0.824, 0.042)
@printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "log price",
    est.beta_IV[idx_price], est.se_IV[idx_price], -1.038, 0.056)
for k in 1:3
    @printf("  %-22s %7.3f (%5.3f)   %7.3f (%5.3f)\n", "σ$k",
        est.sigma[k], est.sigma_se[k],
        [0.968, 0.649, 0.391][k], [0.024, 0.022, 0.022][k])
end
println("  ", "-"^65)

println()
println("  Group sizes:")
for k in 1:3
    println("    Group $k: $(group_sizes[k]) products")
end

# Check sigma validity
all_valid = all(0 .< est.sigma .< 1)
println("\n  σ all in (0,1): $all_valid")
price_neg = est.beta_IV[idx_price] < 0
println("  Price coefficient negative: $price_neg")

println("\n" * "="^70)
if all_valid && price_neg
    println("✓ Estimation produces valid results")
else
    println("✗ Issues with estimation — check above")
end
println("="^70)
