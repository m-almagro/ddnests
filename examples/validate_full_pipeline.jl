# ============================================================================
# validate_full_pipeline.jl — Full pipeline using paper's saved classifications
# ============================================================================
#
# This script validates the complete DDNL pipeline (classification → elbow →
# estimation → elasticities) by loading the paper's saved classification
# results and SSR values, then running elbow selection and estimation.
#
# This avoids re-running the expensive classification step (hours) while
# validating that the package's estimation, elbow selection, and elasticity
# code produces the paper's K=4 results:
#   σ = [0.523, 0.807, 0.336, 0.042]
#   price = -0.416
#   group sizes: 659/242/1102/803
#
# TO RUN:
#   cd ddnests
#   julia --project=. examples/validate_full_pipeline.jl
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ddnests
using DataFrames, CSV, Statistics, LinearAlgebra, Random, Printf

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
println("="^70)
println("DDNL Full Pipeline Validation (using paper's saved classifications)")
println("="^70)

const DATA_DIR = joinpath(homedir(), "Dropbox",
    "data_driven_nests_data", "Nielsen", "data_replication")

println("\n[1/6] Loading data...")
df_panel = DataFrame(CSV.File(joinpath(DATA_DIR, "panel_beer_structural_estimation.csv")))
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

data = DDNLData(Y, X, prices, shares, product_ids, market_ids,
                Z, J, M, size(X, 2))

opts = DDNLOptions(
    K_range = 2:10,
    tol = 1e-5,
    max_iter = 1000,
    n_starts = 2000,
    k_selection = :in_sample_elbow,
    cv_folds = 10,
    method = :kmeans,
    intercept_type = :market,
    group_dummies_utility = false,
    verbose = true
)

# --------------------------------------------------------------------------
# 2. Load paper's saved classifications and SSR values
# --------------------------------------------------------------------------
println("\n[2/6] Loading paper's saved classification results...")

class_file = joinpath(DATA_DIR, "classification_results.csv")
ssr_file   = joinpath(DATA_DIR, "max_iter_done.csv")

isfile(class_file) || error("Cannot find: $class_file")
isfile(ssr_file)   || error("Cannot find: $ssr_file")

class_df = DataFrame(CSV.File(class_file))
sort!(class_df, :product)
ssr_df = DataFrame(CSV.File(ssr_file))

# Build ClassificationResult for each K that has a group_kN column
classifications = ClassificationResult[]

for K in opts.K_range
    col_name = Symbol("group_k$K")
    if !hasproperty(class_df, col_name)
        println("  Skipping K=$K: no column $col_name in classification CSV")
        continue
    end

    group_labels = Vector{Int}(class_df[!, col_name])
    gi = zeros(Int, J, K)
    for j in 1:J
        gi[j, group_labels[j]] = 1
    end

    # SSR from max_iter_done.csv (rows correspond to K=2,3,4,...)
    ssr_row = K - 1  # K=2 is row 1, K=3 is row 2, etc.
    ssr_val = ssr_df.SSR[ssr_row]

    group_sizes = [sum(group_labels .== k) for k in 1:K]
    println("  K=$K: SSR=$(round(ssr_val, digits=2)), groups=$group_sizes")

    push!(classifications, ClassificationResult(
        K, gi, group_labels, Float64[], ssr_val, true, 0
    ))
end

# --------------------------------------------------------------------------
# 3. Elbow selection (matching paper's procedure)
# --------------------------------------------------------------------------
println("\n[3/6] Running elbow selection...")

elbow = select_k(classifications, data, opts)
println("  Chord method:    K = $(elbow.k_chord)")
println("  2nd derivative:  K = $(elbow.k_secondderiv)")
println("  Selected K*:     $(elbow.k_star)")

# --------------------------------------------------------------------------
# 4. Second-step estimation at K*
# --------------------------------------------------------------------------
K_star = elbow.k_star
println("\n[4/6] Running second-step estimation (K=$K_star)...")

cr_star = classifications[findfirst(c -> c.k == K_star, classifications)]
estimation = second_step_estimate(data, cr_star.groups, K_star, opts)

# --------------------------------------------------------------------------
# 5. Elasticities
# --------------------------------------------------------------------------
println("\n[5/6] Computing elasticities...")

price_idx = findfirst(==("price"), estimation.coef_names)
beta_price = estimation.beta_IV[price_idx]

elas_mat = compute_elasticities(data.shares, cr_star.groups, beta_price,
                                estimation.sigma, J, M)

weights = vec(mean(shares, dims=2))
elas_stats_df = elasticity_stats(elas_mat, cr_star.groups, weights)
agg = aggregate_elasticities(elas_mat, cr_star.groups, weights)

# --------------------------------------------------------------------------
# 6. Display results and compare with paper
# --------------------------------------------------------------------------
println("\n[6/6] Results")
println("="^70)

# Coefficient indices
n_exo = M + size(X, 2)  # MFE(48) + 2 exo chars = 50
idx_lus   = n_exo - 1   # log_unit_size is second-to-last exo
idx_luq   = n_exo       # log_upc_quantity is last exo
idx_price = n_exo + 1   # price comes after X_exo

println("\n  ESTIMATION — Package vs Paper (K=$K_star)")
println("  ", "-"^60)
@printf("  %-22s %14s %14s\n", "", "Package", "Paper")
println("  ", "-"^60)

# Paper's K=4 target values
paper_iv = Dict(
    "log_unit_size"    => (0.339, 0.025),
    "log_upc_quantity" => (0.441, 0.026),
    "log_price"        => (-0.416, 0.033),
    "sigma_1"          => (0.523, 0.014),
    "sigma_2"          => (0.807, 0.016),
    "sigma_3"          => (0.336, 0.013),
    "sigma_4"          => (0.042, 0.015),
)

@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log unit size",
    estimation.beta_IV[idx_lus], estimation.se_IV[idx_lus], 0.339, 0.025)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log upc quantity",
    estimation.beta_IV[idx_luq], estimation.se_IV[idx_luq], 0.441, 0.026)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log price",
    estimation.beta_IV[idx_price], estimation.se_IV[idx_price], -0.416, 0.033)
for k in 1:K_star
    paper_σ = [0.523, 0.807, 0.336, 0.042][k]
    paper_se = [0.014, 0.016, 0.013, 0.015][k]
    @printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "σ$k",
        estimation.sigma[k], estimation.sigma_se[k], paper_σ, paper_se)
end
println("  ", "-"^60)

println("\n  Group sizes:")
for k in 1:K_star
    n_k = sum(cr_star.group_labels .== k)
    paper_sizes = [659, 242, 1102, 803]
    paper_n = K_star == 4 ? paper_sizes[k] : "?"
    println("    Group $k: $n_k products (paper: $paper_n)")
end

println("\n  Elasticity statistics:")
@printf("    Mean own-price:   %.4f (paper: -3.561)\n", agg.own_mean)
@printf("    Mean cross-price: %.6f (paper: 0.011)\n", agg.cross_mean)

println("\n  Validation checks:")
println("    All σ ∈ (0,1): ", all(0 .< estimation.sigma .< 1))
println("    Price < 0:     ", estimation.beta_IV[idx_price] < 0)
println("    K* = 4:        ", K_star == 4)
println("="^70)
