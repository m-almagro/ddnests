# ============================================================================
# beer_market_example.jl — DDNL applied to the paper's beer market data
# ============================================================================
#
# This example replicates the paper's empirical application using real
# Nielsen scanner data from the US beer market (2022).
#
# Pipeline:
#   1. Load pre-processed panel and product characteristics CSVs
#   2. Build DDNLData using pre-computed GH / Hausman instruments
#   3. Run DDNL classification + elbow selection + estimation
#   4. Display and save results (tables, elbow plots, elasticity heatmap)
#
# DATA LOCATION:
#   The data files live outside the repo in Dropbox. Set DATA_DIR below.
#   Two CSV files are required (produced by the Stata scripts 1a / 1b):
#     - panel_beer_structural_estimation.csv   (~134 K rows)
#     - product_characteristics_estimation.csv (~2 800 rows)
#
# TO RUN:
#   julia --threads=auto beer_market_example.jl
#   julia --threads=auto beer_market_example.jl hausman    # Hausman instruments
# ============================================================================

using ddnests
using DataFrames, CSV, Statistics, LinearAlgebra, Random, Printf

# --------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------
const DATA_DIR = joinpath(homedir(), "Library", "CloudStorage", "Dropbox",
    "DataDrivenNests", "data_driven_nests_data", "Nielsen",
    "data_replication")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "..", "output")

# IV type: pass "hausman" as CLI arg, default is GH
const IV_TYPE = length(ARGS) > 0 && ARGS[1] == "hausman" ? :hausman : :GH

# K range to evaluate (paper uses 2:10 — wider range gives better elbow context)
const K_RANGE = 2:5

# EM restarts — paper uses 2000 (fewer restarts risk suboptimal SSR)
const N_STARTS = 2000

mkpath(joinpath(OUTPUT_DIR, "figures"))
mkpath(joinpath(OUTPUT_DIR, "tables"))

println("="^65)
println("ddnests.jl — Beer Market Empirical Application")
println("="^65)
println("Data dir : $DATA_DIR")
println("IV type  : $IV_TYPE")
println("K range  : $K_RANGE")
println("Restarts : $N_STARTS")
println("-"^65)

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
println("\n[1/5] Loading data...")

panel_path = joinpath(DATA_DIR, "panel_beer_structural_estimation.csv")
chars_path = joinpath(DATA_DIR, "product_characteristics_estimation.csv")

if !isfile(panel_path)
    error("""
    Data file not found: $panel_path
    Make sure the Stata scripts 1a and 1b have been run and DATA_DIR is correct.
    """)
end

df_panel = DataFrame(CSV.File(panel_path))
df_chars = DataFrame(CSV.File(chars_path))

println("  Panel observations : $(nrow(df_panel))")
println("  Products           : $(length(unique(df_panel.product)))")
println("  Markets (states)   : $(length(unique(df_panel.market)))")
println("  Columns            : $(names(df_panel))")

# --------------------------------------------------------------------------
# 2. Prepare the data array
# --------------------------------------------------------------------------
println("\n[2/5] Preparing estimation data...")

# Sort by product, then market (required for product-major row ordering)
sort!(df_panel, [:product, :market])

J = length(unique(df_panel.product))
M = length(unique(df_panel.market))
N = nrow(df_panel)

println("  J (products) = $J,  M (markets) = $M,  N = $N")

# Integer product and market indices (already 1..J and 1..M in the CSV)
product_ids = Vector{Int}(df_panel.product)
market_ids = Vector{Int}(df_panel.market)

# Dependent variable: log(s_j / s_0),  zero = missing in balanced panel
Y = Vector{Float64}(df_panel.y)

# Exogenous product characteristics (without market FEs; those are added
# inside second_step_estimate via intercept_type = :market)
X = Matrix{Float64}(hcat(df_panel.log_unit_size, df_panel.log_upc_quantity))

# Endogenous price
prices = Vector{Float64}(df_panel.log_price)

# Market shares: J × M matrix
shares = zeros(J, M)
for row in 1:N
    j = product_ids[row]
    m = market_ids[row]
    shares[j, m] = df_panel.market_share[row]
end

# --------------------------------------------------------------------------
# 3. Select and build instrument matrix
# --------------------------------------------------------------------------
println("\n[3/5] Building instrument matrix (iv_type = $IV_TYPE)...")

# The instruments are pre-computed by Stata (1b_prep_product_characteristics.do)
# and stored directly in the panel CSV — we just extract the relevant columns.
#
#   GH instruments   : gh_iv_unit_size, gh_iv_upc_quantity
#   Hausman instrument: hausman_iv
#
# Using pre-computed instruments guarantees consistency with the paper's
# results. If you want to rebuild from scratch, call
#   ddnests.gandhi_houde_iv(df_panel, [:log_unit_size, :log_upc_quantity],
#                                   :market, :product)

if IV_TYPE == :GH
    Z = Matrix{Float64}(hcat(df_panel.gh_iv_upc_quantity,
        df_panel.gh_iv_unit_size))
    println("  Using GH instruments: gh_iv_upc_quantity, gh_iv_unit_size")
else
    Z = reshape(Vector{Float64}(df_panel.hausman_iv), :, 1)
    println("  Using Hausman instrument: hausman_iv")
end

# --------------------------------------------------------------------------
# 4. Construct DDNLData and run pipeline
# --------------------------------------------------------------------------
println("\n[4/5] Running DDNL pipeline...")

data = DDNLData(
    Y, X, prices, shares,
    product_ids, market_ids,
    Z,
    J, M, size(X, 2)
)

opts = DDNLOptions(
    K_range=K_RANGE,
    tol=1e-5,
    max_iter=1000,
    n_starts=N_STARTS,
    n_kmeans=0,
    k_selection=:in_sample_elbow,
    cv_folds=10,
    method=:kmeans,
    intercept_type=:market,   # market fixed effects in second step
    group_dummies_utility=false,
    verbose=true
)

result = ddnl(data, opts)

# --------------------------------------------------------------------------
# 5. Display results
# --------------------------------------------------------------------------
println("\n" * "="^65)
println("RESULTS SUMMARY")
println("="^65)

K_star = result.optimal_k
println("\nOptimal number of groups : $K_star")

# --- Nesting parameters ---
println("\nNesting parameters (σ_k) by group:")
for k in 1:K_star
    σ = round(result.estimation.sigma[k], digits=4)
    se = round(result.estimation.sigma_se[k], digits=4)
    println("  Group $k : σ = $σ  (s.e. = $se)")
end

# --- Price coefficient ---
price_idx = findfirst(==("price"), result.estimation.coef_names)
β_p_iv = round(result.estimation.beta_IV[price_idx], digits=4)
β_p_ols = round(result.estimation.beta_OLS[price_idx], digits=4)
println("\nPrice coefficient  — OLS : $β_p_ols  |  IV ($(result.estimation.iv_spec)) : $β_p_iv")

# --- Elasticity table ---
println("\nElasticity statistics by group:")
println(result.elasticity_stats)

agg_w = vec(mean(shares, dims=2))
agg = aggregate_elasticities(result.elasticity_matrix,
    result.classifications[findfirst(c -> c.k == K_star,
        result.classifications)].groups,
    agg_w)
@printf("\nAggregate own-price  elasticity : %.4f  (s.d. %.4f)\n",
    agg.own_mean, agg.own_sd)
@printf("Aggregate cross-price elasticity : %.6f  (s.d. %.6f)\n",
    agg.cross_mean, agg.cross_sd)

# --- Elbow summary ---
println("\nElbow selection:")
println("  Chord method       : K = $(result.elbow.k_chord)")
println("  2nd-derivative     : K = $(result.elbow.k_secondderiv)")
println("  Selected K*        : $(result.elbow.k_star)")
println("\nSSR by K:")
for (k, ssr) in zip(result.elbow.k_values, result.elbow.ssr_by_k)
    selected = k == result.optimal_k ? " ← K*" : ""
    @printf("  K=%2d : SSR = %.4f%s\n", k, ssr, selected)
end

# --- Group composition ---
println("\nGroup composition (products per group):")
for k in 1:K_star
    n_k = sum(result.group_labels .== k)
    println("  Group $k : $n_k products")
end

# Optionally join group labels back to product characteristics for inspection
df_groups = select(df_chars, :product, :brand, :beer_type,
    :log_unit_size, :log_upc_quantity, :log_price)
df_groups[!, :group] = result.group_labels[df_groups.product]

println("\nSample products per group (first 3 per group):")
for k in 1:K_star
    subset = filter(r -> r.group == k, df_groups)
    println("  --- Group $k ($(nrow(subset)) products) ---")
    for row in eachrow(first(subset, 3))
        @printf("    %-45s  type=%-18s  log_size=%.2f  log_qty=%.2f\n",
            row.brand, row.beer_type, row.log_unit_size, row.log_upc_quantity)
    end
end

# --------------------------------------------------------------------------
# 6. Plots
# --------------------------------------------------------------------------
println("\n[5/5] Generating plots...")

suffix = IV_TYPE == :GH ? "" : "_hausman"
cr_star = result.classifications[findfirst(c -> c.k == K_star, result.classifications)]

# Elbow plots — all four styles
for style in [:combined, :chord, :simple, :derivative]
    fname = joinpath(OUTPUT_DIR, "figures", "elbow_$(style)$(suffix).pdf")
    plot_elbow(result.elbow; style=style, save_path=fname)
    println("  Saved: $fname")
end

# Elasticity heatmap
fname_heat = joinpath(OUTPUT_DIR, "figures",
    "elasticity_heatmap_k$(K_star)$(suffix).png")
plot_elasticity_heatmap(result.elasticity_matrix, cr_star.groups;
    title_str="Elasticity Matrix — K=$K_star ($IV_TYPE instruments)",
    save_path=fname_heat)
println("  Saved: $fname_heat")

# Group elasticity bar chart
fname_bars = joinpath(OUTPUT_DIR, "figures",
    "group_elasticities_k$(K_star)$(suffix).pdf")
plot_group_elasticities(result.elasticity_stats; save_path=fname_bars)
println("  Saved: $fname_bars")

# --------------------------------------------------------------------------
# 7. Save outputs to CSV
# --------------------------------------------------------------------------
fname_elas_mat = joinpath(OUTPUT_DIR, "tables",
    "mat_elasticities_k$(K_star)$(suffix).csv")
fname_elas_stats = joinpath(OUTPUT_DIR, "tables",
    "elasticities_stats_k$(K_star)$(suffix).csv")
fname_groups = joinpath(OUTPUT_DIR, "tables",
    "product_groups_k$(K_star)$(suffix).csv")

CSV.write(fname_elas_mat, elasticity_dataframe(result.elasticity_matrix))
CSV.write(fname_elas_stats, result.elasticity_stats)
CSV.write(fname_groups, df_groups)

println("  Saved: $fname_elas_mat")
println("  Saved: $fname_elas_stats")
println("  Saved: $fname_groups")

println("\n" * "="^65)
println("Beer market DDNL estimation complete.")
println("="^65)
