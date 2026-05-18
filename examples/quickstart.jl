# ============================================================================
# quickstart.jl — Minimal working example of the ddnests.jl package
# ============================================================================
#
# This example generates synthetic market data and runs the full DDNL pipeline:
#   1. Constructs Gandhi-Houde instruments
#   2. Classifies products into nests via EM/k-means
#   3. Selects optimal K via distance-elbow method
#   4. Estimates nested logit parameters (OLS + IV)
#   5. Computes and plots elasticities
#
# To run:
#   julia --threads=4 quickstart.jl
# ============================================================================

using ddnests
using DataFrames, Random, Statistics

# --------------------------------------------------------------------------
# Step 1: Generate synthetic panel data
# --------------------------------------------------------------------------
Random.seed!(2024)

J = 20    # products
M = 30    # markets
K_true = 3  # true number of groups

# True group assignments
true_groups = vcat(fill(1, 7), fill(2, 7), fill(3, 6))

# True parameters
sigma_true = [0.4, 0.6, 0.8]
beta_x1 = 1.5
beta_x2 = 0.8
beta_p = -3.0

println("Generating synthetic data: $J products, $M markets, $K_true true groups")

rows = []
for j in 1:J
    g = true_groups[j]
    for m in 1:M
        # Product characteristics (differ systematically by group)
        x1 = g * 0.5 + 0.5 * randn()
        x2 = (4 - g) * 0.3 + 0.4 * randn()
        z_cost = randn()  # cost shifter

        # Price: endogenous (correlated with demand shock)
        xi = 0.5 * randn()  # demand shock
        price = 2.0 + 0.3 * x1 + 0.2 * x2 + 0.5 * z_cost + 0.3 * xi + 0.1 * randn()
        log_price = log(max(price, 0.1))

        # Market share (simplified nested logit)
        delta = beta_x1 * x1 + beta_x2 * x2 + beta_p * log_price + xi
        share = exp(delta) / (1.0 + J * exp(delta / J))
        share = clamp(share, 1e-6, 0.3)

        # Dependent variable: log(s_j / s_0)
        s0 = 1.0 - J * share  # approximate outside share
        s0 = max(s0, 0.01)
        y = log(share / s0)

        push!(rows, (
            product_id = j,
            market_id = m,
            log_price = log_price,
            market_share = share,
            log_size = x1,
            log_quantity = x2,
            y = y
        ))
    end
end

df = DataFrame(rows)
println("Data generated: $(nrow(df)) observations\n")

# --------------------------------------------------------------------------
# Step 2: Run the DDNL pipeline
# --------------------------------------------------------------------------
println("Running DDNL pipeline...")
println("=" ^ 60)

# Build instruments first (GH instruments from characteristics)
Z_mat = gandhi_houde_iv(df, [:log_size, :log_quantity], :market_id, :product_id)
df.gh_iv1 = Z_mat[:, 1]
df.gh_iv2 = Z_mat[:, 2]

result = ddnl_from_df(df;
    y_col = :y,
    price_col = :log_price,
    product_col = :product_id,
    market_col = :market_id,
    share_col = :market_share,
    x_cols = [:log_size, :log_quantity],
    Z_cols = [:gh_iv1, :gh_iv2],
    K_range = 2:6,
    n_starts = 100,
    max_iter = 500,
    verbose = true
)

# --------------------------------------------------------------------------
# Step 3: Inspect results
# --------------------------------------------------------------------------
println("\n" * "=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)

println("\nOptimal number of groups: $(result.optimal_k)")
println("Group assignments: $(result.group_labels)")
println("\nNesting parameters (σ):")
for (k, s) in enumerate(result.estimation.sigma)
    println("  Group $k: σ = $(round(s, digits=4)) ± $(round(result.estimation.sigma_se[k], digits=4))")
end

println("\nPrice coefficient (IV): $(round(result.estimation.beta_IV[findfirst(==("price"), result.estimation.coef_names)], digits=4))")
println("IV specification: $(result.estimation.iv_spec)")

println("\nElasticity statistics by group:")
println(result.elasticity_stats)

agg = aggregate_elasticities(result.elasticity_matrix,
    result.classifications[findfirst(c -> c.k == result.optimal_k, result.classifications)].groups,
    vec(mean(result.data.shares, dims=2)))
println("\nAggregate elasticities:")
println("  Own-price (mean): $(round(agg.own_mean, digits=4))")
println("  Own-price (sd):   $(round(agg.own_sd, digits=4))")
println("  Cross-price (mean): $(round(agg.cross_mean, digits=6))")

# --------------------------------------------------------------------------
# Step 4: Generate plots
# --------------------------------------------------------------------------
println("\nGenerating plots...")

# Elbow plot
p_elbow = plot_elbow(result.elbow; style=:combined)
display(p_elbow)

# Elasticity heatmap
cr_star = result.classifications[findfirst(c -> c.k == result.optimal_k, result.classifications)]
p_heatmap = plot_elasticity_heatmap(result.elasticity_matrix, cr_star.groups;
    title_str="Elasticity Matrix (K=$(result.optimal_k))")
display(p_heatmap)

# Group elasticity bar chart
p_bars = plot_group_elasticities(result.elasticity_stats)
display(p_bars)

println("\nDone! All plots displayed.")
