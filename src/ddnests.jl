# ============================================================================
# ddnests.jl — Data-Driven Nested Logit estimation package
#
# Implements the DDNL methodology: jointly estimates the nesting structure
# and parameters of nested logit demand models from market-level data.
# ============================================================================

module ddnests

using LinearAlgebra, SparseArrays, Statistics, Random, DataFrames, CSV
using Distributions, StatsBase, Plots, Printf

# --------------------------------------------------------------------------
# Source files
# --------------------------------------------------------------------------
include("types.jl")
include("utils.jl")
include("instruments.jl")
include("classification.jl")
include("elbow.jl")
include("estimation.jl")
include("elasticities.jl")
include("plotting.jl")

# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------

# Types
export DDNLData, DDNLOptions, ClassificationResult, ElbowResult, EstimationResult, DDNLResult

# Main pipeline
export ddnl, ddnl_from_df

# Instruments (utility helpers — user calls these to build Z before passing to DDNLData)
export gandhi_houde_iv, hausman_iv

# Classification
export classify, classify_all

# Elbow
export find_elbow_chord, find_elbow_secondderiv, select_k, select_k_cv

# Estimation
export second_step_estimate, ols_robust, iv_2sls

# Elasticities
export compute_elasticities, elasticity_stats, aggregate_elasticities, elasticity_dataframe

# Plotting
export plot_elbow, plot_elasticity_heatmap, plot_group_elasticities

# Utilities
export wmean, wstd, wmedian, make_dummies, make_market_fe
export labels_to_indicator, indicator_to_labels, conditional_probs

# Balance check
export balance_check

# --------------------------------------------------------------------------
# Balance Check
# --------------------------------------------------------------------------

"""
    balance_check(data::DDNLData, opts::DDNLOptions)

Check panel balance and report missing data statistics.
Prints diagnostics about product coverage across markets and missing observations.
"""
function balance_check(data::DDNLData, opts::DDNLOptions)
    J, M = data.J, data.M
    Y = data.Y

    # Count how many markets each product appears in (non-zero Y)
    markets_per_product = zeros(Int, J)
    for j in 1:J
        rows = ((j-1)*M+1):(j*M)
        markets_per_product[j] = sum(Y[rows] .!= 0)
    end

    min_mkts = minimum(markets_per_product)
    mean_mkts = mean(markets_per_product)
    max_mkts = maximum(markets_per_product)

    if opts.verbose
        println("  Panel balance: min=$(min_mkts), mean=$(round(mean_mkts, digits=1)), max=$(max_mkts) markets per product")
    end

    if min_mkts < 2
        n_problematic = sum(markets_per_product .< 2)
        println("  WARNING: $(n_problematic) product(s) appear in fewer than 2 markets. Classification may be unreliable for these products.")
    end

    # Check for missing/zero values in Y
    n_missing = sum(Y .== 0)
    if n_missing > 0
        if opts.verbose
            println("  ⚠ Warning: Unbalanced panel detected ($(n_missing) missing observations). Missing values are treated as absent from those markets for ALL variables.")
        end
    end

    return (min_markets=min_mkts, mean_markets=mean_mkts, max_markets=max_mkts, n_missing=n_missing)
end

# --------------------------------------------------------------------------
# High-level API
# --------------------------------------------------------------------------

"""
    ddnl(data::DDNLData; kwargs...)

Run the full DDNL pipeline: classification → elbow selection → estimation → elasticities.

# Arguments
- `data::DDNLData`: Prepared input data.
- `kwargs...`: Passed to `DDNLOptions` constructor (e.g., `K_range=2:8`, `n_starts=2000`).

# Returns
- `DDNLResult`: Complete results including optimal K, group assignments, estimation
  results, elasticity matrix, and elasticity statistics.

# Example
```julia
result = ddnl(data; K_range=2:10, n_starts=2000)
println("Optimal K: ", result.optimal_k)
println("Nesting parameters: ", result.estimation.sigma)
plot_elbow(result.elbow)
plot_elasticity_heatmap(result.elasticity_matrix, result.classifications[end].groups)
```
"""
function ddnl(data::DDNLData; kwargs...)
    opts = DDNLOptions(; kwargs...)
    return _ddnl_run(data, opts)
end

"""
    ddnl(data::DDNLData, opts::DDNLOptions)

Run the full DDNL pipeline with explicit options.
"""
function ddnl(data::DDNLData, opts::DDNLOptions)
    return _ddnl_run(data, opts)
end

function _ddnl_run(data::DDNLData, opts::DDNLOptions)
    opts.verbose && println("=" ^ 60)
    opts.verbose && println("ddnests.jl — DDNL Estimation Pipeline")
    opts.verbose && println("=" ^ 60)
    opts.verbose && println("Products: $(data.J), Markets: $(data.M), Characteristics: $(data.P)")
    opts.verbose && println("K range: $(opts.K_range), K selection: $(opts.k_selection)")
    opts.verbose && println("Method: $(opts.method), Restarts: $(opts.n_starts)")
    opts.verbose && println("-" ^ 60)

    # Balance check (before classification)
    opts.verbose && println("\n[Pre-check] Panel balance diagnostics...")
    balance_check(data, opts)

    # Step 1: Classification for each K
    opts.verbose && println("\n[Step 1/4] Classification...")
    classifications = classify_all(data, opts)

    # Step 2: Elbow selection
    opts.verbose && println("\n[Step 2/4] Selecting optimal K...")
    elbow = select_k(classifications, data, opts)
    opts.verbose && println("  Chord method: K = $(elbow.k_chord)")
    opts.verbose && println("  2nd derivative: K = $(elbow.k_secondderiv)")
    opts.verbose && println("  Selected K*: $(elbow.k_star)")

    # Step 3: Second-step estimation at optimal K
    opts.verbose && println("\n[Step 3/4] Second-step estimation (K=$(elbow.k_star))...")
    cr_star = classifications[findfirst(c -> c.k == elbow.k_star, classifications)]
    estimation = second_step_estimate(data, cr_star.groups, elbow.k_star, opts)

    opts.verbose && println("  IV specification: $(estimation.iv_spec)")
    opts.verbose && println("  Nesting parameters (σ): $(round.(estimation.sigma, digits=4))")

    # Step 4: Elasticities
    opts.verbose && println("\n[Step 4/4] Computing elasticities...")

    # Extract price coefficient index
    price_idx = findfirst(==("price"), estimation.coef_names)
    beta_price = estimation.beta_IV[price_idx]

    elas_mat = compute_elasticities(data.shares, cr_star.groups, beta_price,
                                    estimation.sigma, data.J, data.M)

    # Weights: mean market share per product
    weights = vec(mean(data.shares, dims=2))
    elas_stats_df = elasticity_stats(elas_mat, cr_star.groups, weights)
    agg = aggregate_elasticities(elas_mat, cr_star.groups, weights)

    opts.verbose && println("  Mean own-price elasticity: $(round(agg.own_mean, digits=4))")
    opts.verbose && println("  Mean cross-price elasticity: $(round(agg.cross_mean, digits=6))")
    opts.verbose && println("\n" * "=" ^ 60)
    opts.verbose && println("DDNL estimation complete.")
    opts.verbose && println("=" ^ 60)

    return DDNLResult(
        data, opts, classifications, elbow, estimation,
        elas_mat, elas_stats_df, elbow.k_star, cr_star.group_labels
    )
end

"""
    ddnl_from_df(df::DataFrame;
                 y_col::Symbol = :y,
                 price_col::Symbol = :log_price,
                 product_col::Symbol = :product_id,
                 market_col::Symbol = :market_id,
                 share_col::Symbol = :market_share,
                 x_cols::Vector{Symbol} = Symbol[],
                 Z_cols::Vector{Symbol} = Symbol[],
                 kwargs...)

Convenience constructor: build `DDNLData` from a DataFrame and run the full pipeline.

# Arguments
- `df`: Panel data with one row per product-market observation.
- `y_col`: Column with dependent variable `log(s_j/s_0)`.
- `price_col`: Column with (log) prices.
- `product_col`: Column with product identifiers.
- `market_col`: Column with market identifiers.
- `share_col`: Column with market shares.
- `x_cols`: Columns with exogenous product characteristics.
- `Z_cols`: Columns in the DataFrame that form the instrument matrix for price endogeneity.
- `kwargs...`: Passed to `DDNLOptions`.

# Returns
- `DDNLResult`

# Example
```julia
# First build instruments (e.g., using gandhi_houde_iv helper)
df.gh_iv1, df.gh_iv2 = eachcol(gandhi_houde_iv(df, [:log_size, :log_qty], :market_id, :product_id))

result = ddnl_from_df(df;
    y_col = :log_share_ratio,
    price_col = :log_price,
    product_col = :product,
    market_col = :market,
    share_col = :share,
    x_cols = [:log_size, :log_quantity],
    Z_cols = [:gh_iv1, :gh_iv2],
    K_range = 2:8
)
```
"""
function ddnl_from_df(df::DataFrame;
                      y_col::Symbol = :y,
                      price_col::Symbol = :log_price,
                      product_col::Symbol = :product_id,
                      market_col::Symbol = :market_id,
                      share_col::Symbol = :market_share,
                      x_cols::Vector{Symbol} = Symbol[],
                      Z_cols::Vector{Symbol} = Symbol[],
                      kwargs...)

    # Extract options
    opts = DDNLOptions(; kwargs...)

    # Sort data by product, then market
    products = sort(unique(df[!, product_col]))
    markets = sort(unique(df[!, market_col]))
    J = length(products)
    M = length(markets)

    # Create product/market integer mappings
    prod_map = Dict(p => i for (i, p) in enumerate(products))
    mkt_map = Dict(m => i for (i, m) in enumerate(markets))

    # Sort DataFrame
    df_sorted = sort(df, [product_col, market_col])

    # Build arrays
    Y = Vector{Float64}(df_sorted[!, y_col])
    prices = Vector{Float64}(df_sorted[!, price_col])
    product_ids = [prod_map[p] for p in df_sorted[!, product_col]]
    market_ids = [mkt_map[m] for m in df_sorted[!, market_col]]

    # Exogenous characteristics
    if isempty(x_cols)
        error("Must specify at least one exogenous characteristic column via `x_cols`.")
    end
    X = Matrix{Float64}(df_sorted[!, x_cols])
    P = length(x_cols)

    # Market shares (J × M matrix)
    shares = zeros(J, M)
    for row in 1:nrow(df_sorted)
        j = product_ids[row]
        m = market_ids[row]
        shares[j, m] = df_sorted[row, share_col]
    end

    # Extract instrument matrix from DataFrame columns
    if isempty(Z_cols)
        error("Must specify instrument columns via `Z_cols`. Use gandhi_houde_iv() or hausman_iv() to build instruments first, then add them as columns to your DataFrame.")
    end
    Z = Matrix{Float64}(df_sorted[!, Z_cols])

    data = DDNLData(Y, X, prices, shares, product_ids, market_ids,
                    Z, J, M, P)

    return ddnl(data, opts)
end

end # module
