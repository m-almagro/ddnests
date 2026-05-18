# ============================================================================
# instruments.jl — Gandhi-Houde and Hausman instrument construction
# ============================================================================

using DataFrames, Statistics

"""
    gandhi_houde_iv(df::DataFrame, characteristics::Vector{Symbol}, market_col::Symbol, product_col::Symbol)

Construct Gandhi-Houde (GH) instruments for each product-market observation.

For each characteristic `x` and market `m`, the GH instrument for product `j` is:

    GH_{j,x,m} = (1/J_m) * Σ_{k in m} (x_j - x_k)²

This equals the mean squared deviation of product j's characteristic from all products
in the same market, capturing differentiation from competitors.

# Arguments
- `df::DataFrame`: Panel data with product-market observations.
- `characteristics::Vector{Symbol}`: Column names of characteristics to instrument (e.g., `[:log_unit_size, :log_quantity]`).
- `market_col::Symbol`: Column identifying markets.
- `product_col::Symbol`: Column identifying products.

# Returns
- `Matrix{Float64}`: N × length(characteristics) instrument matrix, where N = nrow(df).
"""
function gandhi_houde_iv(df::DataFrame, characteristics::Vector{Symbol},
                         market_col::Symbol, product_col::Symbol)
    n = nrow(df)
    n_chars = length(characteristics)
    Z = zeros(n, n_chars)

    markets = unique(df[!, market_col])

    for mkt in markets
        mask = df[!, market_col] .== mkt
        idx = findall(mask)
        J_m = length(idx)
        J_m <= 1 && continue

        for (c, char) in enumerate(characteristics)
            x = df[idx, char]
            # Sum of squared values and sum of values in this market
            sum_x = sum(x)
            sum_x_sq = sum(x .^ 2)

            for (i, row_idx) in enumerate(idx)
                x_j = x[i]
                # GH_iv = (sum_x_sq - 2*sum_x*x_j + J_m*x_j^2) / J_m
                # This is the mean squared deviation from x_j to all products in market
                Z[row_idx, c] = (sum_x_sq - 2.0 * sum_x * x_j + J_m * x_j^2) / J_m
            end
        end
    end

    return Z
end

"""
    hausman_iv(df::DataFrame, price_col::Symbol, product_col::Symbol, market_col::Symbol)

Construct Hausman instruments: leave-one-out mean price across other markets for the same product.

    hausman_iv_{j,m} = (Σ_{m' ≠ m} price_{j,m'}) / (N_j - 1)

where `N_j` is the number of markets in which product j appears.

# Arguments
- `df::DataFrame`: Panel data with product-market observations.
- `price_col::Symbol`: Column with (log) prices.
- `product_col::Symbol`: Column identifying products.
- `market_col::Symbol`: Column identifying markets.

# Returns
- `Vector{Float64}`: N × 1 instrument vector.
"""
function hausman_iv(df::DataFrame, price_col::Symbol, product_col::Symbol, market_col::Symbol)
    n = nrow(df)
    Z = zeros(n)

    products = unique(df[!, product_col])

    for prod in products
        mask = df[!, product_col] .== prod
        idx = findall(mask)
        n_obs = length(idx)
        n_obs <= 1 && continue

        prices = df[idx, price_col]
        total_price = sum(prices)

        for (i, row_idx) in enumerate(idx)
            # Leave-one-out mean: (total - own) / (n - 1)
            Z[row_idx] = (total_price - prices[i]) / (n_obs - 1)
        end
    end

    return Z
end

"""
    build_instruments(df, characteristics, price_col, product_col, market_col; iv_type=:GH)

Deprecated. Use `gandhi_houde_iv` or `hausman_iv` directly and pass the resulting
matrix as `Z` to `DDNLData`.

# Example (new API)
```julia
Z = gandhi_houde_iv(df, [:log_size, :log_qty], :market_id, :product_id)
data = DDNLData(Y, X, prices, shares, product_id, market_id, Z, J, M, P)
```
"""
function build_instruments(df::DataFrame;
                           characteristics::Vector{Symbol} = Symbol[],
                           iv_type::Symbol = :GH,
                           price_col::Symbol = :log_price,
                           product_col::Symbol = :product_id,
                           market_col::Symbol = :market_id)
    @warn "build_instruments is deprecated. Use gandhi_houde_iv() or hausman_iv() directly and pass the result as Z to DDNLData."
    if iv_type == :GH
        if isempty(characteristics)
            error("Gandhi-Houde instruments require specifying `characteristics`.")
        end
        return gandhi_houde_iv(df, characteristics, market_col, product_col)
    elseif iv_type == :hausman
        z = hausman_iv(df, price_col, product_col, market_col)
        return reshape(z, :, 1)
    else
        error("Unknown iv_type: $(iv_type)")
    end
end
