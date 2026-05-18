# ddnests.jl

A Julia package for **Data-Driven Nested Logit (DDNL)** demand estimation. It jointly estimates the nesting structure and structural parameters of nested logit models from market-level data, without requiring the researcher to pre-specify product groupings.

## Overview

Traditional nested logit models require the researcher to define product groups (nests) a priori. `ddnests.jl` treats the nesting structure as an unknown object to be estimated from the data, using an EM-based classification algorithm combined with a distance-elbow method for selecting the optimal number of groups.

The package implements the full estimation pipeline:

1. **Product Classification** — EM algorithm and k-means clustering to assign products to nests
2. **K Selection** — Distance-elbow method (in-sample or cross-validated) to determine the optimal number of groups
3. **Structural Estimation** — Second-step OLS and IV (2SLS) estimation of nested logit parameters
4. **Elasticity Computation** — Own-price, within-nest cross-price, and across-nest cross-price elasticities
5. **Visualization** — Elbow plots and elasticity heatmaps

Instrument construction utilities (`gandhi_houde_iv`, `hausman_iv`) are provided as helpers; users build their instrument matrix externally and pass it directly to the package.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/m-almagro/ddnests")
```

Or in development mode:

```julia
] dev path/to/ddnests
```

## Quick Start

```julia
using ddnests, DataFrames, CSV

# Load your panel data (one row per product-market observation)
df = DataFrame(CSV.File("your_data.csv"))

# Step 1: Build instruments externally (user's choice of strategy)
Z_mat = gandhi_houde_iv(df, [:log_size, :log_qty], :market_id, :product_id)
# Add instrument columns to DataFrame
df.gh_iv1 = Z_mat[:, 1]
df.gh_iv2 = Z_mat[:, 2]

# Step 2: Run the full DDNL pipeline
result = ddnl_from_df(df;
    y_col          = :log_share_ratio,    # log(s_j / s_0)
    price_col      = :log_price,          # log prices
    product_col    = :product_id,         # product identifier
    market_col     = :market_id,          # market identifier
    share_col      = :market_share,       # market shares
    x_cols         = [:log_size, :log_qty], # exogenous characteristics
    Z_cols         = [:gh_iv1, :gh_iv2],  # instrument columns
    K_range        = 2:10,                # range of group counts to evaluate
    k_selection    = :in_sample_elbow,    # or :out_of_sample_elbow
    n_starts       = 2000,                # number of EM random restarts
    verbose        = true
)

# Access results
println("Optimal K: ", result.optimal_k)
println("Nesting parameters: ", result.estimation.sigma)
println("Group assignments: ", result.group_labels)

# Generate plots
plot_elbow(result.elbow; style=:combined, save_path="elbow.pdf")
plot_elasticity_heatmap(result.elasticity_matrix,
    result.classifications[findfirst(c -> c.k == result.optimal_k, result.classifications)].groups;
    save_path="elasticity_heatmap.png")
```

## K Selection Methods

The package provides two approaches for selecting the optimal number of groups:

### In-sample elbow (`:in_sample_elbow`, default)

Uses two complementary heuristics on the SSR-vs-K curve:
- **Chord method**: Maximum perpendicular distance to the chord connecting endpoints.
- **Second-derivative method**: Maximum deceleration of SSR percentage drops.

The package always selects the **maximum** K among sigma-validated candidates from both methods. This avoids under-fitting when the K_range is narrow.

**Note on K_range**: With a narrow range (e.g., 2:5), the chord method has few points and may underestimate K. Use `K_range=2:10` for robust selection.

### Out-of-sample cross-validation (`:out_of_sample_elbow`)

Splits markets into `cv_folds` folds. For each K:
1. Holds out a fold of markets (sets Y=0 for those markets).
2. Estimates group-specific parameters on the remaining markets.
3. Evaluates prediction error on the held-out markets.

Selects K that minimizes average out-of-sample MSE. More robust to overfitting but computationally more expensive.

```julia
result = ddnl(data; k_selection=:out_of_sample_elbow, cv_folds=10)
```

## Missing Data Handling

The package handles unbalanced panels where some products are absent from certain markets:

- Observations with `Y = 0` are treated as missing (absent from that market).
- A **balance check** is run before classification, reporting coverage statistics and warnings.
- Missing observations are excluded from parameter estimation but the panel structure is preserved.
- If any product appears in fewer than 2 markets, a warning is issued (classification may be unreliable).

## Methodology

### The Nested Logit Model

The nested logit model partitions J products into K groups (nests). Consumer i's indirect utility for product j in market m is:

```
u_{ijm} = x_j' B + a p_{jm} + xi_{jm} + zeta_{igm} + (1 - sigma_g) eps_{ijm}
```

where sigma_g in (0,1) is the nesting parameter for group g.

The log market share ratio takes the linear form:

```
log(s_{jm}) - log(s_{0m}) = x_j' B + a p_{jm} + sigma_g log(s_{j|g,m}) + xi_{jm}
```

### Classification Algorithm

The package uses an EM-type algorithm:

**E-step**: Given group assignments, estimate group-specific intercepts and slopes via sparse regression.

**M-step**: Given parameters, assign each product to the group minimizing its MSE.

The algorithm runs from many random initializations (parallel via `Threads.@threads`) and selects the solution with lowest SSR.

### Instrumental Variables

The user supplies their own instrument matrix Z for price endogeneity. Helper functions are provided:

**`gandhi_houde_iv`**: Mean squared deviation of product characteristics from competitors in the same market.

**`hausman_iv`**: Leave-one-out mean price across other markets for the same product.

Group instruments for within-nest shares are built internally during estimation.

## Data Requirements

Your input DataFrame must contain:

| Column | Description |
|--------|-------------|
| Product ID | Unique product identifier (integer or string) |
| Market ID | Unique market identifier (integer or string) |
| `y` | Dependent variable: `log(s_j / s_0)` where `s_0` is the outside option share |
| Price | (Log) prices |
| Market share | Product-level market shares `s_j` |
| Characteristics | At least one exogenous product characteristic |
| Instruments | Columns forming the price instrument matrix Z |

**Data structure:** The data should be a balanced or unbalanced panel with one row per product-market observation. Products not present in a market should have `y = 0` (the package treats zeros as missing).

## API Reference

### Types

#### `DDNLData`
Input data container.
- `Y::Vector{Float64}` — Dependent variable log(s_j/s_0)
- `X::Matrix{Float64}` — Exogenous characteristics (JM x P)
- `prices::Vector{Float64}` — Observed prices (JM x 1)
- `shares::Matrix{Float64}` — Market shares (J x M)
- `product_id::Vector{Int}` — Product identifiers
- `market_id::Vector{Int}` — Market identifiers
- `Z::Matrix{Float64}` — Instrument matrix for price endogeneity (user-supplied)
- `J::Int` — Number of products
- `M::Int` — Number of markets
- `P::Int` — Number of characteristics

#### `DDNLOptions`
Configuration for the estimation pipeline.
```julia
DDNLOptions(;
    K_range = 2:10,            # range of group counts to evaluate
    tol = 1e-5,                # EM convergence tolerance
    max_iter = 1000,           # max EM iterations per run
    n_starts = 2000,           # number of random restarts
    n_kmeans = 0,              # number of k-means warm-starts
    k_selection = :in_sample_elbow,  # :in_sample_elbow or :out_of_sample_elbow
    cv_folds = 10,             # number of CV folds (for out-of-sample)
    method = :kmeans,          # :kmeans or :em
    intercept_type = :market,  # :market, :baseline, or :none
    group_dummies_utility = false,  # include K-1 group dummies in utility
    verbose = true
)
```

### Functions

#### High-Level Pipeline

```julia
ddnl(data::DDNLData; kwargs...) -> DDNLResult
ddnl(data::DDNLData, opts::DDNLOptions) -> DDNLResult
ddnl_from_df(df::DataFrame; y_col, price_col, product_col, market_col,
             share_col, x_cols, Z_cols, kwargs...) -> DDNLResult
```

#### Instrument Helpers

```julia
gandhi_houde_iv(df, characteristics, market_col, product_col) -> Matrix{Float64}
hausman_iv(df, price_col, product_col, market_col) -> Vector{Float64}
```

Users call these to build their instrument matrix, then pass it via `Z_cols` (in `ddnl_from_df`) or directly as the `Z` field in `DDNLData`.

#### Classification

```julia
classify(data::DDNLData, opts::DDNLOptions, K::Int) -> ClassificationResult
classify_all(data::DDNLData, opts::DDNLOptions) -> Vector{ClassificationResult}
```

#### Elbow Selection

```julia
find_elbow_chord(k_vals, ssr_vals) -> (k_star, distances)
find_elbow_secondderiv(k_vals, ssr_vals) -> (k_star, pct_drops, second_derivs)
select_k(classifications, data, opts) -> ElbowResult
select_k_cv(classifications, data, opts) -> ElbowResult
```

#### Second-Step Estimation

```julia
second_step_estimate(data, gi, K, opts) -> EstimationResult
ols_robust(Y, X) -> (beta, se)
iv_2sls(Y, X, Z) -> (beta, se)
```

#### Elasticities

```julia
compute_elasticities(shares, gi, beta_price, sigma, J, M) -> Matrix{Float64}
elasticity_stats(elas_mat, gi, weights) -> DataFrame
aggregate_elasticities(elas_mat, gi, weights) -> NamedTuple
elasticity_dataframe(elas_mat) -> DataFrame
```

#### Balance Check

```julia
balance_check(data::DDNLData, opts::DDNLOptions) -> NamedTuple
```

Reports panel balance diagnostics: min/mean/max markets per product, and warns about missing data.

## Performance Tips

- **Use multiple threads.** The classification step parallelizes EM restarts across threads. Start Julia with:
  ```bash
  julia --threads=auto
  ```

- **Adjust `n_starts` to your problem size.** For small problems (J < 20), 100-500 starts suffice. For large problems (J > 50), use 2000+.

- **Use `K_range=2:10` for robust K selection.** Narrow ranges give the chord method too few points.

- **Use `:out_of_sample_elbow` for high-dimensional problems.** Cross-validation is more robust but slower.

## Package Structure

```
ddnests/
├── Project.toml
├── README.md
├── src/
│   ├── ddnests.jl   # Main module, high-level API
│   ├── types.jl             # All type definitions
│   ├── utils.jl             # Weighted stats, dummies, sparse helpers
│   ├── instruments.jl       # GH and Hausman IV helpers
│   ├── classification.jl    # EM algorithm and k-means classification
│   ├── elbow.jl             # Distance-elbow and CV K selection
│   ├── estimation.jl        # Second-step OLS + IV estimation
│   ├── elasticities.jl      # Elasticity computation and statistics
│   └── plotting.jl          # Elbow plots and elasticity heatmaps
├── test/
└── examples/
    ├── quickstart.jl
    └── monte_carlo_example.jl
```

## Citation

If you use this package in your research, please cite:

```
@article{ddnl2026,
  title={Data-Driven Nested Logit},
  author={...},
  year={2026}
}
```

## License

MIT
