# ============================================================================
# monte_carlo_example.jl — Monte Carlo simulation with known nesting structure
#
# Generates synthetic nested logit data with K=3 true groups, introduces
# random missing values (~20%), runs the DDNL pipeline, and verifies
# recovery of the true K=3.
# ============================================================================

using ddnests
using Random, Statistics, LinearAlgebra, DataFrames

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
J = 100         # Number of products
M = 30          # Number of markets
K_true = 3      # True number of groups
P = 2           # Number of exogenous characteristics
missing_frac = 0.20  # Fraction of observations to set as missing

# True parameters
true_sigma = [0.3, 0.6, 0.8]   # Nesting parameters per group
true_alpha = -2.0               # Price coefficient
true_beta = [1.5, -0.5]        # Coefficients on exogenous chars

# True group assignments (products 1-5 in group 1, 6-10 in group 2, 11-15 in group 3)
true_groups = vcat(fill(1, 5), fill(2, 5), fill(3, 5))

rng = Xoshiro(12345)

# --------------------------------------------------------------------------
# Generate synthetic data
# --------------------------------------------------------------------------
println("Generating synthetic nested logit data...")
println("  J=$J products, M=$M markets, K_true=$K_true groups")
println("  True groups: ", true_groups)
println("  True sigma: ", true_sigma)

# Product characteristics (fixed across markets)
X_chars = randn(rng, J, P)

# Prices with market variation
prices_base = 2.0 .+ 0.5 .* randn(rng, J)
prices = zeros(J * M)
X_full = zeros(J * M, P)
product_ids = zeros(Int, J * M)
market_ids = zeros(Int, J * M)

for j in 1:J
    for m in 1:M
        row = (j - 1) * M + m
        prices[row] = prices_base[j] + 0.3 * randn(rng)
        X_full[row, :] = X_chars[j, :]
        product_ids[row] = j
        market_ids[row] = m
    end
end

# Generate shares using the nested logit model
shares = zeros(J, M)
for m in 1:M
    # Compute utilities
    V = zeros(J)
    for j in 1:J
        row = (j - 1) * M + m
        V[j] = dot(true_beta, X_chars[j, :]) + true_alpha * prices[row] + 0.5 * randn(rng)
    end

    # Compute nested logit shares
    for k in 1:K_true
        products_in_k = findall(true_groups .== k)
        sigma_k = true_sigma[k]

        # Inclusive value for nest k
        IV_k = sum(exp.(V[products_in_k] ./ sigma_k))

        for j in products_in_k
            # Within-nest probability * nest probability
            s_jk = exp(V[j] / sigma_k) / IV_k
            nest_prob = IV_k^sigma_k / (1.0 + sum(
                sum(exp.(V[findall(true_groups .== kk)] ./ true_sigma[kk]))^true_sigma[kk]
                for kk in 1:K_true
            ))
            shares[j, m] = s_jk * nest_prob
        end
    end

    # Normalize to ensure they sum to < 1
    total = sum(shares[:, m])
    if total >= 1.0
        shares[:, m] ./= (total + 0.1)
    end
end

# Build Y = log(s_j / s_0) where s_0 = 1 - sum(s_j)
Y = zeros(J * M)
for j in 1:J
    for m in 1:M
        row = (j - 1) * M + m
        s0 = 1.0 - sum(shares[:, m])
        if shares[j, m] > 0 && s0 > 0
            Y[row] = log(shares[j, m] / s0)
        end
    end
end

# --------------------------------------------------------------------------
# Introduce missing values (~20%)
# --------------------------------------------------------------------------
n_obs = J * M
n_missing_target = round(Int, missing_frac * n_obs)
missing_indices = sort(shuffle(rng, collect(1:n_obs))[1:n_missing_target])

println("\nIntroducing $(length(missing_indices)) missing observations ($(round(100*missing_frac, digits=0))% of panel)...")

Y_missing = copy(Y)
shares_missing = copy(shares)
for idx in missing_indices
    Y_missing[idx] = 0.0
    j = product_ids[idx]
    m = market_ids[idx]
    shares_missing[j, m] = 0.0
end

# --------------------------------------------------------------------------
# Build instruments (GH-style from characteristics)
# --------------------------------------------------------------------------
# Build GH instruments manually for this synthetic data
Z_instruments = zeros(J * M, P)
for m in 1:M
    rows_in_m = [(j - 1) * M + m for j in 1:J]
    for p in 1:P
        x_vals = X_full[rows_in_m, p]
        sum_x = sum(x_vals)
        sum_x_sq = sum(x_vals .^ 2)
        for (i, row) in enumerate(rows_in_m)
            x_j = x_vals[i]
            Z_instruments[row, p] = (sum_x_sq - 2.0 * sum_x * x_j + J * x_j^2) / J
        end
    end
end

# --------------------------------------------------------------------------
# Run DDNL pipeline
# --------------------------------------------------------------------------
println("\nConstructing DDNLData...")
data = DDNLData(Y_missing, X_full, prices, shares_missing, product_ids, market_ids,
    Z_instruments, J, M, P)

println("Running DDNL pipeline (K_range=2:6)...\n")
result = ddnl(data;
    K_range=2:6,
    n_starts=500,
    k_selection=:in_sample_elbow,
    verbose=true
)

# --------------------------------------------------------------------------
# Verify results
# --------------------------------------------------------------------------
println("\n" * "="^60)
println("VERIFICATION")
println("="^60)
println("True K: $K_true")
println("Estimated K*: $(result.optimal_k)")
println("True groups: ", true_groups)
println("Estimated groups: ", result.group_labels)

if result.optimal_k == K_true
    println("\nSUCCESS: Correctly recovered K=$K_true groups!")
else
    println("\nNOTE: Estimated K=$(result.optimal_k) differs from true K=$K_true.")
    println("This can happen with missing data or limited sample size.")
end

println("\nTrue sigma: ", true_sigma)
println("Estimated sigma: ", round.(result.estimation.sigma, digits=4))
println("\nElbow details:")
println("  Chord method K: $(result.elbow.k_chord)")
println("  2nd derivative K: $(result.elbow.k_secondderiv)")
println("  SSR by K: ", round.(result.elbow.ssr_by_k, digits=2))
