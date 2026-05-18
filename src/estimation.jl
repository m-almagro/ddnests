# ============================================================================
# estimation.jl — Second-step nested logit estimation (OLS + IV)
#
# Matches the paper's second_step() function:
#   - Explicit MFE regressors (full M columns, no FWL)
#   - Group IVs built from full X_exo (including MFE)
#   - QR-based 2SLS with robust standard errors
# ============================================================================

using LinearAlgebra, Statistics, DataFrames

"""
    build_group_log_cond_probs(shares, gi, J, M)

Build log conditional probability regressors: log(s_j / Σ_{i∈k} s_i).
Returns JM × K matrix, nonzero only for products in each group.
"""
function build_group_log_cond_probs(shares::Matrix{Float64}, gi::Matrix{Int}, J::Int, M::Int)
    K = size(gi, 2)
    log_cond = zeros(J * M, K)
    for m in 1:M, k in 1:K
        products_in_k = findall(==(1), gi[:, k])
        group_share = sum(shares[products_in_k, m])
        for j in products_in_k
            row = (j - 1) * M + m
            if group_share > 0 && shares[j, m] > 0
                log_cond[row, k] = log(shares[j, m] / group_share)
            end
        end
    end
    return log_cond
end

"""
    build_group_iv(X_exo, gi, J, M)

Build group IVs from exogenous characteristics (matching paper).
For each variable p and group k:  IV = log(exp(x_jp) / Σ_{i∈k} exp(x_ip))
Returns JM × (P_exo * K) matrix.
"""
function build_group_iv(X_exo::Matrix{Float64}, gi::Matrix{Int}, J::Int, M::Int)
    K = size(gi, 2)
    P_exo = size(X_exo, 2)
    IV = zeros(J * M, P_exo * K)

    for p in 1:P_exo
        exp_x = exp.(X_exo[:, p])
        exp_x_r = reshape(exp_x, M, J)'  # J × M
        sum_exp = gi' * exp_x_r           # K × M

        for k in 1:K
            products_in_k = findall(==(1), gi[:, k])
            for j in products_in_k, m in 1:M
                row = (j - 1) * M + m
                if sum_exp[k, m] > 0 && exp_x_r[j, m] > 0
                    IV[row, (p-1)*K + k] = log(exp_x_r[j, m] / sum_exp[k, m])
                end
            end
        end
    end
    return IV
end

"""
    iv_2sls(Y, X, Z)

Two-stage least squares with QR-based projection and robust standard errors.
Validated to match FixedEffectModels.reg() on the beer market data.
"""
function iv_2sls(Y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64})
    N = length(Y)
    P = size(X, 2)

    # QR projection: P_Z = Q_r * Q_r'
    F = qr(Z, ColumnNorm())
    R_diag = abs.(diag(F.R))
    tol = max(size(Z)...) * eps(Float64) * maximum(R_diag)
    r = sum(R_diag .> tol)
    r == 0 && return zeros(P), fill(Inf, P)

    Q_mat = Matrix(F.Q)[:, 1:r]
    X_hat = Q_mat * (Q_mat' * X)

    # Second stage
    XhX = Symmetric(X_hat' * X)
    beta = try
        cholesky(XhX) \ (X_hat' * Y)
    catch
        pinv(Matrix(XhX)) * (X_hat' * Y)
    end

    # Robust SEs (White/sandwich)
    resid = Y - X * beta
    S = X_hat' * (resid.^2 .* X_hat)
    XhX_inv = try inv(cholesky(XhX)) catch; pinv(Matrix(XhX)) end
    V = XhX_inv * S * XhX_inv
    se = sqrt.(abs.(diag(V)))
    return beta, se
end

"""
    ols_robust(Y, X)

OLS with HC1 robust standard errors.
"""
function ols_robust(Y::Vector{Float64}, X::Matrix{Float64})
    N = length(Y)
    P = size(X, 2)
    beta = X \ Y
    resid = Y - X * beta
    XtX_inv = pinv(X' * X)
    S = X' * (resid.^2 .* X)
    V = XtX_inv * S * XtX_inv .* (N / max(N - P, 1))
    se = sqrt.(abs.(diag(V)))
    return beta, se
end

"""
    second_step_estimate(data, gi, K, opts)

Second-step structural estimation matching the paper's `second_step()` exactly:
1. X_exo = [MFE(M), exogenous chars] — explicit market FEs, full M columns
2. Group IVs built from full X_exo (including MFE columns)
3. QR-based 2SLS (validated to match FixedEffectModels.reg())
"""
function second_step_estimate(data::DDNLData, gi::Matrix{Int}, K::Int, opts::DDNLOptions)
    J, M = data.J, data.M
    Y = data.Y
    non_missing = Y .!= 0

    # --- X_exo: [MFE(M), exogenous chars] matching paper ---
    X_chars = copy(data.X)
    if opts.intercept_type == :market
        mfe = kron(ones(J), Matrix{Float64}(I, M, M))  # JM × M (full, no drop)
        X_exo = hcat(mfe, X_chars)
    elseif opts.intercept_type == :baseline
        X_exo = hcat(ones(J * M), X_chars)
    else
        X_exo = X_chars
    end

    # Endogenous price
    prices_col = reshape(data.prices, :, 1)

    # Log conditional probabilities (JM × K)
    log_cond_prob = build_group_log_cond_probs(data.shares, gi, J, M)

    # Group IVs from full X_exo (matching paper)
    IV_group = build_group_iv(X_exo, gi, J, M)

    # --- Assemble X_full and Z_full ---
    # X_full = [X_exo, price, log_cond_prob]
    X_full = hcat(X_exo, prices_col, log_cond_prob)

    # Z_full = [X_exo (own instruments), Z (excluded price IVs), IV_group (excluded)]
    Z_full = hcat(X_exo, data.Z, IV_group)

    # Filter to non-missing
    Y_f      = Y[non_missing]
    X_full_f = X_full[non_missing, :]
    Z_full_f = Z_full[non_missing, :]

    # --- OLS ---
    beta_OLS, se_OLS = ols_robust(Y_f, X_full_f)

    # --- Full IV (2SLS) ---
    beta_IV, se_IV = iv_2sls(Y_f, X_full_f, Z_full_f)

    # Transform: σ = 1 - coef on log_cond_prob
    beta_OLS[end-K+1:end] = 1.0 .- beta_OLS[end-K+1:end]
    beta_IV[end-K+1:end]  = 1.0 .- beta_IV[end-K+1:end]

    sigma    = beta_IV[end-K+1:end]
    sigma_se = se_IV[end-K+1:end]

    # Build coefficient names
    n_exo = size(X_exo, 2)
    coef_names = String[]
    for i in 1:n_exo; push!(coef_names, "x_exo_$i"); end
    push!(coef_names, "price")
    for k in 1:K; push!(coef_names, "sigma_$k"); end

    return EstimationResult(
        K, beta_OLS, se_OLS, beta_IV, se_IV,
        sigma, sigma_se, coef_names, :full
    )
end
