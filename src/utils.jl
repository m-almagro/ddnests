# ============================================================================
# utils.jl — Helper utilities for ddnests.jl
# ============================================================================

using LinearAlgebra, SparseArrays, Statistics, DataFrames

# --------------------------------------------------------------------------
# Weighted statistics
# --------------------------------------------------------------------------

"""
    wmean(x, w)

Weighted mean of `x` with weights `w`, skipping NaN values.
"""
function wmean(x::AbstractVector, w::AbstractVector)
    valid = .!isnan.(x) .& .!isnan.(w)
    any(valid) || return NaN
    return sum(x[valid] .* w[valid]) / sum(w[valid])
end

"""
    wstd(x, w)

Weighted standard deviation of `x` with weights `w`, skipping NaN values.
"""
function wstd(x::AbstractVector, w::AbstractVector)
    valid = .!isnan.(x) .& .!isnan.(w)
    sum(valid) > 1 || return NaN
    mu = wmean(x[valid], w[valid])
    sw = sum(w[valid])
    return sqrt(sum(w[valid] .* (x[valid] .- mu).^2) / sw)
end

"""
    wmedian(x, w)

Weighted median of `x` with weights `w`, skipping NaN/missing values.
"""
function wmedian(x::AbstractVector, w::AbstractVector)
    valid = .!isnan.(x) .& .!isnan.(w) .& (w .> 0)
    any(valid) || return NaN
    xv, wv = x[valid], w[valid]
    idx = sortperm(xv)
    xv, wv = xv[idx], wv[idx]
    cum = cumsum(wv)
    cum ./= cum[end]
    i = findfirst(>=(0.5), cum)
    return xv[i]
end

# --------------------------------------------------------------------------
# Dummy variable construction
# --------------------------------------------------------------------------

"""
    make_dummies(v::AbstractVector)

Create a one-hot dummy matrix from integer vector `v` (drops first category).
Returns `(dummies, categories)` where `dummies` is N × (n_categories - 1).
"""
function make_dummies(v::AbstractVector{<:Integer})
    cats = sort(unique(v))
    n = length(v)
    ncats = length(cats) - 1  # drop first category
    D = zeros(Int, n, ncats)
    cat_map = Dict(c => i for (i, c) in enumerate(cats))
    for i in 1:n
        idx = cat_map[v[i]]
        if idx > 1
            D[i, idx - 1] = 1
        end
    end
    return D, cats[2:end]
end

"""
    make_market_fe(market_id::AbstractVector{Int}, M::Int)

Create market fixed-effect dummies (drops first market). Returns sparse N × (M-1) matrix.
"""
function make_market_fe(market_id::AbstractVector{Int}, M::Int)
    markets = sort(unique(market_id))
    n = length(market_id)
    market_map = Dict(m => i for (i, m) in enumerate(markets))
    I_idx = Int[]
    J_idx = Int[]
    for i in 1:n
        idx = market_map[market_id[i]]
        if idx > 1
            push!(I_idx, i)
            push!(J_idx, idx - 1)
        end
    end
    return sparse(I_idx, J_idx, ones(Int, length(I_idx)), n, M - 1)
end

# --------------------------------------------------------------------------
# Group indicator utilities
# --------------------------------------------------------------------------

"""
    group_dummies_sparse(gi::Matrix{Int}, J::Int, M::Int, K::Int)

Create sparse group-market indicator matrix (JM × KM) from group indicator `gi` (J × K).
For product j in group k and market m: row = (j-1)*M + m, col = (m-1)*K + k.

Handles multi-group assignment (ties): if gi[j,:] has multiple nonzero entries,
entries are created for ALL active groups (matching paper's group_dummies).
"""
function group_dummies_sparse(gi::Matrix{Int}, J::Int, M::Int, K::Int)
    I_idx = Int[]
    J_idx = Int[]
    V = Int[]
    for j in 1:J
        active_groups = findall(!iszero, gi[j, :])
        for k in active_groups
            val = gi[j, k]
            for m in 1:M
                row = (j - 1) * M + m
                col = (m - 1) * K + k
                push!(I_idx, row)
                push!(J_idx, col)
                push!(V, val)
            end
        end
    end
    return sparse(I_idx, J_idx, V, J * M, K * M)
end

"""
    group_regressors_sparse(gi::Matrix{Int}, X_hat::Matrix{Float64}, J::Int, M::Int, K::Int)

Create sparse design matrix (JM × Q*K) interacting regressors with group membership.
Element = X_hat[row, q] if product in row belongs to group k, placed in column (q-1)*K + k.
Uses q-major ordering to match paper: reshape(par, K, Q)' recovers the Q × K coefficient matrix.
"""
function group_regressors_sparse(gi::Matrix{Int}, X_hat::Matrix{Float64}, J::Int, M::Int, K::Int)
    Q = size(X_hat, 2)
    n = J * M
    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]
    for j in 1:J
        active_groups = findall(!iszero, gi[j, :])
        for k in active_groups
            g_val = gi[j, k]
            for q in 1:Q
                for m in 1:M
                    row = (j - 1) * M + m
                    val = X_hat[row, q]
                    if abs(val) > 1e-12
                        push!(I_idx, row)
                        push!(J_idx, (q - 1) * K + k)
                        push!(V, val * g_val)
                    end
                end
            end
        end
    end
    return sparse(I_idx, J_idx, V, n, Q * K)
end

"""
    labels_to_indicator(labels::AbstractVector{Int}, K::Int)

Convert integer group labels (1…K) to one-hot indicator matrix (J × K).
"""
function labels_to_indicator(labels::AbstractVector{Int}, K::Int)
    J = length(labels)
    gi = zeros(Int, J, K)
    for j in 1:J
        gi[j, labels[j]] = 1
    end
    return gi
end

"""
    indicator_to_labels(gi::Matrix{Int})

Convert one-hot indicator matrix (J × K) to integer group labels.
"""
function indicator_to_labels(gi::Matrix{Int})
    J = size(gi, 1)
    labels = zeros(Int, J)
    for j in 1:J
        labels[j] = findfirst(==(1), gi[j, :])
    end
    return labels
end

# --------------------------------------------------------------------------
# Conditional probabilities
# --------------------------------------------------------------------------

"""
    conditional_probs(shares::Matrix{Float64}, gi::Matrix{Int}, J::Int, M::Int)

Compute within-nest conditional market shares: `s_{j|g} = s_j / Σ_{i∈g} s_i`.
Returns vector of length J*M with `log(s_{j|g})`.
"""
function conditional_probs(shares::Matrix{Float64}, gi::Matrix{Int}, J::Int, M::Int)
    K = size(gi, 2)
    log_cond = zeros(J * M)
    for m in 1:M
        for k in 1:K
            products_in_k = findall(==(1), gi[:, k])
            group_share = sum(shares[products_in_k, m])
            for j in products_in_k
                row = (j - 1) * M + m
                if group_share > 0 && shares[j, m] > 0
                    log_cond[row] = log(shares[j, m] / group_share)
                else
                    log_cond[row] = NaN
                end
            end
        end
    end
    return log_cond
end

# --------------------------------------------------------------------------
# Robust linear algebra
# --------------------------------------------------------------------------

"""
    robust_solve(X::AbstractMatrix, y::AbstractVector; tol=1e-10)

Solve `X \\ y` with fallback to dense QR with column pivoting for rank-deficient systems.
Matches the paper's `robust_lm` exactly:
1. Try sparse `\\` first
2. If result has NaN, try sparse QR to check rank
3. If rank-deficient, fall back to dense pivoted QR with relative tolerance
"""
function robust_solve(X::AbstractMatrix, y::AbstractVector; tol=1e-10)
    # Step 1: Try sparse solve (matching paper line 1086: Ztot_filtered \\ Y_filtered)
    par_new = try
        X \ y
    catch
        nothing
    end

    # Step 2: Check for NaN (matching paper line 1087: if any(isnan, par_new))
    if par_new !== nothing && !any(isnan, par_new)
        return par_new
    end

    # Step 3: robust_lm fallback (matching paper lines 650-688)
    # First try sparse QR to check rank
    try
        F_sparse = qr(X)
        R_diag = abs.(diag(F_sparse.R))
        min_diag = minimum(R_diag)
        max_diag = maximum(R_diag)
        if min_diag > tol * max_diag
            # Matrix appears full rank, use fast sparse solve
            return X \ y
        end
    catch
        # Sparse QR failed, proceed to dense
    end

    # Dense pivoted QR (matching paper lines 673-687)
    Xd = Matrix(X)
    F = qr(Xd, ColumnNorm())
    R_diag = abs.(diag(F.R))
    r = sum(R_diag .> tol * maximum(R_diag))  # relative tolerance, matching paper

    beta = zeros(size(Xd, 2))
    beta[F.p[1:r]] = F.R[1:r, 1:r] \ (Matrix(F.Q)' * y)[1:r]
    return beta
end

"""
    common_columns(X::Matrix, Z::Matrix)

Find columns in X that also appear (exactly) in Z. Returns the column indices in X.
"""
function common_columns(X::Matrix, Z::Matrix)
    idx = Int[]
    for jx in 1:size(X, 2)
        for jz in 1:size(Z, 2)
            if X[:, jx] ≈ Z[:, jz]
                push!(idx, jx)
                break
            end
        end
    end
    return idx
end

"""
    endogenous_columns(X::Matrix, Z::Matrix)

Find columns in X that do NOT appear in Z (the endogenous regressors).
"""
function endogenous_columns(X::Matrix, Z::Matrix)
    common = common_columns(X, Z)
    return setdiff(1:size(X, 2), common)
end
