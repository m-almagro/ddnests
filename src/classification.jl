# ============================================================================
# classification.jl — EM and K-means classification engine
# ============================================================================

using LinearAlgebra, SparseArrays, Statistics, Random, StatsBase, Distributions

"""
    price_first_stage(X::Matrix{Float64}, Z::Matrix{Float64}, J::Int, M::Int)

Estimate first-stage price regression: regress price (last column of X) on instruments Z.
Returns predicted prices as a vector of length J*M.
"""
function price_first_stage(X::Matrix{Float64}, Z::Matrix{Float64}, J::Int, M::Int)
    p = X[:, end]
    # OLS: p = Z * γ + ε → p̂ = Z * (Z'Z)⁻¹Z'p
    gamma = Z \ p
    p_hat = Z * gamma
    return p_hat
end

"""
    initial_condition_random(J::Int, K::Int, Y::Vector{Float64}, M::Int, rng::AbstractRNG)

Generate a random product-to-group assignment ensuring no empty group-market cells.
Returns a J × K one-hot indicator matrix.
"""
function initial_condition_random(J::Int, K::Int, Y::Vector{Float64}, M::Int, rng::AbstractRNG)
    gi_init = zeros(Int, J, K)
    for j in 1:J
        gi_init[j, rand(rng, 1:K)] = 1
    end
    gi_init = wise_swap!(gi_init, Y, J, M, K, rng)
    return gi_init
end

"""
    wise_swap!(gi::Matrix{Int}, Y::Vector{Float64}, J::Int, M::Int, K::Int, rng::AbstractRNG)

Fix empty group-market cells by swapping products between groups.
Ensures every group has non-missing observations in every market.
"""
function wise_swap!(gi::Matrix{Int}, Y::Vector{Float64}, J::Int, M::Int, K::Int, rng::AbstractRNG)
    non_missing = Y .!= 0

    # Build initial group-market indicators (matching paper's pre-loop setup)
    gi_aux = copy(gi)
    gitot = group_dummies_sparse(gi_aux, J, M, K)
    gitot_filtered = Matrix{Int}(gitot[non_missing, :])
    group_sums = vec(sum(gitot_filtered, dims=1))

    attempt = 0
    while minimum(group_sums) == 0 && attempt < 100
        attempt += 1

        # Reset gi_aux from original gi on each attempt (matching paper line 942)
        gi_aux = copy(gi)

        # Find an empty group-market cell
        empty_groups = findall(iszero, group_sums)
        empty_gm = rand(rng, empty_groups)
        old_m = cld(empty_gm, K)
        old_g = mod1(empty_gm, K)

        # Try to swap a product from another group into old_g
        candidates = collect(setdiff(1:K, old_g))
        shuffle!(rng, candidates)

        j_candidate = 0
        idx = 1
        while j_candidate == 0 && idx <= length(candidates)
            new_g = candidates[idx]
            non_missing_market = reshape(non_missing, M, J)[old_m, :]
            eligible = findall(Vector{Int}(gi_aux[:, new_g] .* non_missing_market) .== 1)
            if !isempty(eligible)
                j_candidate = rand(rng, eligible)
            end
            idx += 1
        end

        if j_candidate == 0
            # No swap found — fill empty column in gitot_filtered (matching paper line 971)
            empty_column = findmin(vec(sum(gitot_filtered, dims=1)))[2]
            gitot_filtered[:, empty_column] .= 1
        else
            gi_aux[j_candidate, :] .= 0
            gi_aux[j_candidate, old_g] = 1
            # Rebuild dummies after swap (matching paper lines 975-976)
            gitot_aux = group_dummies_sparse(gi_aux, J, M, K)
            gitot_filtered = Matrix{Int}(gitot_aux[non_missing, :])
        end

        # NOTE: paper does NOT recompute group_sums here.
        # group_sums stays frozen at its pre-loop value, so the loop always
        # runs all 100 iterations when there are initially empty cells.
        # This is intentional to match the paper's wise_swap exactly.
    end

    return gi_aux
end

"""
    initial_condition_kmeans(X_mean::Matrix{Float64}, K::Int, J::Int, M::Int,
                            Y::Vector{Float64}, rng::AbstractRNG)

Initialize groups using k-means clustering on mean product characteristics.
Perturbs characteristics slightly to encourage different solutions across restarts.
"""
function initial_condition_kmeans(X_mean::Matrix{Float64}, K::Int, J::Int, M::Int,
                                  Y::Vector{Float64}, rng::AbstractRNG)
    Q = size(X_mean, 1)
    d = MvNormal(zeros(Q), vec(1.0 ./ sqrt.(M * ones(Q))))
    X_perturbed = X_mean .+ rand(rng, d, size(X_mean, 2))
    result = kmeans(X_perturbed, K)

    gi = zeros(Int, J, K)
    for j in 1:J
        gi[j, result.assignments[j]] = 1
    end

    # Fix empty cells
    gi = wise_swap!(gi, Y, J, M, K, rng)
    return gi
end

"""
    group_estimates(gi::Matrix{Int}, Y::Vector{Float64}, X_hat::Matrix{Float64},
                    J::Int, M::Int, K::Int)

EM E-step: given group assignment, estimate group-specific parameters via sparse regression.

The regression is: Y = [group_dummies | group_regressors] * θ + ε

Returns `(parameters, ssr, Y_predicted)`.
"""
function group_estimates(gi::Matrix{Int}, Y::Vector{Float64}, X_hat::Matrix{Float64},
                         J::Int, M::Int, K::Int)
    non_missing = Y .!= 0

    # Build sparse design matrix: [group dummies (JM × KM) | group regressors (JM × QK)]
    Ztot = hcat(group_dummies_sparse(gi, J, M, K),
                group_regressors_sparse(gi, X_hat, J, M, K))

    Y_filtered = Y[non_missing]
    Ztot_filtered = Ztot[non_missing, :]

    par_new = robust_solve(Ztot_filtered, Y_filtered)

    Y_predicted_filtered = Ztot_filtered * par_new
    residuals = Y_filtered - Y_predicted_filtered
    ssr = dot(residuals, residuals)

    Y_predicted = zeros(length(Y))
    Y_predicted[non_missing] = Y_predicted_filtered

    return par_new, ssr, Y_predicted
end

"""
    group_classifier(par::Vector{Float64}, Y::Vector{Float64}, X_hat::Matrix{Float64},
                     J::Int, M::Int, K::Int)

EM M-step: given parameters, reassign each product to the group that minimizes its MSE.

Returns `(gi, ssr_by_product)`.
"""
function group_classifier(par::Vector{Float64}, Y::Vector{Float64}, X_hat::Matrix{Float64},
                           J::Int, M::Int, K::Int)
    non_missing = Y .!= 0
    Q = size(X_hat, 2)

    # Extract parameters
    delta_mat = reshape(par[1:K*M], K, M)'       # M × K
    beta_mat = reshape(par[K*M+1:K*M+Q*K], K, Q)' # Q × K

    group_SSR = zeros(J, K)

    for k in 1:K
        U = Y - X_hat * beta_mat[:, k] - kron(ones(J), delta_mat[:, k])
        non_missing_fe = kron(ones(J), delta_mat[:, k]) .!= 0
        valid = non_missing_fe .& non_missing
        N_valid = sum(valid)
        N_valid == 0 && continue
        U_sq = (valid .* U) .^ 2 ./ N_valid
        RU = reshape(U_sq, M, J)
        group_SSR[:, k] = vec(sum(RU, dims=1))
    end

    SSR_minimum = vec(minimum(group_SSR, dims=2))
    gi = zeros(Int, J, K)
    for k in 1:K
        gi[:, k] = Int.(group_SSR[:, k] .== SSR_minimum)
    end

    return gi, SSR_minimum
end

"""
    run_em(gi_init::Matrix{Int}, Y::Vector{Float64}, X_hat::Matrix{Float64},
           J::Int, M::Int, K::Int, tol::Float64, max_iter::Int)

Run EM algorithm from a given initial group assignment until convergence.

Alternates between:
1. `group_estimates` (E-step): estimate parameters given groups
2. `group_classifier` (M-step): reassign products given parameters

Returns `(parameters, groups, ssr, n_iterations)`.
"""
function run_em(gi_init::Matrix{Int}, Y::Vector{Float64}, X_hat::Matrix{Float64},
                J::Int, M::Int, K::Int, tol::Float64, max_iter::Int)
    Q = size(X_hat, 2)
    gi_cur = copy(gi_init)
    par_cur = zeros(K * M + Q * K)
    par_prev = copy(par_cur)
    ssr_cur = Inf
    delta_par = 1.0
    count = 0

    while delta_par > 0 && count <= max_iter
        par_cur, ssr_cur, _ = group_estimates(gi_cur, Y, X_hat, J, M, K)
        gi_new, _ = group_classifier(par_cur, Y, X_hat, J, M, K)
        delta_par = count > 0 ? norm(par_cur - par_prev) : 1.0
        par_prev = copy(par_cur)
        gi_cur = copy(gi_new)
        count += 1
    end

    converged = delta_par <= tol || delta_par == 0
    return par_cur, gi_cur, ssr_cur, count, converged
end

"""
    sort_groups(gi::Matrix{Int}, par::Vector{Float64}, K::Int)

Sort group assignments by increasing price coefficient (last K entries of par).
Returns reordered indicator matrix.
"""
function sort_groups(gi::Matrix{Int}, par::Vector{Float64}, K::Int)
    est_beta = par[end-K+1:end]
    perm = sortperm(est_beta)
    sorting = zeros(Int, K, K)
    for i in 1:K
        sorting[i, perm[i]] = 1
    end
    return gi * sorting'
end

"""
    classify(data::DDNLData, opts::DDNLOptions, K::Int)

Run multi-start classification for a given number of groups K.

Uses parallel EM restarts (via `Threads.@threads`) from random and/or k-means
initial conditions. Selects the solution with lowest SSR.

Returns a `ClassificationResult`.
"""
function classify(data::DDNLData, opts::DDNLOptions, K::Int)
    J, M = data.J, data.M
    Y = data.Y
    Q_x = size(data.X, 2)

    # --------------------------------------------------------------------------
    # First-stage price prediction and X_hat construction
    #
    # The paper's 3_group_classifier_Nielsen_extended.jl has a subtle variable
    # scope: the GLOBAL X, Z (set at line 161-162) include market FEs from
    # the K=1 build. Even though data_nielsen is rebuilt at line 233 with
    # no_fe (setting options.P = 3), the K loop at line 302-303 uses the
    # GLOBAL X, Z (with market FEs) for price_first_stage and X_hat:
    #
    #   p_hat = price_first_stage(X, Z, options)   # X, Z include MFE
    #   X_hat = [common_columns(X, Z) p_hat]       # 51 cols: MFE + exo + p_hat
    #   data_nielsen.X_hat = X_hat
    #
    # So X_hat has M + Q_x + 1 columns (48 + 2 + 1 = 51 for beer data).
    # The MFE columns are collinear with group_dummies (K*M), but the
    # paper's robust_lm handles rank deficiency via pivoted QR.
    #
    # The Nsim formula uses options.P = 3 (from the no_fe rebuild), NOT 51.
    # --------------------------------------------------------------------------
    # Market FE: JM × M — matching paper's kron(ones(J), I(M))
    mfe = repeat(Matrix{Float64}(I, M, M), J, 1)  # stack J copies of I(M)

    # X and Z matching paper's global vars (lines 161-162, built with market_fe)
    X_full = hcat(mfe, data.X, data.prices)         # [MFE, exo_chars, price]
    Z_full = hcat(mfe, data.X, data.Z)              # [MFE, exo_chars, price_IVs]
    p_hat = price_first_stage(X_full, Z_full, J, M)

    # X_hat = [common_columns(X, Z), p_hat] = [MFE, exo_chars, p_hat]
    # (common columns = everything in X except price = MFE + exo_chars)
    X_hat = hcat(mfe, data.X, p_hat)

    # Adaptive restarts: paper's options.P = 3 (set by no_fe rebuild at line 233)
    # NOT size(X_hat, 2) = 51. This is a quirk of the paper's variable scoping.
    P_paper = Q_x + 1
    n_adaptive = max(K * (P_paper + M) * 50, opts.n_starts)
    n_random = n_adaptive
    n_km = opts.n_kmeans
    Nsim = n_random + n_km

    # Pre-generate all initial conditions
    all_gi_init = Vector{Matrix{Int}}(undef, Nsim)
    for i in 1:n_random
        all_gi_init[i] = initial_condition_random(J, K, Y, M, Xoshiro(1234 + i))
    end

    if n_km > 0
        # Compute mean product characteristics for k-means initialization
        X_mean = zeros(Q_x, J)
        for j in 1:J
            rows = ((j-1)*M+1):(j*M)
            valid = Y[rows] .!= 0
            if any(valid)
                X_mean[:, j] = vec(mean(data.X[rows[valid], :], dims=1))
            end
        end
        for i in 1:n_km
            all_gi_init[n_random + i] = initial_condition_kmeans(X_mean, K, J, M, Y, Xoshiro(5678 + i))
        end
    end

    # Run EM from all initializations in parallel
    par_list = Vector{Vector{Float64}}(undef, Nsim)
    gi_list = Vector{Matrix{Int}}(undef, Nsim)
    residuals = fill(Inf, Nsim)
    converged_flags = fill(false, Nsim)

    Threads.@threads for i in 1:Nsim
        par_i, gi_i, ssr_i, count_i, conv_i = run_em(
            all_gi_init[i], Y, X_hat, J, M, K, opts.tol, opts.max_iter
        )
        par_list[i] = par_i
        gi_list[i] = gi_i
        residuals[i] = ssr_i
        converged_flags[i] = conv_i
    end

    # Select best solution (lowest SSR) — matching paper's classify_parallel
    best_idx = argmin(residuals)
    best_par = par_list[best_idx]
    best_ssr = residuals[best_idx]  # Raw EM SSR (paper stores this in SSR_mat)

    # Re-run group_classifier on best parameters (matching paper line 1330)
    best_gi, _ = group_classifier(best_par, Y, X_hat, J, M, K)

    # Sort groups by price coefficient (matching paper line 337)
    best_gi_sorted = sort_groups(best_gi, best_par, K)
    # Re-estimate with sorted groups for consistent parameters (matching paper line 338)
    best_par_sorted, _, _ = group_estimates(best_gi_sorted, Y, X_hat, J, M, K)

    labels = indicator_to_labels(best_gi_sorted)

    if opts.verbose
        println("  K=$K: best SSR = $(round(best_ssr, digits=4)) " *
                "($(sum(converged_flags))/$(Nsim) converged)")
    end

    # Store raw EM SSR (matching paper: SSR_mat[k] = minimum(first_step.residual))
    return ClassificationResult(
        K, best_gi_sorted, labels, best_par_sorted,
        best_ssr, converged_flags[best_idx], 0
    )
end

"""
    classify_all(data::DDNLData, opts::DDNLOptions)

Run classification for all K in `opts.K_range`. Returns a vector of `ClassificationResult`.
"""
function classify_all(data::DDNLData, opts::DDNLOptions)
    Q_x = size(data.X, 2)
    P_paper = Q_x + 1  # Paper's P = exo_chars + price (no MFE in EM loop)
    results = ClassificationResult[]
    for K in opts.K_range
        n_adaptive = max(K * (P_paper + data.M) * 50, opts.n_starts)
        opts.verbose && println("Classifying K=$K (Nsim=$n_adaptive)...")
        push!(results, classify(data, opts, K))
    end
    return results
end
