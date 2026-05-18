# ============================================================================
# elbow.jl — Distance-elbow method for optimal K selection
# ============================================================================

using Statistics

"""
    find_elbow_chord(k_vals::Vector{Int}, ssr_vals::Vector{Float64})

Select optimal K using the maximum perpendicular distance to chord method.

1. Normalizes both axes (K and SSR) to [0, 1].
2. Draws a chord from the first point (k_min, SSR_max) to the last point (k_max, SSR_min).
3. Computes the perpendicular distance from each intermediate point to this chord.
4. Returns the K with the maximum distance (the "elbow").

# Returns
`(k_star, distances)` — the optimal K and the vector of perpendicular distances.
"""
function find_elbow_chord(k_vals::Vector{Int}, ssr_vals::Vector{Float64})
    n = length(k_vals)
    n < 3 && return (k_vals[1], zeros(n))

    # Normalize to [0, 1]
    k_min, k_max = extrema(k_vals)
    ssr_min, ssr_max = extrema(ssr_vals)

    k_range = k_max - k_min
    ssr_range = ssr_max - ssr_min

    k_range == 0 && return (k_vals[1], zeros(n))
    ssr_range == 0 && return (k_vals[1], zeros(n))

    k_norm = (k_vals .- k_min) ./ k_range
    ssr_norm = (ssr_vals .- ssr_min) ./ ssr_range

    # Chord line from first to last point
    p1 = [k_norm[1], ssr_norm[1]]
    p2 = [k_norm[end], ssr_norm[end]]
    line_vec = p2 - p1
    line_len = norm(line_vec)

    # Perpendicular distance from each point to the chord
    distances = zeros(n)
    for i in 1:n
        point = [k_norm[i], ssr_norm[i]]
        # Distance = |cross product| / |line_vec|
        v = point - p1
        distances[i] = abs(v[1] * line_vec[2] - v[2] * line_vec[1]) / line_len
    end

    best_idx = argmax(distances)
    return (k_vals[best_idx], distances)
end

"""
    find_elbow_secondderiv(k_vals::Vector{Int}, ssr_vals::Vector{Float64})

Select optimal K using the maximum second-derivative (deceleration) method.

1. Computes the percentage SSR drop at each transition: `pct[i] = 100*(SSR[i-1] - SSR[i]) / SSR[i-1]`.
2. Computes the second derivative (deceleration): `Δ²[i] = pct[i] - pct[i+1]`.
3. Returns the K where Δ² is maximum.

# Returns
`(k_star, pct_drops, second_derivs)` — optimal K, percentage drops, and second derivatives.
"""
function find_elbow_secondderiv(k_vals::Vector{Int}, ssr_vals::Vector{Float64})
    n = length(k_vals)
    n < 3 && return (k_vals[1], Float64[], Float64[])

    # Percentage SSR drops
    pct_drops = zeros(n - 1)
    for i in 2:n
        if ssr_vals[i-1] > 0
            pct_drops[i-1] = 100.0 * (ssr_vals[i-1] - ssr_vals[i]) / ssr_vals[i-1]
        end
    end

    # Second derivatives (deceleration)
    second_derivs = zeros(n - 2)
    for i in 1:(n-2)
        second_derivs[i] = pct_drops[i] - pct_drops[i+1]
    end

    best_idx = argmax(second_derivs)
    # The second derivative at index i corresponds to k_vals[i+1]
    return (k_vals[best_idx + 1], pct_drops, second_derivs)
end

"""
    select_k(classifications::Vector{ClassificationResult}, data::DDNLData, opts::DDNLOptions)

Select the optimal number of groups K using both elbow methods.

Dispatches based on `opts.k_selection`:
- `:in_sample_elbow`: Uses chord and 2nd-derivative methods, always picks the maximum valid K.
- `:out_of_sample_elbow`: Uses cross-validation to select K that minimizes out-of-sample prediction error.

Validates candidates by checking that IV nesting parameters σ ∈ (0, 1) when possible.
Returns an `ElbowResult`.
"""
function select_k(classifications::Vector{ClassificationResult}, data::DDNLData, opts::DDNLOptions)
    if opts.k_selection == :out_of_sample_elbow
        return select_k_cv(classifications, data, opts)
    end

    # In-sample elbow method
    k_vals = [c.k for c in classifications]
    ssr_vals = [c.ssr for c in classifications]

    k_chord, distances = find_elbow_chord(k_vals, ssr_vals)
    k_sd, _, _ = find_elbow_secondderiv(k_vals, ssr_vals)

    candidates = unique([k_chord, k_sd])

    function _sigma_valid(k_cand)
        idx = findfirst(c -> c.k == k_cand, classifications)
        idx === nothing && return false
        cr = classifications[idx]
        try
            est = second_step_estimate(data, cr.groups, cr.k, opts)
            return all(0 .< est.sigma .< 1)
        catch e
            opts.verbose && println("    σ validation failed for K=$k_cand: $e")
            return false
        end
    end

    # Always pick the MAXIMUM valid K between the two methods
    valid_candidates = filter(_sigma_valid, candidates)
    if !isempty(valid_candidates)
        k_star = maximum(valid_candidates)
    else
        # Fallback: pick the maximum candidate without sigma validation
        k_star = maximum(candidates)
    end

    return ElbowResult(k_chord, k_sd, k_star, ssr_vals, k_vals, distances)
end

"""
    select_k_cv(classifications::Vector{ClassificationResult}, data::DDNLData, opts::DDNLOptions)

Select optimal K using out-of-sample cross-validation on markets.

For each K and each fold:
1. Hold out a subset of markets (set Y=0 for those markets).
2. Re-run `group_estimates` on the remaining markets using the current K's classification.
3. Compute prediction error on the held-out markets.

Returns the K with minimum average out-of-sample MSE, packaged as an `ElbowResult`.
"""
function select_k_cv(classifications::Vector{ClassificationResult}, data::DDNLData, opts::DDNLOptions)
    J, M = data.J, data.M
    Y = data.Y
    k_vals = [c.k for c in classifications]
    ssr_vals = [c.ssr for c in classifications]

    # Also compute in-sample elbow for reporting
    k_chord, distances = find_elbow_chord(k_vals, ssr_vals)
    k_sd, _, _ = find_elbow_secondderiv(k_vals, ssr_vals)

    # Build X_hat for group_estimates (same as in classify)
    mfe = repeat(Matrix{Float64}(I, M, M), J, 1)
    X_full = hcat(mfe, data.X, data.prices)
    Z_full = hcat(mfe, data.X, data.Z)
    p_hat = price_first_stage(X_full, Z_full, J, M)
    X_hat = hcat(mfe, data.X, p_hat)

    # Create market folds
    n_folds = min(opts.cv_folds, M)
    market_indices = collect(1:M)
    # Deterministic shuffle for reproducibility
    rng = Xoshiro(42)
    shuffle!(rng, market_indices)
    fold_size = cld(M, n_folds)

    cv_mse = zeros(length(classifications))

    for (ci, cr) in enumerate(classifications)
        K = cr.k
        gi = cr.groups
        fold_errors = zeros(n_folds)

        for fold in 1:n_folds
            # Determine held-out markets for this fold
            fold_start = (fold - 1) * fold_size + 1
            fold_end = min(fold * fold_size, M)
            held_out_markets = market_indices[fold_start:fold_end]

            # Create modified Y: set held-out market observations to 0 (treated as missing)
            Y_train = copy(Y)
            for m in held_out_markets
                for j in 1:J
                    row = (j - 1) * M + m
                    Y_train[row] = 0.0
                end
            end

            # Run group_estimates on training data
            try
                _, _, Y_predicted = group_estimates(gi, Y_train, X_hat, J, M, K)

                # Compute prediction error on held-out markets
                n_test = 0
                sse_test = 0.0
                for m in held_out_markets
                    for j in 1:J
                        row = (j - 1) * M + m
                        if Y[row] != 0.0  # Only evaluate on originally non-missing obs
                            sse_test += (Y[row] - Y_predicted[row])^2
                            n_test += 1
                        end
                    end
                end
                fold_errors[fold] = n_test > 0 ? sse_test / n_test : Inf
            catch
                fold_errors[fold] = Inf
            end
        end

        cv_mse[ci] = mean(fold_errors)
    end

    # Select K with minimum CV MSE
    best_cv_idx = argmin(cv_mse)
    k_cv = k_vals[best_cv_idx]

    # Validate sigma for CV-selected K
    function _sigma_valid_cv(k_cand)
        idx = findfirst(c -> c.k == k_cand, classifications)
        idx === nothing && return false
        cr = classifications[idx]
        try
            est = second_step_estimate(data, cr.groups, cr.k, opts)
            return all(0 .< est.sigma .< 1)
        catch e
            opts.verbose && println("    σ validation failed for K=$k_cand: $e")
            return false
        end
    end

    if _sigma_valid_cv(k_cv)
        k_star = k_cv
    else
        # Fallback: try in-sample elbow candidates
        candidates = unique([k_chord, k_sd, k_cv])
        valid_candidates = filter(_sigma_valid_cv, candidates)
        k_star = !isempty(valid_candidates) ? maximum(valid_candidates) : maximum(candidates)
    end

    opts.verbose && println("  CV MSE by K: $(round.(cv_mse, digits=4))")
    opts.verbose && println("  CV-selected K: $(k_cv)")

    return ElbowResult(k_chord, k_sd, k_star, ssr_vals, k_vals, distances)
end
