# ============================================================================
# elasticities.jl — Nested logit elasticity computation and statistics
# ============================================================================

using LinearAlgebra, Statistics, DataFrames

"""
    compute_elasticities(shares::Matrix{Float64}, gi::Matrix{Int}, beta_price::Float64,
                         sigma::Vector{Float64}, J::Int, M::Int)

Compute the J × J mean elasticity matrix for a nested logit model.

Uses the nested logit elasticity formulas:

**Own-price elasticity** (product j with respect to its own price):

    ε_{jj} = (β_p / σ_g) * p_j * [1 - σ_g * s_j - (1 - σ_g) * s_{j|g}]

**Cross-price elasticity** (product j with respect to price of product i in same nest):

    ε_{ji} = β_p * p_i * [(1 - 1/σ_g) * s_{i|g} - s_i]   if i,j in same group g

**Cross-price elasticity** (product j with respect to price of product i in different nest):

    ε_{ji} = -β_p * p_i * s_i                               if i,j in different groups

The function returns the mean across non-zero markets for each product pair.

# Arguments
- `shares`: Market share matrix (J × M).
- `gi`: Group indicator matrix (J × K).
- `beta_price`: Price coefficient (β_p), typically negative.
- `sigma`: Nesting parameters vector (K × 1), each σ_k ∈ (0, 1).
- `J`: Number of products.
- `M`: Number of markets.

# Returns
- `Matrix{Float64}`: J × J mean elasticity matrix.
"""
function compute_elasticities(shares::Matrix{Float64}, gi::Matrix{Int}, beta_price::Float64,
                              sigma::Vector{Float64}, J::Int, M::Int)
    K = size(gi, 2)

    # Conditional probabilities: s_{j|g} = s_j / Σ_{i∈g} s_i
    cond_prob_mat = zeros(J, M)
    for m in 1:M
        for k in 1:K
            products_in_k = findall(==(1), gi[:, k])
            group_share = sum(shares[products_in_k, m])
            for j in products_in_k
                if group_share > 0
                    cond_prob_mat[j, m] = shares[j, m] / group_share
                end
            end
        end
    end

    # σ vector for each product
    sigma_vec = gi * sigma  # J × 1

    elasticity = zeros(J, J, M)

    for j in 1:J
        sigma_j = sigma_vec[j]
        for m in 1:M
            # Own-price elasticity
            elasticity[j, j, m] = (1.0 / sigma_j) *
                (1.0 - sigma_j * shares[j, m] - (1.0 - sigma_j) * cond_prob_mat[j, m]) * beta_price
        end

        # Cross-price elasticities
        gi_j = gi[j, :]
        for i in 1:J
            i == j && continue
            gi_i = gi[i, :]
            sigma_i = sigma_vec[i]
            same_group = dot(gi_i, gi_j) > 0

            for m in 1:M
                if same_group
                    elasticity[j, i, m] = ((1.0 - 1.0 / sigma_i) * cond_prob_mat[i, m] - shares[i, m]) * beta_price
                else
                    elasticity[j, i, m] = -shares[i, m] * beta_price
                end
            end
        end
    end

    # Mean across non-zero markets
    mean_elasticity = zeros(J, J)
    for i in 1:J
        for j in 1:J
            vals = elasticity[i, j, :]
            vals_nz = vals[vals .!= 0]
            mean_elasticity[i, j] = isempty(vals_nz) ? 0.0 : mean(vals_nz)
        end
    end

    return mean_elasticity
end

"""
    elasticity_stats(elas_mat::Matrix{Float64}, gi::Matrix{Int},
                     weights::Vector{Float64})

Compute per-group elasticity statistics: own-price, within-group cross, and across-group cross.

# Arguments
- `elas_mat`: J × J elasticity matrix.
- `gi`: Group indicator matrix (J × K).
- `weights`: Product-level weights (e.g., market share or unit sales), length J.

# Returns
- `DataFrame` with columns: `group`, `own_elasticity`, `within_cross`, `across_cross`, `n_products`.
"""
function elasticity_stats(elas_mat::Matrix{Float64}, gi::Matrix{Int},
                          weights::Vector{Float64})
    K = size(gi, 2)
    J = size(gi, 1)

    rows = []

    for k in 1:K
        idx = findall(==(1), gi[:, k])
        n_k = length(idx)
        w_k = weights[idx]
        w_k_norm = w_k ./ sum(w_k)

        # Own-price elasticities (weighted by product weight)
        own_elas = [elas_mat[j, j] for j in idx]
        own_mean = sum(own_elas .* w_k_norm)

        # Within-group cross elasticities
        within_cross_vals = Float64[]
        within_cross_weights = Float64[]
        for (ii, i) in enumerate(idx)
            for (jj, j) in enumerate(idx)
                i == j && continue
                push!(within_cross_vals, elas_mat[i, j])
                push!(within_cross_weights, w_k_norm[ii] * w_k_norm[jj])
            end
        end
        within_cross = isempty(within_cross_vals) ? 0.0 :
            sum(within_cross_vals .* within_cross_weights) / sum(within_cross_weights)

        # Across-group cross elasticities
        other_idx = findall(i -> gi[i, k] != 1, 1:J)
        across_cross_vals = Float64[]
        across_cross_weights = Float64[]
        w_other = weights[other_idx]
        w_other_norm = length(w_other) > 0 ? w_other ./ sum(w_other) : Float64[]
        for (ii, i) in enumerate(idx)
            for (jj, j) in enumerate(other_idx)
                push!(across_cross_vals, elas_mat[i, j])
                push!(across_cross_weights, w_k_norm[ii] * w_other_norm[jj])
            end
        end
        across_cross = isempty(across_cross_vals) ? 0.0 :
            sum(across_cross_vals .* across_cross_weights) / sum(across_cross_weights)

        push!(rows, (group=k, own_elasticity=own_mean, within_cross=within_cross,
                     across_cross=across_cross, n_products=n_k))
    end

    return DataFrame(rows)
end

"""
    aggregate_elasticities(elas_mat::Matrix{Float64}, gi::Matrix{Int},
                           weights::Vector{Float64})

Compute overall weighted elasticity statistics across all groups.

# Returns
- `NamedTuple` with `own_mean`, `own_sd`, `cross_mean`, `cross_sd`.
"""
function aggregate_elasticities(elas_mat::Matrix{Float64}, gi::Matrix{Int},
                                weights::Vector{Float64})
    J = size(elas_mat, 1)
    w_norm = weights ./ sum(weights)

    # Own-price elasticities
    own = [elas_mat[j, j] for j in 1:J]
    own_mean = sum(own .* w_norm)
    own_sd = sqrt(sum(w_norm .* (own .- own_mean).^2))

    # Cross-price elasticities (off-diagonal)
    cross_vals = Float64[]
    cross_w = Float64[]
    for i in 1:J
        for j in 1:J
            i == j && continue
            push!(cross_vals, elas_mat[i, j])
            push!(cross_w, w_norm[i] * w_norm[j])
        end
    end
    cw_sum = sum(cross_w)
    cross_mean = sum(cross_vals .* cross_w) / cw_sum
    cross_sd = sqrt(sum(cross_w .* (cross_vals .- cross_mean).^2) / cw_sum)

    return (own_mean=own_mean, own_sd=own_sd, cross_mean=cross_mean, cross_sd=cross_sd)
end

"""
    elasticity_dataframe(elas_mat::Matrix{Float64})

Convert a J × J elasticity matrix to a DataFrame with product indices and column names `E_j1`, `E_j2`, etc.
"""
function elasticity_dataframe(elas_mat::Matrix{Float64})
    J = size(elas_mat, 1)
    colnames = ["product"; ["E_j$k" for k in 1:J]...]
    df = DataFrame(hcat(collect(1:J), elas_mat), Symbol.(colnames))
    return df
end
