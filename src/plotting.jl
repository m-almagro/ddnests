# ============================================================================
# plotting.jl — Elbow plots and elasticity heatmaps
# ============================================================================

using Plots, StatsPlots, Printf, LinearAlgebra

# --------------------------------------------------------------------------
# Elbow plots
# --------------------------------------------------------------------------

"""
    plot_elbow(elbow::ElbowResult; style=:combined, save_path=nothing)

Generate elbow plots for K selection.

# Styles
- `:chord` — Normalized 2D plot with perpendicular distance segments to chord.
- `:simple` — Raw SSR vs K with vertical line at optimal K.
- `:derivative` — Two-panel: SSR curve + bar chart of % drops and second derivatives.
- `:combined` — Single panel with vertical lines for both elbow candidates.

# Arguments
- `elbow::ElbowResult`: Output from `select_k()`.
- `style::Symbol`: Plot style (default `:combined`).
- `save_path::Union{String,Nothing}`: If provided, saves plot to this path (PDF or PNG).

# Returns
- A `Plots.Plot` object.
"""
function plot_elbow(elbow::ElbowResult; style::Symbol=:combined, save_path=nothing)
    k_vals = elbow.k_values
    ssr_vals = elbow.ssr_by_k

    p = if style == :chord
        _plot_elbow_chord(k_vals, ssr_vals, elbow.k_chord, elbow.distances)
    elseif style == :simple
        _plot_elbow_simple(k_vals, ssr_vals, elbow.k_star)
    elseif style == :derivative
        _plot_elbow_derivative(k_vals, ssr_vals, elbow.k_secondderiv)
    elseif style == :combined
        _plot_elbow_combined(k_vals, ssr_vals, elbow.k_chord, elbow.k_secondderiv)
    else
        error("Unknown elbow plot style: $style. Use :chord, :simple, :derivative, or :combined.")
    end

    if save_path !== nothing
        savefig(p, save_path)
    end

    return p
end

function _plot_elbow_chord(k_vals, ssr_vals, k_chord, distances)
    n = length(k_vals)
    k_min, k_max = extrema(k_vals)
    ssr_min, ssr_max = extrema(ssr_vals)
    k_range = Float64(k_max - k_min)
    ssr_range = ssr_max - ssr_min

    k_norm = (k_vals .- k_min) ./ k_range
    ssr_norm = (ssr_vals .- ssr_min) ./ ssr_range

    p1 = [k_norm[1], ssr_norm[1]]
    p2 = [k_norm[end], ssr_norm[end]]

    p = plot(k_norm, ssr_norm, marker=:circle, markersize=5, linewidth=2,
             color=:steelblue, label="SSR (normalized)",
             xlabel="K (normalized)", ylabel="SSR (normalized)",
             title="Distance-Elbow Method", legend=:topright,
             grid=true, framestyle=:box)

    # Chord line
    plot!(p, [p1[1], p2[1]], [p1[2], p2[2]], linewidth=3, color=:maroon,
          linestyle=:solid, label="Chord", alpha=0.7)

    # Perpendicular distance segments
    line_vec = p2 - p1
    line_len_sq = dot(line_vec, line_vec)
    for i in 1:n
        pt = [k_norm[i], ssr_norm[i]]
        t = dot(pt - p1, line_vec) / line_len_sq
        foot = p1 + t * line_vec
        plot!(p, [pt[1], foot[1]], [pt[2], foot[2]],
              linewidth=1, color=:grey, linestyle=:dash, alpha=0.8, label="")
    end

    # Optimal K line
    k_chord_norm = (k_chord - k_min) / k_range
    vline!(p, [k_chord_norm], linewidth=2, color=:darkred, linestyle=:dash,
           label="K* = $k_chord")

    return p
end

function _plot_elbow_simple(k_vals, ssr_vals, k_star)
    p = plot(k_vals, ssr_vals, marker=:circle, markersize=5, linewidth=2,
             color=:steelblue, label="SSR",
             xlabel="Number of Groups (K)", ylabel="Sum of Squared Residuals",
             title="Elbow Plot", legend=:topright,
             xticks=k_vals, grid=true, framestyle=:box)
    vline!(p, [k_star], linewidth=2, color=:darkred, linestyle=:dash,
           label="K* = $k_star")
    return p
end

function _plot_elbow_derivative(k_vals, ssr_vals, k_sd)
    n = length(k_vals)

    # Percentage drops
    pct_drops = zeros(n - 1)
    for i in 2:n
        ssr_vals[i-1] > 0 && (pct_drops[i-1] = 100.0 * (ssr_vals[i-1] - ssr_vals[i]) / ssr_vals[i-1])
    end

    # Second derivatives
    second_derivs = [pct_drops[i] - pct_drops[i+1] for i in 1:(n-2)]

    # Top panel: SSR vs K
    p1 = plot(k_vals, ssr_vals, marker=:circle, markersize=5, linewidth=2,
              color=:steelblue, label="SSR",
              ylabel="SSR", title="SSR and Derivative Analysis",
              xticks=k_vals, grid=true, framestyle=:box)
    vline!(p1, [k_sd], linewidth=2, color=:darkred, linestyle=:dash, label="K* = $k_sd")

    # Bottom panel: % drops (bars) + second derivative (line)
    transition_labels = ["$(k_vals[i])→$(k_vals[i+1])" for i in 1:(n-1)]
    p2 = bar(1:(n-1), pct_drops, label="% SSR drop", color=:lightblue, alpha=0.7,
             xlabel="Transition", ylabel="% drop / deceleration",
             xticks=(1:(n-1), transition_labels), framestyle=:box)
    if n > 2
        plot!(p2, 1:(n-2), second_derivs, marker=:diamond, markersize=5,
              linewidth=2, color=:orange, label="2nd derivative")
    end
    hline!(p2, [0], linewidth=1, color=:grey, linestyle=:dot, label="")
    sd_idx = findfirst(==(k_sd), k_vals)
    if sd_idx !== nothing && sd_idx > 1
        vline!(p2, [sd_idx - 1], linewidth=2, color=:darkred, linestyle=:dash, label="")
    end

    p = plot(p1, p2, layout=(2, 1), size=(700, 600))
    return p
end

function _plot_elbow_combined(k_vals, ssr_vals, k_chord, k_sd)
    p = plot(k_vals, ssr_vals, marker=:circle, markersize=5, linewidth=2,
             color=:steelblue, label="SSR",
             xlabel="Number of Groups (K)", ylabel="Sum of Squared Residuals",
             title="Elbow Plot", legend=:topright,
             xticks=k_vals, grid=true, framestyle=:box)

    if k_chord == k_sd
        vline!(p, [k_chord], linewidth=2, color=:darkred, linestyle=:dash,
               label="Chord & 2nd deriv. (K=$k_chord)")
    else
        vline!(p, [k_chord], linewidth=2, color=:darkred, linestyle=:dash,
               label="Chord (K=$k_chord)")
        vline!(p, [k_sd], linewidth=2, color=:blue, linestyle=:dot,
               label="2nd deriv. (K=$k_sd)")
    end

    return p
end

# --------------------------------------------------------------------------
# Elasticity heatmaps
# --------------------------------------------------------------------------

"""
    plot_elasticity_heatmap(elas_mat::Matrix{Float64}, gi::Matrix{Int};
                           log_scale_cross=true, save_path=nothing,
                           product_labels=nothing, title_str="Elasticity Matrix")

Plot a heatmap of the J × J elasticity matrix, sorted by group membership.

Products are reordered so that members of the same group are adjacent.
Diagonal (own-price) and off-diagonal (cross-price) elements use different color scales.

# Arguments
- `elas_mat`: J × J elasticity matrix.
- `gi`: Group indicator matrix (J × K).
- `log_scale_cross`: If true, off-diagonal elements are shown on log₁₀ scale (default `true`).
- `save_path`: If provided, saves to this file path.
- `product_labels`: Optional vector of product labels/names.
- `title_str`: Plot title.

# Returns
- A `Plots.Plot` object.
"""
function plot_elasticity_heatmap(elas_mat::Matrix{Float64}, gi::Matrix{Int};
                                log_scale_cross::Bool=true,
                                save_path=nothing,
                                product_labels=nothing,
                                title_str::String="Elasticity Matrix")
    J = size(elas_mat, 1)
    K = size(gi, 2)

    # Sort products by group
    group_labels = indicator_to_labels(gi)
    sort_idx = sortperm(group_labels)

    sorted_elas = elas_mat[sort_idx, sort_idx]
    sorted_groups = group_labels[sort_idx]

    # Create the matrix to plot
    plot_mat = copy(sorted_elas)

    if log_scale_cross
        for i in 1:J
            for j in 1:J
                if i != j
                    plot_mat[i, j] = log10(abs(sorted_elas[i, j]) + 1e-9)
                end
            end
        end
    end

    p = heatmap(plot_mat,
                color=:viridis,
                xlabel="Product",
                ylabel="Product",
                title=title_str,
                aspect_ratio=:equal,
                size=(700, 600),
                framestyle=:box)

    # Add group boundary lines
    boundaries = Int[]
    for k in 1:K
        n_in_group = sum(sorted_groups .== k)
        push!(boundaries, sum(sorted_groups .<= k))
    end

    for b in boundaries[1:end-1]
        hline!(p, [b + 0.5], color=:white, linewidth=2, label="")
        vline!(p, [b + 0.5], color=:white, linewidth=2, label="")
    end

    if save_path !== nothing
        savefig(p, save_path)
    end

    return p
end

"""
    plot_group_elasticities(stats_df::DataFrame; save_path=nothing)

Bar chart of own-price, within-group cross, and across-group cross elasticities by group.

# Arguments
- `stats_df`: Output from `elasticity_stats()`.
- `save_path`: If provided, saves to this file path.

# Returns
- A `Plots.Plot` object.
"""
function plot_group_elasticities(stats_df::DataFrame; save_path=nothing)
    K = nrow(stats_df)
    group_labels = ["Group $k" for k in stats_df.group]

    own = stats_df.own_elasticity
    within = stats_df.within_cross
    across = stats_df.across_cross

    p = groupedbar(
        [own within across],
        bar_position=:dodge,
        bar_width=0.25,
        label=["Own-price" "Within-group cross" "Across-group cross"],
        xticks=(1:K, group_labels),
        ylabel="Elasticity",
        title="Elasticity Statistics by Group",
        legend=:topright,
        color=[:steelblue :orange :green],
        framestyle=:box
    )

    hline!(p, [0], color=:grey, linestyle=:dot, label="")

    if save_path !== nothing
        savefig(p, save_path)
    end

    return p
end
