# ============================================================================
# validate_estimation_only.jl — Test estimation with saved K=3 groups
# ============================================================================
#
# Skips classification. Loads the paper's K=3 classification from the saved
# CSV, then runs estimation to compare against Table A4.
#
# Uses only standard packages (no FixedEffectModels) — QR-based 2SLS.
# ============================================================================

using DataFrames, CSV, Statistics, LinearAlgebra, Printf

# --------------------------------------------------------------------------
# 2SLS with QR projection (self-contained, no package dependency)
# --------------------------------------------------------------------------
function iv_2sls_qr(Y::Vector{Float64}, X::Matrix{Float64}, Z::Matrix{Float64})
    N = length(Y)
    P = size(X, 2)

    # QR-based projection of X onto column space of Z
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

    # Robust SEs
    resid = Y - X * beta
    S = X_hat' * (resid.^2 .* X_hat)
    XhX_inv = try inv(cholesky(XhX)) catch; pinv(Matrix(XhX)) end
    V = XhX_inv * S * XhX_inv
    se = sqrt.(abs.(diag(V)))
    return beta, se
end

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

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
println("Loading data...")

const DATA_DIR = joinpath(homedir(), "Dropbox",
    "data_driven_nests_data", "Nielsen", "data_replication")

df_panel = DataFrame(CSV.File(joinpath(DATA_DIR, "panel_beer_structural_estimation.csv")))
sort!(df_panel, [:product, :market])

J = length(unique(df_panel.product))
M = length(unique(df_panel.market))
N = nrow(df_panel)
println("  J=$J, M=$M, N=$N")

product_ids = Vector{Int}(df_panel.product)
market_ids  = Vector{Int}(df_panel.market)
Y           = Vector{Float64}(df_panel.y)
X_chars     = Matrix{Float64}(hcat(df_panel.log_unit_size, df_panel.log_upc_quantity))
prices      = Vector{Float64}(df_panel.log_price)
Z_price_raw = Matrix{Float64}(hcat(df_panel.gh_iv_upc_quantity, df_panel.gh_iv_unit_size))

shares = zeros(J, M)
for row in 1:N
    shares[product_ids[row], market_ids[row]] = df_panel.market_share[row]
end

# --------------------------------------------------------------------------
# 2. Load K=3 classification from paper's saved results
# --------------------------------------------------------------------------
println("Loading K=3 classification...")

class_file = joinpath(DATA_DIR, "classification_results.csv")
isfile(class_file) || error("Cannot find: $class_file")
class_df = DataFrame(CSV.File(class_file))
hasproperty(class_df, :group_k3) || error("No group_k3 column in classification CSV")
group_labels = Vector{Int}(class_df.group_k3)
println("  Loaded from paper's classification_results.csv")

K = 3
gi = zeros(Int, J, K)
for j in 1:J
    gi[j, group_labels[j]] = 1
end

group_sizes = [sum(group_labels .== k) for k in 1:K]
println("  Group sizes: $group_sizes")

# --------------------------------------------------------------------------
# 3. Build estimation matrices (matching paper's second_step exactly)
# --------------------------------------------------------------------------
println("\nBuilding estimation matrices...")

non_missing = Y .!= 0
n_obs = sum(non_missing)
println("  Non-missing observations: $n_obs")

# X_exo = [MFE(M), log_unit_size, log_upc_quantity]  — matching paper
mfe = kron(ones(J), Matrix{Float64}(I, M, M))  # JM × M (full M columns)
X_exo = hcat(mfe, X_chars)                      # JM × 50
println("  X_exo columns: $(size(X_exo, 2))")

# Log conditional probabilities (JM × K)
log_cond = zeros(J * M, K)
for m in 1:M, k in 1:K
    prods_k = findall(==(1), gi[:, k])
    gs = sum(shares[prods_k, m])
    for j in prods_k
        row = (j - 1) * M + m
        if gs > 0 && shares[j, m] > 0
            log_cond[row, k] = log(shares[j, m] / gs)
        end
    end
end

# Group IVs from full X_exo (matching paper exactly)
println("  Building group IVs from $(size(X_exo, 2)) X_exo columns...")
P_exo = size(X_exo, 2)
IV_group = zeros(J * M, P_exo * K)
for p in 1:P_exo
    exp_x = exp.(X_exo[:, p])
    exp_x_r = reshape(exp_x, M, J)'   # J × M
    sum_exp = gi' * exp_x_r            # K × M
    for k in 1:K
        prods_k = findall(==(1), gi[:, k])
        for j in prods_k, m in 1:M
            row = (j - 1) * M + m
            if sum_exp[k, m] > 0 && exp_x_r[j, m] > 0
                IV_group[row, (p-1)*K + k] = log(exp_x_r[j, m] / sum_exp[k, m])
            end
        end
    end
end
println("  Group IV columns: $(size(IV_group, 2))")

# --------------------------------------------------------------------------
# 4. Assemble X and Z for 2SLS (matching paper's reg() call)
# --------------------------------------------------------------------------
println("\nAssembling regressor and instrument matrices...")

# X_full = [X_exo, price, log_cond_prob]
#         = [MFE(48), exo(2), price(1), σ(3)] = 54 columns
X_full = hcat(X_exo, prices, log_cond)

# Z_full = [X_exo, Z_price_instruments, IV_group]
#   (exogenous regressors serve as own instruments;
#    endogenous price + log_cond_prob instrumented by Z_price + IV_group)
Z_full = hcat(X_exo, Z_price_raw, IV_group)

# Filter to non-missing
Y_f      = Y[non_missing]
X_full_f = X_full[non_missing, :]
Z_full_f = Z_full[non_missing, :]

println("  X_full: $(size(X_full_f, 2)) columns")
println("  Z_full: $(size(Z_full_f, 2)) columns")
println("  Observations: $(length(Y_f))")

# --------------------------------------------------------------------------
# 5. Run regressions
# --------------------------------------------------------------------------
println("\nRunning OLS...")
beta_OLS, se_OLS = ols_robust(Y_f, X_full_f)
beta_OLS[end-K+1:end] = 1.0 .- beta_OLS[end-K+1:end]
println("  OLS done.")

println("Running Full IV (2SLS with QR)...")
beta_IV, se_IV = iv_2sls_qr(Y_f, X_full_f, Z_full_f)
beta_IV[end-K+1:end] = 1.0 .- beta_IV[end-K+1:end]
println("  IV done.")

# --------------------------------------------------------------------------
# 6. Display results
# --------------------------------------------------------------------------
# Coefficient order: [MFE(48), log_unit_size, log_upc_quantity, price, σ1, σ2, σ3]
idx_lus   = M + 1
idx_luq   = M + 2
idx_price = M + 3
idx_σ     = (M + 4):(M + 3 + K)

println("\n" * "="^70)
println("RESULTS — Package vs Paper (Table A4, K=3)")
println("="^70)

println("\n  OLS:")
println("  ", "-"^60)
@printf("  %-22s %14s %14s\n", "", "Package", "Paper")
println("  ", "-"^60)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log unit size",
    beta_OLS[idx_lus], se_OLS[idx_lus], 0.059, 0.006)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log upc quantity",
    beta_OLS[idx_luq], se_OLS[idx_luq], 0.067, 0.004)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log price",
    beta_OLS[idx_price], se_OLS[idx_price], -0.067, 0.005)
for k in 1:K
    paper_σ = [0.272, 0.047, -0.204][k]
    @printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "σ$k",
        beta_OLS[idx_σ[k]], se_OLS[idx_σ[k]], paper_σ, 0.001)
end

println("\n  IV (Full):")
println("  ", "-"^60)
@printf("  %-22s %14s %14s\n", "", "Package", "Paper")
println("  ", "-"^60)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log unit size",
    beta_IV[idx_lus], se_IV[idx_lus], 0.681, 0.040)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log upc quantity",
    beta_IV[idx_luq], se_IV[idx_luq], 0.824, 0.042)
@printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "log price",
    beta_IV[idx_price], se_IV[idx_price], -1.038, 0.056)
for k in 1:K
    paper_σ = [0.968, 0.649, 0.391][k]
    paper_se = [0.024, 0.022, 0.022][k]
    @printf("  %-22s %7.3f (%5.3f)  %7.3f (%5.3f)\n", "σ$k",
        beta_IV[idx_σ[k]], se_IV[idx_σ[k]], paper_σ, paper_se)
end
println("  ", "-"^60)

println("\n  Group sizes: $group_sizes")
println("  Non-missing obs: $n_obs  (paper: 64497)")
println("\n  σ in (0,1): ", all(0 .< beta_IV[idx_σ] .< 1))
println("  Price < 0:  ", beta_IV[idx_price] < 0)
println("="^70)
