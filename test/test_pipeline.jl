@testset "Pipeline (ddnl_from_df)" begin

    @testset "Synthetic data end-to-end" begin
        # Generate a small synthetic dataset with known group structure
        Random.seed!(2024)
        J = 8    # products
        M = 10   # markets
        K = 2    # true groups

        # True group assignments: first 4 products in group 1, last 4 in group 2
        true_groups = vcat(ones(Int, 4), 2 * ones(Int, 4))

        # True parameters
        sigma_true = [0.5, 0.7]
        beta_x = 1.0
        beta_p = -2.0

        # Build panel DataFrame
        rows = []
        for j in 1:J
            for m in 1:M
                g = true_groups[j]
                x = 0.5 * g + 0.3 * randn()  # characteristic differs by group
                z = randn()  # cost shifter (instrument)
                price = 1.0 + 0.5 * x + 0.3 * z + 0.1 * randn()
                log_price = log(price)

                # Generate shares from nested logit structure
                share = 0.05 + 0.02 * randn()
                share = clamp(share, 0.001, 0.15)

                # Compute within-nest share (simplified)
                log_share_ratio = beta_x * x + beta_p * log_price + randn() * 0.5

                push!(rows, (
                    product_id = j,
                    market_id = m,
                    log_price = log_price,
                    market_share = share,
                    log_size = x,
                    y = log_share_ratio
                ))
            end
        end
        df = DataFrame(rows)

        # Build instruments first
        Z_mat = gandhi_houde_iv(df, [:log_size], :market_id, :product_id)
        df.gh_iv1 = Z_mat[:, 1]

        # Run pipeline with minimal settings for speed
        result = ddnl_from_df(df;
            y_col = :y,
            price_col = :log_price,
            product_col = :product_id,
            market_col = :market_id,
            share_col = :market_share,
            x_cols = [:log_size],
            Z_cols = [:gh_iv1],
            K_range = 2:4,
            n_starts = 20,  # small for test speed
            max_iter = 100,
            verbose = false
        )

        # Basic structure checks
        @test result isa DDNLResult
        @test result.optimal_k in 2:4
        @test length(result.group_labels) == J
        @test all(1 .<= result.group_labels .<= result.optimal_k)
        @test size(result.elasticity_matrix) == (J, J)
        @test nrow(result.elasticity_stats) == result.optimal_k

        # Estimation results
        @test result.estimation isa EstimationResult
        @test length(result.estimation.sigma) == result.optimal_k
        @test length(result.estimation.beta_IV) > 0

        # Elbow results
        @test result.elbow isa ElbowResult
        @test result.elbow.k_star in 2:4
        @test length(result.elbow.ssr_by_k) == 3  # K=2,3,4

        # Classifications
        @test length(result.classifications) == 3
        for cr in result.classifications
            @test cr isa ClassificationResult
            @test cr.ssr > 0
        end
    end
end
