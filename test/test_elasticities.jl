@testset "Elasticities" begin

    @testset "compute_elasticities" begin
        J, M, K = 4, 3, 2
        gi = [1 0; 1 0; 0 1; 0 1]
        shares = rand(J, M) .* 0.1  # small shares
        # Normalize so shares sum < 1 per market
        for m in 1:M
            shares[:, m] ./= (sum(shares[:, m]) + 0.5)
        end

        beta_price = -2.0
        sigma = [0.5, 0.7]

        elas = compute_elasticities(shares, gi, beta_price, sigma, J, M)
        @test size(elas) == (J, J)

        # Own-price elasticities should be negative (beta_price < 0, formula yields negative)
        for j in 1:J
            @test elas[j, j] < 0
        end

        # Cross-price elasticities should be positive (substitutes)
        for i in 1:J, j in 1:J
            i == j && continue
            @test elas[i, j] > 0
        end

        # Within-group cross elasticities should be larger than between-group
        # Products 1,2 in same group; products 3,4 in same group
        within_12 = elas[1, 2]
        between_13 = elas[1, 3]
        @test within_12 > between_13
    end

    @testset "elasticity_stats" begin
        J, K = 4, 2
        gi = [1 0; 1 0; 0 1; 0 1]
        elas_mat = [-3.0 0.5 0.1 0.1;
                     0.5 -2.5 0.1 0.1;
                     0.1  0.1 -4.0 0.8;
                     0.1  0.1  0.8 -3.5]
        weights = [0.3, 0.2, 0.3, 0.2]

        stats = elasticity_stats(elas_mat, gi, weights)
        @test nrow(stats) == 2
        @test :own_elasticity in propertynames(stats)
        @test :within_cross in propertynames(stats)
        @test :across_cross in propertynames(stats)

        # Own elasticities should be negative
        @test all(stats.own_elasticity .< 0)
        # Within-cross should be larger than across-cross
        @test all(stats.within_cross .> stats.across_cross)
    end

    @testset "aggregate_elasticities" begin
        J = 3
        elas_mat = [-2.0 0.3 0.1; 0.3 -3.0 0.2; 0.1 0.2 -2.5]
        gi = [1 0; 1 0; 0 1]
        weights = [0.4, 0.3, 0.3]

        agg = aggregate_elasticities(elas_mat, gi, weights)
        @test agg.own_mean < 0
        @test agg.own_sd > 0
        @test agg.cross_mean > 0
    end

    @testset "elasticity_dataframe" begin
        elas = randn(3, 3)
        df = elasticity_dataframe(elas)
        @test nrow(df) == 3
        @test ncol(df) == 4  # product + 3 elasticity columns
    end
end
