@testset "Estimation" begin

    @testset "ols_robust" begin
        Random.seed!(123)
        N = 100
        X = hcat(ones(N), randn(N))
        beta_true = [2.0, 3.0]
        Y = X * beta_true + 0.1 * randn(N)

        beta, se = ols_robust(Y, X)
        @test length(beta) == 2
        @test length(se) == 2
        @test beta ≈ beta_true atol=0.5
        @test all(se .> 0)
    end

    @testset "iv_2sls" begin
        Random.seed!(456)
        N = 200

        # Endogenous regressor correlated with error
        z = randn(N)          # instrument
        e = randn(N)          # error
        x = 0.5 * z + 0.5 * e  # endogenous (correlated with e)
        y = 2.0 .+ 3.0 * x + e

        X = hcat(ones(N), x)
        Z = hcat(ones(N), z)

        beta_iv, se_iv = iv_2sls(y, X, Z)
        # IV should recover true coefficients better than OLS
        @test length(beta_iv) == 2
        @test beta_iv[2] ≈ 3.0 atol=1.0  # should be closer to 3 than OLS
        @test all(se_iv .> 0)

        # OLS for comparison (should be biased)
        beta_ols, _ = ols_robust(y, X)
        # IV estimate should be closer to true value
        @test abs(beta_iv[2] - 3.0) < abs(beta_ols[2] - 3.0) + 0.5
    end
end
