@testset "Instruments" begin

    @testset "Gandhi-Houde IV" begin
        # 4 products in 2 markets
        df = DataFrame(
            product_id = [1, 2, 3, 4, 1, 2, 3, 4],
            market_id   = [1, 1, 1, 1, 2, 2, 2, 2],
            log_size    = [1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1],
            log_qty     = [0.5, 1.0, 1.5, 2.0, 0.6, 1.1, 1.6, 2.1]
        )

        Z = gandhi_houde_iv(df, [:log_size, :log_qty], :market_id, :product_id)
        @test size(Z) == (8, 2)

        # GH IV should be non-negative (mean squared deviation)
        @test all(Z .>= 0)

        # Products with extreme characteristics should have larger IV values
        # Product 4 (log_size=4) is furthest from market mean → largest GH IV
        @test Z[4, 1] > Z[2, 1]
    end

    @testset "Hausman IV" begin
        df = DataFrame(
            product_id = [1, 2, 1, 2, 1, 2],
            market_id   = [1, 1, 2, 2, 3, 3],
            log_price   = [1.0, 2.0, 1.5, 2.5, 1.2, 2.2]
        )

        Z = hausman_iv(df, :log_price, :product_id, :market_id)
        @test length(Z) == 6

        # Product 1, market 1: leave-one-out mean of prices in markets 2,3
        # = (1.5 + 1.2) / 2 = 1.35
        @test Z[1] ≈ (1.5 + 1.2) / 2

        # Product 2, market 2: leave-one-out mean of prices in markets 1,3
        # = (2.0 + 2.2) / 2 = 2.1
        @test Z[4] ≈ (2.0 + 2.2) / 2
    end

    @testset "instrument helpers" begin
        df = DataFrame(
            product_id = [1, 2, 1, 2],
            market_id   = [1, 1, 2, 2],
            log_price   = [1.0, 2.0, 1.5, 2.5],
            log_size    = [0.5, 1.0, 0.6, 1.1]
        )

        Z_gh = gandhi_houde_iv(df, [:log_size], :market_id, :product_id)
        @test size(Z_gh, 2) == 1

        Z_h = hausman_iv(df, :log_price, :product_id, :market_id)
        @test length(Z_h) == 4
    end
end
