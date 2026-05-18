@testset "Elbow Selection" begin

    @testset "find_elbow_chord" begin
        # Classic elbow shape: steep drop then flat
        k_vals = collect(2:8)
        ssr_vals = [100.0, 60.0, 35.0, 30.0, 28.0, 27.0, 26.5]

        k_star, distances = find_elbow_chord(k_vals, ssr_vals)
        @test k_star in k_vals
        @test length(distances) == length(k_vals)
        # Elbow should be around k=4 (where the curve flattens)
        @test k_star in [3, 4]
    end

    @testset "find_elbow_secondderiv" begin
        k_vals = collect(2:8)
        ssr_vals = [100.0, 60.0, 35.0, 30.0, 28.0, 27.0, 26.5]

        k_sd, pct_drops, second_derivs = find_elbow_secondderiv(k_vals, ssr_vals)
        @test k_sd in k_vals
        @test length(pct_drops) == length(k_vals) - 1
        @test length(second_derivs) == length(k_vals) - 2
        # Maximum deceleration should be early (around k=3 or 4)
        @test k_sd in [3, 4, 5]
    end

    @testset "edge cases" begin
        # Only 2 points
        k_star, dist = find_elbow_chord([2, 3], [10.0, 5.0])
        @test k_star in [2, 3]

        # Flat SSR
        k_star2, dist2 = find_elbow_chord(collect(2:5), [10.0, 10.0, 10.0, 10.0])
        @test k_star2 in 2:5
    end
end
