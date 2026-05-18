@testset "Classification" begin

    @testset "initial_condition_random" begin
        J, K, M = 10, 3, 5
        Y = randn(J * M)  # no missing
        rng = Random.Xoshiro(42)

        gi = ddnests.initial_condition_random(J, K, Y, M, rng)
        @test size(gi) == (J, K)
        # Each product assigned to exactly one group
        @test all(sum(gi, dims=2) .== 1)
        # No empty groups
        @test all(sum(gi, dims=1) .>= 1)
    end

    @testset "group_estimates and group_classifier roundtrip" begin
        J, M, K = 6, 4, 2
        N = J * M

        # Create simple data where group structure is obvious
        gi_true = [1 0; 1 0; 1 0; 0 1; 0 1; 0 1]  # 3 products per group
        X_hat = randn(N, 2)
        Y = zeros(N)

        # Generate Y from known group structure
        beta1 = [1.0, -0.5]
        beta2 = [0.5, -1.0]
        delta1 = randn(M)
        delta2 = randn(M)
        for j in 1:J
            k = findfirst(==(1), gi_true[j, :])
            beta = k == 1 ? beta1 : beta2
            delta = k == 1 ? delta1 : delta2
            for m in 1:M
                row = (j - 1) * M + m
                Y[row] = dot(X_hat[row, :], beta) + delta[m] + 0.01 * randn()
            end
        end

        par, ssr, Y_pred = ddnests.group_estimates(gi_true, Y, X_hat, J, M, K)
        @test ssr < 1.0  # should fit well with true groups
        @test length(par) == K * M + size(X_hat, 2) * K

        gi_est, _ = ddnests.group_classifier(par, Y, X_hat, J, M, K)
        @test size(gi_est) == (J, K)
        @test all(sum(gi_est, dims=2) .== 1)
    end

    @testset "sort_groups" begin
        gi = [1 0; 0 1; 1 0]
        # par with last K=2 entries being price coefficients
        par = [zeros(10); -2.0; -1.0]  # group 1 has lower price coef
        sorted = ddnests.sort_groups(gi, par, 2)
        # After sorting by ascending price coef, group 1 stays first
        @test sorted == gi

        # Reverse: group 2 has lower price coef
        par2 = [zeros(10); -1.0; -2.0]
        sorted2 = ddnests.sort_groups(gi, par2, 2)
        # Groups should be swapped
        @test sorted2 == [0 1; 1 0; 0 1]
    end
end
