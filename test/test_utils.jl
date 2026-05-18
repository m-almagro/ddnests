@testset "Utils" begin

    @testset "wmean" begin
        @test wmean([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) ≈ 2.0
        @test wmean([1.0, 3.0], [1.0, 3.0]) ≈ 2.5
        @test wmean([NaN, 2.0, 3.0], [1.0, 1.0, 1.0]) ≈ 2.5
        @test isnan(wmean([NaN], [1.0]))
    end

    @testset "wstd" begin
        @test wstd([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) ≈ 0.0
        @test wstd([1.0, 3.0], [1.0, 1.0]) ≈ 1.0
    end

    @testset "wmedian" begin
        @test wmedian([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) ≈ 2.0
        @test wmedian([1.0, 10.0], [9.0, 1.0]) ≈ 1.0  # heavily weighted toward 1
    end

    @testset "make_dummies" begin
        v = [1, 2, 3, 1, 2]
        D, cats = make_dummies(v)
        @test size(D) == (5, 2)  # drops first category
        @test cats == [2, 3]
        @test D[1, :] == [0, 0]  # category 1 → baseline
        @test D[2, :] == [1, 0]  # category 2
        @test D[3, :] == [0, 1]  # category 3
    end

    @testset "labels_to_indicator / indicator_to_labels" begin
        labels = [1, 2, 3, 1, 2]
        gi = labels_to_indicator(labels, 3)
        @test size(gi) == (5, 3)
        @test gi[1, :] == [1, 0, 0]
        @test gi[2, :] == [0, 1, 0]
        @test gi[3, :] == [0, 0, 1]

        labels_back = indicator_to_labels(gi)
        @test labels_back == labels
    end

    @testset "conditional_probs" begin
        shares = [0.3 0.2; 0.2 0.3; 0.1 0.1]  # 3 products × 2 markets
        gi = [1 0; 1 0; 0 1]  # products 1,2 in group 1; product 3 in group 2
        J, M = 3, 2

        log_cp = conditional_probs(shares, gi, J, M)
        # Product 1, market 1: log(0.3 / (0.3+0.2)) = log(0.6)
        @test log_cp[1] ≈ log(0.3 / 0.5)
        # Product 3, market 1: log(0.1 / 0.1) = 0
        @test log_cp[5] ≈ 0.0 atol=1e-10
    end

    @testset "robust_solve" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = A * [1.0, 2.0] + randn(3) * 0.001
        x = ddnests.robust_solve(A, b)
        @test x ≈ [1.0, 2.0] atol=0.1
    end
end
