using Base.Iterators: take, zip
using Test
using Arbitrary
using VectorSpaces

I2 = Vec{2, Int}
@test VectorSpaces.isvectorspace(I2)
F2 = Vec{2, Float64}
@test VectorSpaces.isvectorspace(F2)
F3 = Vec{3, Float64}
@test VectorSpaces.isvectorspace(F3)

F5 = VProd{F2, F3, Float64}
@test VectorSpaces.isvectorspace(F5)

F0 = VEmpty{Float64}
@test VectorSpaces.isvectorspace(F0)
F0′ = VProd{F0, F0, Float64}
@test VectorSpaces.isvectorspace(F0′)
F3′ = VProd{F3, F0, Float64}
@test VectorSpaces.isvectorspace(F3′)
F10 = VProd{F5, F2, Float64}
@test VectorSpaces.isvectorspace(F10)

F6 = VExp{Vec{2, Vec{3, Float64}}, Float64}
@test VectorSpaces.isvectorspace(F6)

F1 = VUnit{Float64}
@test VectorSpaces.isvectorspace(F1)
F1′ = VExp{VUnit{VUnit{Float64}}, Float64}
@test VectorSpaces.isvectorspace(F1′)
F3′′ = VExp{Vec{3, VUnit{Float64}}, Float64}
@test VectorSpaces.isvectorspace(F3′′)
F6′ = VExp{retype(F6, F1), Float64}
@test VectorSpaces.isvectorspace(F6′)



const BigRational = Rational{BigInt}

R = BigRational
CR = VComplex{R}
QR = VQuaternion{R}
@test VectorSpaces.isvectorspace(CR)
#@test VectorSpaces.isvectorspace(QR)

@testset "Scalar" begin
    z = zero(R)
    e = one(R)

    as = arbitrary(R)
    bs = arbitrary(R)
    cs = arbitrary(R)

    for (a, b, c) in take(zip(as, bs, cs), 100)

        # Commutativity
        @test isequal(a + b, b + a)
        # Associativity
        @test isequal((a + b) + c, a + (b + c))
        # Identity
        @test isequal(z + a, a)
        @test isequal(a + z, a)
        # Inverse
        @test isequal((-a) + a, z)
        @test isequal(a + (-a), z)
        @test isequal(-(-a), a)
        # Subtraction
        @test isequal(a - b, a + (-b))

        # Commutativity
        @test isequal(a * b, b * a)
        # Associativity
        @test isequal((a * b) * c, a * (b * c))
        # Identity
        @test isequal(e * a, a)
        @test isequal(a * e, a)
        # Inverse
        if !isequal(a, z)
            @test isequal(inv(a) * a, e)
            @test isequal(a * inv(a), e)
            @test isequal(inv(inv(a)), a)
        end
        # Division
        if !isequal(b, z)
            @test isequal(a / b, a * inv(b))
        end

        # Distributivity
        @test isequal(a * (b + c), a * b + a * c)
        @test isequal((a + b) * c, a * c + b * c)

        # Conjugate
        @test isequal(conj(conj(a)), a)
        isreal(a) = isequal(a, conj(a))
        @test isreal(conj(a) * a)
    end
end

@testset "Complex" begin
    z = zero(CR)
    e = one(CR)

    as = arbitrary(CR)
    bs = arbitrary(CR)
    cs = arbitrary(CR)

    for (a, b, c) in take(zip(as, bs, cs), 100)

        # Commutativity
        @test isequal(a + b, b + a)
        # Associativity
        @test isequal((a + b) + c, a + (b + c))
        # Identity
        @test isequal(z + a, a)
        @test isequal(a + z, a)
        # Inverse
        @test isequal((-a) + a, z)
        @test isequal(a + (-a), z)
        @test isequal(-(-a), a)
        # Subtraction
        @test isequal(a - b, a + (-b))

        # Commutativity
        @test isequal(a * b, b * a)
        # Associativity
        @test isequal((a * b) * c, a * (b * c))
        # Identity
        @test isequal(e * a, a)
        @test isequal(a * e, a)
        # Inverse
        if !isequal(a, z)
            @test isequal(inv(a) * a, e)
            @test isequal(a * inv(a), e)
            @test isequal(inv(inv(a)), a)
        end
        # Division
        if !isequal(b, z)
            @test isequal(a / b, a * inv(b))
        end

        # Distributivity
        @test isequal(a * (b + c), a * b + a * c)
        @test isequal((a + b) * c, a * c + b * c)

        # Conjugate
        @test isequal(conj(conj(a)), a)
        isreal(a) = isequal(a, conj(a))
        @test isreal(conj(a) * a)
    end
end

# @testset "Quaternion" begin
#     z = zero(CR)
#     e = one(CR)
# 
#     as = arbitrary(CR)
#     bs = arbitrary(CR)
#     cs = arbitrary(CR)
# 
#     for (a, b, c) in take(zip(as, bs, cs), 100)
# 
#         # Commutativity
#         @test isequal(a + b, b + a)
#         # Associativity
#         @test isequal((a + b) + c, a + (b + c))
#         # Identity
#         @test isequal(z + a, a)
#         @test isequal(a + z, a)
#         # Inverse
#         @test isequal((-a) + a, z)
#         @test isequal(a + (-a), z)
#         @test isequal(-(-a), a)
#         # Subtraction
#         @test isequal(a - b, a + (-b))
# 
#         # Associativity
#         @test isequal((a * b) * c, a * (b * c))
#         # Identity
#         @test isequal(e * a, a)
#         @test isequal(a * e, a)
#         # Inverse
#         if !isequal(a, z)
#             @test isequal(inv(a) * a, e)
#             @test isequal(a * inv(a), e)
#             @test isequal(inv(inv(a)), a)
#         end
#         # Division
#         if !isequal(b, z)
#             @test isequal(a / b, a * inv(b))
#         end
# 
#         # Distributivity
#         @test isequal(a * (b + c), a * b + a * c)
#         @test isequal((a + b) * c, a * c + b * c)
# 
#         # Conjugate
#         @test isequal(conj(conj(a)), a)
#         isreal(a) = isequal(a, conj(a))
#         @test isreal(conj(a) * a)
#     end
# end
