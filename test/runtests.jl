using Test
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
