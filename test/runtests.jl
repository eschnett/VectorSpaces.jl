using Test
using VectorSpaces

I2 = Vec{2, Int}
@test VectorSpaces.isvectorspace(I2)
F3 = Vec{3, Float64}
@test VectorSpaces.isvectorspace(F3)

F1 = VScalar{Float64}
@test VectorSpaces.isvectorspace(F1)

F4 = VProd{F3, F1}
@test VectorSpaces.isvectorspace(F4)

F0 = VUnit{Float64}
@test VectorSpaces.isvectorspace(F0)
