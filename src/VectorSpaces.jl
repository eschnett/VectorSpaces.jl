module VectorSpaces

using IterTools: imap
using Arbitrary
using SimpleTraits



################################################################################

# @traitdef VectorSpace{V}
export VectorSpace
abstract type VectorSpace{S} end

# @traitimpl VectorSpace{V} <- isvectorspace(V)
function isvectorspace(::Type{V}) where {V <: VectorSpace}
    try
        # Element type
        S = stype(V)
        S <: Number || return false
        V8 = retype(V, Int8)
        S8 = stype(V8)
        S8 === Int8 || return false
        VS = retype(V8, S)
        VS === V || return false
        # Constructors
        z = zero(V)::V
        e1 = ones(V)::V
        # Collection properties
        @assert eltype(V) === S
        l = length(e1)
        @assert l == length(collect(e1))
        @assert all(==(S(1)), collect(e1))
        @assert all(==(S(1)), e1)
        @assert all(==(S(3)), map(a->2a+1, e1))
        elin = linear(V)
        @assert sum(elin) == l*(l+1)/2
        # Check whether functions exist and return the right types
        (z * S(1))::V
        S <: Union{AbstractFloat, Rational} && (z / S(1))::V
        (S(1) * z)::V
        S <: Union{AbstractFloat, Rational} && (S(1) \ z)::V
        (+e1)::V
        (-e1)::V
        (z + e1)::V
        (z - e1)::V
        (e1 ⋅ e1)::S
        incomplete_norm(e1)::S
        if S <: AbstractFloat
            norm(e1)::S
        end
        # Check vector space laws
        e1 + z == e1 || return false
        z + e1 == e1 || return false
        S(2) * z == z || return false
        e2 = S(2) * e1
        e5 = S(5) * e1
        e1 + (-e1) == z || return false
        e2 + e1 == e1 + e2 || return false
        e1 + (e2 + e5) == (e1 + e2) + e5 || return false
        # Check dot product laws
        z ⋅ z == S(0) || return false
        if e5 == e2
            e5 ⋅ e2 == S(0) || return false
        else
            e5 ⋅ e2 != S(0) || return false
        end
        (S(3) * e5) ⋅ e2 == S(3) * (e5 ⋅ e2) || return false
        # Check norm laws
        incomplete_norm(z) == S(0) || return false
        if e2 == z
            incomplete_norm(e2) == S(0) || return false
        else
            incomplete_norm(e2) > S(0) || return false
        end
        if S <: AbstractFloat
            norm(z) == S(0) || return false
            if e2 == z
                norm(e2) == S(0) || return false
            else
                norm(e2) > S(0) || return false
            end
        end
        return true
    catch
        return false
    end
end

using LinearAlgebra

export stype, retype
export linear
export ⋅, incomplete_norm, partial_mean, norm, mean

function Base.zeros(::Type{V}) where {V <: VectorSpace}
    S = stype(V)
    map(_ -> zero(S), zero(V))::V
end
function Base.ones(::Type{V}) where {V <: VectorSpace}
    S = stype(V)
    map(_ -> one(S), zero(V))::V
end
function linear(::Type{V}) where {V <: VectorSpace}
    S = stype(V)
    idx = 0
    inc(_) = S(idx += 1)
    map(inc, zero(V))::V
end



################################################################################

export Vec
struct Vec{D, S} <: VectorSpace{S}
    elts::NTuple{D, S}
    # Vec(elts::NTuple{D, S}) where {D, S<:Number} = new{D, S}(elts)
end

function Base.show(io::IO, x::Vec{D, S}) where {D, S}
    print(io, "[")
    for d in 1:D
        d>1 && print(io, ", ")
        print(io, x.elts[d])
    end
    print(io, "]")
end

# function Vec{D, S}(elts...) where {D, S<:Number}
#     elts :: NTuple{D, S}
#     Vec(elts)
# end
# Vec(elts::NTuple{D, S}) where {D, S} = Vec{D, S}(elts)
Vec(r::AbstractRange) = Vec(tuple(r...))

Base.firstindex(x::Vec) = firstindex(x.elts)
Base.lastindex(x::Vec) = lastindex(x.elts)
Base.getindex(x::Vec, i) = x.elts[i]
# Base.IteratorSize(::Type{Vec{D, S}}) where {D, S} = HasLength()
# Base.IteratorEltype(::Type{Vec{D, S}}) where {D, S} = HasEltype()
Base.eltype(::Type{Vec{D, S}}) where {D, S} = S
# Base.size(x::Vec) = (length(x), )
Base.length(x::Vec) = length(x.elts)
Base.iterate(x::Vec) = iterate(x.elts)
Base.iterate(x::Vec, st) = iterate(x.elts, st)
function Base.map(f, x::Vec{D}) where {D}
    relts = map(f, x.elts)
    R = eltype(relts)
    Vec{D, R}(relts)
end
function Base.map(f, x::Vec{D}, y::Vec{D}) where {D}
    relts = map(f, x.elts, y.elts)
    Vec{D, eltype(relts)}(relts)
end



stype(::Type{Vec{D, S}}) where {D, S} = S
retype(::Type{Vec{D, S}}, ::Type{T}) where {D, S, T} = Vec{D, T}



# This needs to accept either a Number or a VectorSpace
Base.zero(::Type{Vec{D, S}}) where {D, S <: Union{Number, VectorSpace}} =
    Vec{D, S}(ntuple(_->zero(S), D))
Base. +(x::Vec{D, S}) where {D, S <: Union{Number, VectorSpace}} =
    Vec{D, S}(map(+, x.elts))
Base. -(x::Vec{D, S}) where {D, S <: Union{Number, VectorSpace}} =
    Vec{D, S}(map(-, x.elts))
Base. *(a::S, x::Vec{D, S}) where {D, S <: Union{Number, VectorSpace}} =
    Vec{D, S}(map(c->a*c, x.elts))
Base. \(a::S, x::Vec{D, S}) where
        {D, S <: Union{AbstractFloat, Rational, VectorSpace}} =
    Vec{D, S}(map(c->a\c, x.elts))
Base. *(x::Vec{D, S}, a::S) where {D, S <: Union{Number, VectorSpace}} =
    Vec{D, S}(map(c->c*a, x.elts))
Base. /(x::Vec{D, S}, a::S) where
        {D, S <: Union{AbstractFloat, Rational, VectorSpace}} =
    Vec{D, S}(map(c->c/a, x.elts))

Base. +(x::Vec{D, S}, y::Vec{D, S}) where {D, S <: Union{Number, VectorSpace}} =
    Vec{D, S}(map(+, x.elts, y.elts))
Base. -(x::Vec{D, S}, y::Vec{D, S}) where {D, S <: Union{Number, VectorSpace}} =
    Vec{D, S}(map(-, x.elts, y.elts))

# @assert isvectorspace(Vec{1, Int})
# @traitimpl VectorSpace{Vec{S}}



LinearAlgebra. ⋅(x::Vec{D, S}, y::Vec{D, S}) where
        {D, S <: Union{Number, VectorSpace}} =
    sum(map(*, x, y))::S



incomplete_norm(x::Vec{D, S}) where {D, S <: Union{Number, VectorSpace}} =
    sum(abs.(x.elts) .^ 2)::S
LinearAlgebra.norm(x::Vec{D, S}) where
        {D, S <: Union{AbstractFloat, VectorSpace}} =
    sqrt(incomplete_norm(x))

function incomplete_norm(x::Vec{D, S}, p::Real) where
        {D, S <: Union{AbstractFloat, VectorSpace}}
    if p == 0
        S(D)
    elseif p == 1
        sum(abs.(x.elts))
    elseif p == 2
        sum(abs.(x.elts) .^ 2)
    elseif p == Inf
        max(abs.(x.elts))
    elseif isinteger(p)
        sum(abs.(x.elts) .^ Int(p))
    else
        sum(abs.(x.elts) .^ p)
    end
end
function LinearAlgebra.norm(x::Vec{D, S}, p::Real) where
        {D, S <: Union{AbstractFloat, VectorSpace}}
    r = incomplete_norm(x, p)
    if p == 0
        r
    elseif p == 1
        r
    elseif p == 2
        sqrt(r)
    else
        r ^ (1/p)
    end
end

function partial_mean(x::Vec{D, S}, p::Real) where
        {D, S <: Union{AbstractFloat, VectorSpace}}
    if p == -Inf
        min(x.elts), S(D)
    elseif p == -1
        sum(inv.(x.elts)), S(D)
    elseif p == 0
        product(x.elts), S(D)
    elseif p == 1
        sum(x.elts), S(D)
    elseif p == 2
        sum(x.elts .^ 2), S(D)
    elseif p == Inf
        max(x.elts), S(D)
    elseif isinteger(p)
        sum(x.elts .^ Int(p)), S(D)
    else
        sum(x.elts .^ p), S(D)
    end
end
function mean(x::Vec{D, S}, p::Real) where
        {D, S <: Union{AbstractFloat, VectorSpace}}
    r, n = partial_mean(x, p)
    if p == -Inf
        r
    elseif p == -1
        inv(r / n)
    elseif p == 0
        r ^ inv(n)
    elseif p == 1
        r / n
    elseif p == 2
        sqrt(r / n)
    elseif p == Inf
        r
    else
        (r / n) ^ (1/p)
    end
end



################################################################################

export VProd
struct VProd{V1, V2, S} <: VectorSpace{S}
    v1::V1
    v2::V2
    function VProd{V1, V2, S}(v1::V1, v2::V2) where
            {S, V1 <: VectorSpace{S}, V2 <: VectorSpace{S}}
       new{V1, V2, S}(v1, v2)
   end
end

function VProd(v1::V1, v2::V2) where {V1 <: VectorSpace, V2 <: VectorSpace}
    S1 = stype(V1)
    S2 = stype(V2)
    @assert S1 === S2
    VProd{V1, V2, S1}(v1, v2)
end

Base.show(io::IO, x::VProd) = print(io, "($(x.v1), $(x.v2))")

Base.firstindex(x::VProd) = 1
Base.lastindex(x::VProd) = length(x)
function Base.getindex(x::VProd, i)
    l1 = length(x.v1)
    if i <= l1
        return x.v1[i - 1 + firstindex(x.v1)]
    end
    i = i - l1
    l2 = length(x.v2)
    if i <= l2
        return x.v2[i - 1 + firstindex(x.v2)]
    end
    @assert false
end
Base.eltype(::Type{VProd{V1, V2, S}}) where {V1, V2, S} = S
Base.length(x::VProd) = length(x.v1) + length(x.v2)
function Base.iterate(x::VProd)
    i1 = iterate(x.v1)
    if i1 !== nothing
        el, st1 = i1
        return el, (false, st1)
    end
    i2 = iterate(x.v2)
    if i2 !== nothing
        el, st2 = i2
        return el, (true, st2)
    end
    return nothing
end
function Base.iterate(x::VProd, st)
    b, st12 = st
    if b === false
        i1 = iterate(x.v1, st12)
        if i1 !== nothing
            el, st1 = i1
            return el, (false, st1)
        end
        i2 = iterate(x.v2)
        if i2 !== nothing
            el, st2 = i2
            return el, (true, st2)
        end
        return nothing
    end
    i2 = iterate(x.v2, st12)
    if i2 !== nothing
        el, st2 = i2
        return el, (true, st2)
    end
    return nothing
end
function Base.map(f, x::VProd{V1, V2}) where {V1, V2}
    r1 = map(f, x.v1)
    r2 = map(f, x.v2)
    R = promote_type(stype(typeof(r1)), stype(typeof(r2)))
    VProd{retype(V1, R), retype(V2, R), R}(r1, r2)
end
function Base.map(f, x::VProd{V1, V2}, y::VProd{V1, V2}) where {V1, V2}
    r1 = map(f, x.v1, y.v1)
    r2 = map(f, x.v2, y.v2)
    R = eltype(Tuple{stype(r1), stype(r2)})
    VProd{retype(V1, R), retype(V2, R), R}(r1, r2)
end



ltype(::Type{VProd{V1, V2, S}}) where {V1, V2, S} = V1
rtype(::Type{VProd{V1, V2, S}}) where {V1, V2, S} = V2
stype(::Type{VProd{V1, V2, S}}) where {V1, V2, S} = S
retype(::Type{VProd{V1, V2, S}}, ::Type{T}) where {V1, V2, S, T} =
    VProd{retype(V1, T), retype(V2, T), T}

Base.zero(::Type{VProd{V1, V2, S}}) where
        {V1, V2, S <: Union{Number, VectorSpace}} =
    VProd{V1, V2, S}(zero(V1), zero(V2))

Base. +(x::VProd{V1, V2, S}) where {V1, V2, S} =
    VProd{V1, V2, S}(+x.v1, +x.v2)
Base. -(x::VProd{V1, V2, S}) where {V1, V2, S} =
    VProd{V1, V2, S}(-x.v1, -x.v2)
Base. *(a::S, x::VProd{V1, V2, S}) where {V1, V2, S} =
    VProd{V1, V2, S}(a*x.v1, a*x.v2)
Base. \(a::S, x::VProd{V1, V2, S}) where
        {V1, V2, S <: Union{AbstractFloat, Rational}} =
    VProd{V1, V2, S}(a\x.v1, a\x.v2)
Base. *(x::VProd{V1, V2, S}, a::S) where {V1, V2, S} =
    VProd{V1, V2, S}(x.v1*a, x.v2*a)
Base. /(x::VProd{V1, V2, S}, a::S) where
        {V1, V2, S <: Union{AbstractFloat, Rational}} =
    VProd{V1, V2, S}(x.v1/a, x.v2/a)

Base. +(x::VProd{V1, V2, S}, y::VProd{V1, V2, S}) where {V1, V2, S} =
    VProd{V1, V2, S}(x.v1+y.v1, x.v2+y.v2)
Base. -(x::VProd{V1, V2, S}, y::VProd{V1, V2, S}) where {V1, V2, S} =
    VProd{V1, V2, S}(x.v1-y.v1, x.v2-y.v2)



LinearAlgebra. ⋅(x::VProd{V1, V2, S}, y::VProd{V1, V2, S}) where {V1, V2, S} =
    (x.v1 ⋅ y.v1 + x.v2 ⋅ y.v2)::S



incomplete_norm(x::VProd{V1, V2, S}) where {V1, V2, S <: Number} =
    (incomplete_norm(x.v1) + incomplete_norm(x.v2))::S
LinearAlgebra.norm(x::VProd{V1, V2, S}) where {V1, V2, S <: AbstractFloat} =
    sqrt(incomplete_norm(x))::S



################################################################################

export VEmpty
struct VEmpty{S} <: VectorSpace{S} end

Base.show(io::IO, x::VEmpty) = print(io, "()")

Base.firstindex(x::VEmpty) = 1
Base.lastindex(x::VEmpty) = 0
Base.eltype(::Type{VEmpty{S}}) where {S} = S
Base.length(x::VEmpty) = 0
Base.iterate(x::VEmpty) = nothing
Base.iterate(x::VEmpty, st) = @assert false
function Base.map(f, x::VEmpty{S}) where {S}
    r = f(S(1))
    R = typeof(r)
    VEmpty{R}()
end
function Base.map(f, x::VEmpty{S}, y::VEmpty{S}) where {S}
    r = f(S(1), S(1))
    R = typeof(r)
    VEmpty{R}()
end



stype(::Type{VEmpty{S}}) where {S} = S
retype(::Type{VEmpty{S}}, ::Type{T}) where {S, T} = VEmpty{T}

Base.zero(::Type{VEmpty{S}}) where {S <: Union{Number, VectorSpace}} =
    VEmpty{S}()

Base. +(x::VEmpty{S}) where {S} = VEmpty{S}()
Base. -(x::VEmpty{S}) where {S} = VEmpty{S}()
Base. *(a::S, x::VEmpty{S}) where {S} = VEmpty{S}()
Base. \(a::S, x::VEmpty{S}) where {S} = VEmpty{S}()
Base. *(x::VEmpty{S}, a::S) where {S} = VEmpty{S}()
Base. /(x::VEmpty{S}, a::S) where {S} = VEmpty{S}()

Base. +(x::VEmpty{S}, y::VEmpty{S}) where {S} = VEmpty{S}()
Base. -(x::VEmpty{S}, y::VEmpty{S}) where {S} = VEmpty{S}()



LinearAlgebra. ⋅(x::VEmpty{S}, y::VEmpty{S}) where {S} = S(0)



incomplete_norm(x::VEmpty{S}) where {S} = S(0)
LinearAlgebra.norm(x::VEmpty{S}) where {S} = S(0)



################################################################################

export VComp
struct VComp{V, S} <: VectorSpace{S}
    v::V
    function VComp{V, S}(v::V) where {V, S}
        @assert S === stype(stype(V))
        new{V, S}(v)
    end
end

function VComp(v::V) where {V <: VectorSpace}
    S = stype(stype(V))
    VComp{V, S}(v)
end

Base.show(io::IO, x::VComp) = print(io, "($(x.v))")

function Base.firstindex(x::VComp)
    i0 = firstindex(x.v)
    li = length(x.v)
    li == 0 && return (i0, 1)
    j0 = firstindex(x.v[i0])
    (i0, j0)
end
function Base.lastindex(x::VComp)
    i1 = lastindex(x.v)
    li = length(x.v)
    li == 0 && return (i1, 0)
    j1 = lastindex(x.v[i1])
    (i1, j1)
end
function Base.getindex(x::VComp, ij)
    i = ij[1]
    j = ij[2]
    x.v[i][j]
end
Base.eltype(::Type{VComp{V, S}}) where {V, S} = S
function Base.length(x::VComp)
    li = length(x.v)
    li == 0 && return 0
    lj = length(x.v[end])
    li * lj
end
function Base.iterate(x::VComp)
    iti = iterate(x.v)
    iti === nothing && return nothing
    eli, sti = iti
    itj = iterate(eli)
    while itj === nothing
        iti = iterate(x.v, sti)
        iti === nothing && return nothing
        eli, sti = iti
        itj = iterate(eli)
    end
    elj, stj = itj
    elj, (iti, stj)
end
function Base.iterate(x::VComp, st)
    iti, stj = st
    eli, sti = iti
    itj = iterate(eli, stj)
    while itj === nothing
        iti = iterate(x.v, sti)
        iti === nothing && return nothing
        eli, sti = iti
        itj = iterate(eli)
    end
    elj, stj = itj
    elj, (iti, stj)
end
mapmap(f, x) = map(a -> map(f, a), x)
mapmap(f, x, y) = map((a, b) -> map(f, a, b), x, y)
Base.map(f, x::VComp) = VComp(mapmap(f, x.v))
Base.map(f, x::VComp{V}, y::VComp{V}) where {V} = VComp(mapmap(f, x.v, y.v))



otype(::Type{VComp{V, S}}) where {V, S} = retype(V, S)
itype(::Type{VComp{V, S}}) where {V, S} = stype(V)
stype(::Type{VComp{V, S}}) where {V, S} = S
function retype(::Type{VComp{V, S}}, ::Type{T}) where {V, S, T}
    VT = retype(V, retype(stype(V), T))
    VComp{VT, T}
end

Base.zero(::Type{VComp{V, S}}) where {V, S <: Union{Number, VectorSpace}} =
    VComp{V, S}(zero(V))

Base. +(x::VComp{V, S}) where {V, S} = VComp{V, S}(mapmap(+, x.v))
Base. -(x::VComp{V, S}) where {V, S} = VComp{V, S}(mapmap(-, x.v))
Base. *(a::S, x::VComp{V, S}) where {V, S} = VComp{V, S}(mapmap(c->a*c, x.v))
Base. \(a::S, x::VComp{V, S}) where {V, S <: Union{AbstractFloat, Rational}} =
    VComp{V, S}(mapmap(c->a\c, x.v))
Base. *(x::VComp{V, S}, a::S) where {V, S} = VComp{V, S}(mapmap(c->c*a, x.v))
Base. /(x::VComp{V, S}, a::S) where {V, S <: Union{AbstractFloat, Rational}} =
    VComp{V, S}(mapmap(c->c/a, x.v))

Base. +(x::VComp{V, S}, y::VComp{V, S}) where {V, S} =
    VComp{V, S}(mapmap(+, x.v, y.v))
Base. -(x::VComp{V, S}, y::VComp{V, S}) where {V, S} =
    VComp{V, S}(mapmap(-, x.v, y.v))



LinearAlgebra. ⋅(x::VComp{V, S}, y::VComp{V, S}) where {V, S} =
    sum(map(⋅, x.v, y.v))::S



incomplete_norm(x::VComp{V, S}) where {V, S <: Number} =
    sum(map(incomplete_norm, x.v))::S
LinearAlgebra.norm(x::VComp{V, S}) where {V, S <: AbstractFloat} =
    sqrt(incomplete_norm(x))::S



################################################################################

export VUnit
struct VUnit{S} <: VectorSpace{S}
    elt::S
end

# VUnit(elt::S) where {S} = VUnit{S}(elt)

Base.show(io::IO, x::VUnit) = print(io, x.elt)

Base.firstindex(x::VUnit) = 1
Base.lastindex(x::VUnit) = 1
Base.getindex(x::VUnit, i) = x.elt
# Base.IteratorSize(::Type{VUnit{S}}) where {S} = HasLength()
# Base.IteratorEltype(::Type{VUnit{S}}) where {S} = HasEltype()
Base.eltype(::Type{VUnit{S}}) where {S} = S
# Base.size(x::VUnit) = (length(x), )
Base.length(x::VUnit) = 1
Base.iterate(x::VUnit) = (x.elt, ())
Base.iterate(x::VUnit, st) = nothing
function Base.map(f, x::VUnit{S}) where {S}
    r = f(x.elt)
    R = typeof(r)
    VUnit{R}(r)
end
function Base.map(f, x::VUnit{S}, y::VUnit{S}) where {S}
    r = f(x.elt, y.elt)
    R = typeof(r)
    VUnit{R}(r)
end



stype(::Type{VUnit{S}}) where {S} = S
retype(::Type{VUnit{S}}, ::Type{T}) where {S, T} = VUnit{T}

Base.zero(::Type{VUnit{S}}) where {S} = VUnit{S}(zero(S))

Base. +(x::VUnit{S}) where {S<:Number} = VUnit{S}(+x.elt)
Base. -(x::VUnit{S}) where {S<:Number} = VUnit{S}(-x.elt)
Base. *(a::S, x::VUnit{S}) where {S<:Number} = VUnit{S}(a * x.elt)
Base. \(a::S, x::VUnit{S}) where
        {S <: Union{AbstractFloat, Rational, VectorSpace}} =
    VUnit{S}(a \ x.elt)
Base. *(x::VUnit{S}, a::S) where {S<:Number} = VUnit{S}(x.elt * a)
Base. /(x::VUnit{S}, a::S) where
        {S <: Union{AbstractFloat, Rational, VectorSpace}} =
    VUnit{S}(x.elt / a)

Base. +(x::VUnit{S}, y::VUnit{S}) where {S<:Number} =
    VUnit{S}(x.elt + y.elt)
Base. -(x::VUnit{S}, y::VUnit{S}) where {S<:Number} =
    VUnit{S}(x.elt - y.elt)



LinearAlgebra. ⋅(x::VUnit{S}, y::VUnit{S}) where {S<:Number} =
    (x.elt * y.elt)::S



incomplete_norm(x::VUnit{S}) where {S<:Number} = (abs(x.elt) ^ 2)::S
LinearAlgebra.norm(x::VUnit{S}) where {S<:AbstractFloat} =
    sqrt(incomplete_norm(x))::S



################################################################################

# See the Cayley–Dickson construction
# <https://en.wikipedia.org/wiki/Cayley–Dickson_construction>
export VComplex
struct VComplex{S} <: VectorSpace{S}
    re::S
    im::S
end

function Base.show(io::IO, x::VComplex{S}) where {S}
    print(io, "($(x.re),$(x.im))")
end

Base.firstindex(x::VComplex) = firstindex((x.re, x.im))
Base.lastindex(x::VComplex) = lastindex((x.re, x.im))
Base.getindex(x::VComplex, i) = getindex((x.re, x.im), i)
Base.eltype(::Type{VComplex{S}}) where {S} = S
Base.length(x::VComplex) = length((x.re, x.im))
Base.iterate(x::VComplex) = iterate((x.re, x.im))
Base.iterate(x::VComplex, st) = iterate((x.re, x.im), st)
function Base.map(f, x::VComplex)
    res = map(f, (x.re, x.im))
    R = eltype(res)
    VComplex{R}(res[1], res[2])
end
function Base.map(f, x::VComplex, y::VComplex)
    res = map(f, (x.re, x.im), (y.re, y.im))
    R = eltype(res)
    VComplex{R}(res[1], res[2])
end

function Base.isequal(x::VComplex{S}, y::VComplex{S}) where {S}
    isequal(x.re, y.re) && isequal(x.im, y.im)
end
function Base. ==(x::VComplex{S}, y::VComplex{S}) where {S}
    x.re == y.re && x.im == y.im
end



function mkcomplex(::Type{T}, arb::Iterators.Stateful) where {T}
    re = popfirst!(arb)::T
    im = popfirst!(arb)::T
    VComplex{T}(re, im)
end
function Arbitrary.arbitrary(::Type{VComplex{T}}, ast::ArbState) where {T}
    iter = Iterators.Stateful(arbitrary(T, ast))
    Generate{VComplex{T}}(() -> mkcomplex(T, iter))
end



stype(::Type{VComplex{S}}) where {S} = S
retype(::Type{VComplex{S}}, ::Type{T}) where {S, T} = VComplex{T}

function Base.zero(::Type{VComplex{S}}) where {S}
    VComplex{S}(zero(S), zero(S))
end
function Base.one(::Type{VComplex{S}}) where {S}
    VComplex{S}(one(S), zero(S))
end

function Base. +(x::VComplex{S}) where {S}
    VComplex{S}(+x.re, +x.im)
end
function Base. -(x::VComplex{S}) where {S}
    VComplex{S}(-x.re, -x.im)
end
function Base.conj(x::VComplex{S}) where {S}
    VComplex{S}(conj(x.re), -x.im)
end
function Base.abs2(x::VComplex{S}) where {S}
    (conj(x) * x).re::S
end
function Base.abs(x::VComplex{S}) where {S}
    sqrt(abs2(x))::S
end
function Base.inv(x::VComplex{S}) where {S}
    conj(x) / abs2(x)
end

function Base. +(x::VComplex{S}, y::VComplex{S}) where {S}
    VComplex{S}(x.re + y.re, x.im + y.im)
end
function Base. -(x::VComplex{S}, y::VComplex{S}) where {S}
    VComplex{S}(x.re - y.re, x.im - y.im)
end

function Base. *(a::S, y::VComplex{S}) where {S}
    VComplex{S}(a * y.re, a * y.im)
end
function Base. *(x::VComplex{S}, b::S) where {S}
    VComplex{S}(x.re * b, x.im * b)
end
function Base. \(a::S, y::VComplex{S}) where {S}
    VComplex{S}(a \ y.re, a \ y.im)
end
function Base. /(x::VComplex{S}, b::S) where {S}
    VComplex{S}(x.re / b, x.im / b)
end

# Note: Do not change the order of arguments, as multiplication might
# not be commutative
function Base. *(x::VComplex{S}, y::VComplex{S}) where {S}
    VComplex{S}(x.re * y.re - conj(y.im) * x.im,
                y.im * x.re + x.im * conj(y.re))
end
function Base. /(x::VComplex{S}, y::VComplex{S}) where {S}
    x * inv(y)
end

LinearAlgebra. ⋅(x::VComplex{S}, y::VComplex{S}) where
        {D, S <: Union{Number, VectorSpace}} =
    (conj(x.re) * y.re + conj(x.im) * y.im)::S

incomplete_norm(x::VComplex{S}) where {S <: Union{Number, VectorSpace}} =
    (abs2(x.re) + abs2(x.im))::S
LinearAlgebra.norm(x::VComplex{S}) where
        {S <: Union{AbstractFloat, VectorSpace}} =
    sqrt(incomplete_norm(x))::S



export VQuaternion
struct VQuaternion{S} <: VectorSpace{S}
    val::VComp{VComplex{VComplex{S}}}
end

function Base.show(io::IO, x::VQuaternion{S}) where {S}
    print(io, x.val)
end

Base.firstindex(x::VQuaternion) = firstindex(x.val)
Base.lastindex(x::VQuaternion) = lastindex(x.val)
Base.getindex(x::VQuaternion, i) = getindex(x.val, i)
Base.eltype(::Type{VQuaternion{S}}) where {S} = S
Base.length(x::VQuaternion) = length(x.val)
Base.iterate(x::VQuaternion) = iterate(x.val)
Base.iterate(x::VQuaternion, st) = iterate(x.val, st)
Base.map(f, x::VQuaternion) = VQuaternion(map(f, x.val))
Base.map(f, x::VQuaternion, y::VQuaternion) = VQuaternion(map(f, x.val, y.val))

Base.isequal(x::VQuaternion, y::VQuaternion) = isequal(x.val.v, y.val.v)
Base. ==(x::VQuaternion, y::VQuaternion) = x.val.v == y.val.v



function Arbitrary.arbitrary(::Type{VQuaternion{T}}, ast::ArbState) where {T}
    imap(x -> VQuaternion(VComp(x)), arbitrary(VComplex{VComplex{T}}, ast))
end



stype(::Type{VQuaternion{S}}) where {S} = S
retype(::Type{VQuaternion{S}}, ::Type{T}) where {S, T} = VQuaternion{T}

function Base.zero(::Type{VQuaternion{S}}) where {S}
    VQuaternion(VComp(zero(VComplex{VComplex{S}})))
end
function Base.one(::Type{VQuaternion{S}}) where {S}
    VQuaternion(VComp(one(VComplex{VComplex{S}})))
end

function Base. +(x::VQuaternion{S}) where {S}
    VQuaternion(+x.val)
end
function Base. -(x::VQuaternion{S}) where {S}
    VQuaternion(-x.val)
end
function Base.conj(x::VQuaternion{S}) where {S}
    VQuaternion(VComp(conj(x.val.v)))
end
function Base.abs2(x::VQuaternion{S}) where {S}
    abs2(x.val.v)::S
end
function Base.abs(x::VQuaternion{S}) where {S}
    abs(x.val.v)::S
end
function Base.inv(x::VQuaternion{S}) where {S}
    VQuaternion(VComp(inv(x.val.v)))
end

function Base. +(x::VQuaternion{S}, y::VQuaternion{S}) where {S}
    VQuaternion(x.val + y.val)
end
function Base. -(x::VQuaternion{S}, y::VQuaternion{S}) where {S}
    VQuaternion(x.val - y.val)
end

function Base. *(a::S, y::VQuaternion{S}) where {S}
    VQuaternion(a * y.val)
end
function Base. *(x::VQuaternion{S}, b::S) where {S}
    VQuaternion(x.val * b)
end
function Base. \(a::S, y::VQuaternion{S}) where {S}
    VQuaternion(a \ y.val)
end
function Base. /(x::VQuaternion{S}, b::S) where {S}
    VQuaternion(x.val / b)
end

function Base. *(x::VQuaternion{S}, y::VQuaternion{S}) where {S}
    VQuaternion(VComp(x.val.v * y.val.v))
end
function Base. /(x::VQuaternion{S}, y::VQuaternion{S}) where {S}
    VQuaternion(VComp(x.val.v / y.val.v))
end

LinearAlgebra. ⋅(x::VQuaternion{S}, y::VQuaternion{S}) where
        {D, S <: Union{Number, VectorSpace}} =
    (x.val ⋅ y.val)::S

incomplete_norm(x::VQuaternion{S}) where {S <: Union{Number, VectorSpace}} =
    incomplete_norm(x.val)::S
LinearAlgebra.norm(x::VQuaternion{S}) where
        {S <: Union{AbstractFloat, VectorSpace}} =
    norm(x.val)::S



export VOctonion
struct VOctonion{S} <: VectorSpace{S}
    val::VComp{VComplex{VQuaternion{S}}}
end

function Base.show(io::IO, x::VOctonion{S}) where {S}
    print(io, x.val)
end

Base.firstindex(x::VOctonion) = firstindex(x.val)
Base.lastindex(x::VOctonion) = lastindex(x.val)
Base.getindex(x::VOctonion, i) = getindex(x.val, i)
Base.eltype(::Type{VOctonion{S}}) where {S} = S
Base.length(x::VOctonion) = length(x.val)
Base.iterate(x::VOctonion) = iterate(x.val)
Base.iterate(x::VOctonion, st) = iterate(x.val, st)
Base.map(f, x::VOctonion) = VOctonion(map(f, x.val))
Base.map(f, x::VOctonion, y::VOctonion) = VOctonion(map(f, x.val, y.val))

Base.isequal(x::VOctonion, y::VOctonion) = isequal(x.val.v, y.val.v)
Base. ==(x::VOctonion, y::VOctonion) = x.val.v == y.val.v



function Arbitrary.arbitrary(::Type{VOctonion{T}}, ast::ArbState) where {T}
    imap(x -> VOctonion(VComp(x)), arbitrary(VComplex{VQuaternion{T}}, ast))
end



stype(::Type{VOctonion{S}}) where {S} = S
retype(::Type{VOctonion{S}}, ::Type{T}) where {S, T} = VOctonion{T}

function Base.zero(::Type{VOctonion{S}}) where {S}
    VOctonion(VComp(zero(VComplex{VQuaternion{S}})))
end
function Base.one(::Type{VOctonion{S}}) where {S}
    VOctonion(VComp(one(VComplex{VQuaternion{S}})))
end

function Base. +(x::VOctonion{S}) where {S}
    VOctonion(+x.val)
end
function Base. -(x::VOctonion{S}) where {S}
    VOctonion(-x.val)
end
function Base.conj(x::VOctonion{S}) where {S}
    VOctonion(VComp(conj(x.val.v)))
end
function Base.abs2(x::VOctonion{S}) where {S}
    abs2(x.val.v)::S
end
function Base.abs(x::VOctonion{S}) where {S}
    abs(x.val.v)::S
end
function Base.inv(x::VOctonion{S}) where {S}
    VOctonion(VComp(inv(x.val.v)))
end

function Base. +(x::VOctonion{S}, y::VOctonion{S}) where {S}
    VOctonion(x.val + y.val)
end
function Base. -(x::VOctonion{S}, y::VOctonion{S}) where {S}
    VOctonion(x.val - y.val)
end

function Base. *(a::S, y::VOctonion{S}) where {S}
    VOctonion(a * y.val)
end
function Base. *(x::VOctonion{S}, b::S) where {S}
    VOctonion(x.val * b)
end
function Base. \(a::S, y::VOctonion{S}) where {S}
    VOctonion(a \ y.val)
end
function Base. /(x::VOctonion{S}, b::S) where {S}
    VOctonion(x.val / b)
end

function Base. *(x::VOctonion{S}, y::VOctonion{S}) where {S}
    VOctonion(VComp(x.val.v * y.val.v))
end
function Base. /(x::VOctonion{S}, y::VOctonion{S}) where {S}
    VOctonion(VComp(x.val.v / y.val.v))
end

LinearAlgebra. ⋅(x::VOctonion{S}, y::VOctonion{S}) where
        {D, S <: Union{Number, VectorSpace}} =
    (x.val ⋅ y.val)::S

incomplete_norm(x::VOctonion{S}) where {S <: Union{Number, VectorSpace}} =
    incomplete_norm(x.val)::S
LinearAlgebra.norm(x::VOctonion{S}) where
        {S <: Union{AbstractFloat, VectorSpace}} =
    norm(x.val)::S



export VSedenion
struct VSedenion{S} <: VectorSpace{S}
    val::VComp{VComplex{VOctonion{S}}}
end

function Base.show(io::IO, x::VSedenion{S}) where {S}
    print(io, x.val)
end

Base.firstindex(x::VSedenion) = firstindex(x.val)
Base.lastindex(x::VSedenion) = lastindex(x.val)
Base.getindex(x::VSedenion, i) = getindex(x.val, i)
Base.eltype(::Type{VSedenion{S}}) where {S} = S
Base.length(x::VSedenion) = length(x.val)
Base.iterate(x::VSedenion) = iterate(x.val)
Base.iterate(x::VSedenion, st) = iterate(x.val, st)
Base.map(f, x::VSedenion) = VSedenion(map(f, x.val))
Base.map(f, x::VSedenion, y::VSedenion) = VSedenion(map(f, x.val, y.val))

Base.isequal(x::VSedenion, y::VSedenion) = isequal(x.val.v, y.val.v)
Base. ==(x::VSedenion, y::VSedenion) = x.val.v == y.val.v



function Arbitrary.arbitrary(::Type{VSedenion{T}}, ast::ArbState) where {T}
    imap(x -> VSedenion(VComp(x)), arbitrary(VComplex{VOctonion{T}}, ast))
end



stype(::Type{VSedenion{S}}) where {S} = S
retype(::Type{VSedenion{S}}, ::Type{T}) where {S, T} = VSedenion{T}

function Base.zero(::Type{VSedenion{S}}) where {S}
    VSedenion(VComp(zero(VComplex{VOctonion{S}})))
end
function Base.one(::Type{VSedenion{S}}) where {S}
    VSedenion(VComp(one(VComplex{VOctonion{S}})))
end

function Base. +(x::VSedenion{S}) where {S}
    VSedenion(+x.val)
end
function Base. -(x::VSedenion{S}) where {S}
    VSedenion(-x.val)
end
function Base.conj(x::VSedenion{S}) where {S}
    VSedenion(VComp(conj(x.val.v)))
end
function Base.abs2(x::VSedenion{S}) where {S}
    abs2(x.val.v)::S
end
function Base.abs(x::VSedenion{S}) where {S}
    abs(x.val.v)::S
end
function Base.inv(x::VSedenion{S}) where {S}
    VSedenion(VComp(inv(x.val.v)))
end

function Base. +(x::VSedenion{S}, y::VSedenion{S}) where {S}
    VSedenion(x.val + y.val)
end
function Base. -(x::VSedenion{S}, y::VSedenion{S}) where {S}
    VSedenion(x.val - y.val)
end

function Base. *(a::S, y::VSedenion{S}) where {S}
    VSedenion(a * y.val)
end
function Base. *(x::VSedenion{S}, b::S) where {S}
    VSedenion(x.val * b)
end
function Base. \(a::S, y::VSedenion{S}) where {S}
    VSedenion(a \ y.val)
end
function Base. /(x::VSedenion{S}, b::S) where {S}
    VSedenion(x.val / b)
end

function Base. *(x::VSedenion{S}, y::VSedenion{S}) where {S}
    VSedenion(VComp(x.val.v * y.val.v))
end
function Base. /(x::VSedenion{S}, y::VSedenion{S}) where {S}
    VSedenion(VComp(x.val.v / y.val.v))
end

LinearAlgebra. ⋅(x::VSedenion{S}, y::VSedenion{S}) where
        {D, S <: Union{Number, VectorSpace}} =
    (x.val ⋅ y.val)::S

incomplete_norm(x::VSedenion{S}) where {S <: Union{Number, VectorSpace}} =
    incomplete_norm(x.val)::S
LinearAlgebra.norm(x::VSedenion{S}) where
        {S <: Union{AbstractFloat, VectorSpace}} =
    norm(x.val)::S

end
