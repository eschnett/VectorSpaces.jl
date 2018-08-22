module VectorSpaces

using SimpleTraits

struct Private end



################################################################################

@traitdef VectorSpace{V}

@traitimpl VectorSpace{V} <- isvectorspace(V)
function isvectorspace(::Type{V}) where {V}
    try
        S = stype(V)
        S <: Number || return false
        # Check whether functions exist and return the right types
        z = zero(V)::V
        e1 = one(V)::V
        (z * S(1))::V
        S<:AbstractFloat && (z / S(1))::V
        (S(1) * z)::V
        S<:AbstractFloat && (S(1) \ z)::V
        (+e1)::V
        (-e1)::V
        (z + e1)::V
        (z - e1)::V
        (e1 ⋅ e1)::S
        if S<:AbstractFloat
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

export stype
export ⋅
export norm



################################################################################

export Vec
struct Vec{D, S<:Number}
    elts::NTuple{D, S}
    # Vec(elts::NTuple{D, S}) where {D, S<:Number} = new{D, S}(elts)
end

function Base.show(io::IO, x::Vec{D, S}) where {D, S}
    print(io, "$S[")
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
Vec{D, S}(r::AbstractRange{S}) where {D, S<:Number} = Vec{D, S}(tuple(r...))

Base.firstindex(x::Vec) = firstindex(x.elts)
Base.lastindex(x::Vec) = lastindex(x.elts)
Base.getindex(x::Vec, i) = x.elts[i]
# Base.IteratorSize(::Type{Vec{D, S}}) where {D, S} = HasLength()
# Base.IteratorEltype(::Type{Vec{D, S}}) where {D, S} = HasEltype()
Base.eltype(x::Vec) = eltype(x.elts)
# Base.size(x::Vec) = (length(x), )
Base.length(x::Vec) = length(x.elts)
Base.iterate(x::Vec) = iterate(x.elts)
Base.iterate(x::Vec, st) = iterate(x.elts, st)



stype(::Type{Vec{D, S}}) where {D, S<:Number} = S

Base.zero(::Type{Vec{D, S}}) where {D, S<:Number} =
    Vec{D, S}(ntuple(_->zero(S), D))
Base.one(::Type{Vec{D, S}}) where {D, S<:Number} =
    Vec{D, S}(ntuple(_->one(S), D))

Base. +(x::Vec{D, S}) where {D, S<:Number} = Vec{D, S}(map(+, x.elts))
Base. -(x::Vec{D, S}) where {D, S<:Number} = Vec{D, S}(map(-, x.elts))
Base. *(a::S, x::Vec{D, S}) where {D, S<:Number} =
    Vec{D, S}(map(c->a*c, x.elts))
Base. \(a::S, x::Vec{D, S}) where {D, S<:AbstractFloat} =
    Vec{D, S}(map(c->a\c, x.elts))
Base. *(x::Vec{D, S}, a::S) where {D, S<:Number} =
    Vec{D, S}(map(c->c*a, x.elts))
Base. /(x::Vec{D, S}, a::S) where {D, S<:AbstractFloat} =
    Vec{D, S}(map(c->c/a, x.elts))

Base. +(x::Vec{D, S}, y::Vec{D, S}) where {D, S<:Number} =
    Vec{D, S}(map(+, x.elts, y.elts))
Base. -(x::Vec{D, S}, y::Vec{D, S}) where {D, S<:Number} =
    Vec{D, S}(map(-, x.elts, y.elts))

# @assert isvectorspace(Vec{1, Int})



LinearAlgebra. ⋅(x::Vec{D, S}, y::Vec{D, S}) where {D, S<:Number} =
    sum(map(*, x, y))



LinearAlgebra.norm(x::Vec{D, S}) where {D, S<:AbstractFloat} =
    sqrt(x ⋅ x)



################################################################################

export VScalar
struct VScalar{S<:Number}
    elt::S
end

Base.show(io::IO, x::VScalar) = print(io, x.elt)

Base.firstindex(x::VScalar) = 1
Base.lastindex(x::VScalar) = 1
Base.getindex(x::VScalar, i) = x.elt
# Base.IteratorSize(::Type{VScalar{S}}) where {S} = HasLength()
# Base.IteratorEltype(::Type{VScalar{S}}) where {S} = HasEltype()
Base.eltype(x::VScalar) = typeof(x.elt)
# Base.size(x::VScalar) = (length(x), )
Base.length(x::VScalar) = 1
Base.iterate(x::VScalar) = (x.elt, ())
Base.iterate(x::VScalar, st) = nothing



stype(::Type{VScalar{S}}) where {S<:Number} = S

Base.zero(::Type{VScalar{S}}) where {S<:Number} = VScalar{S}(zero(S))
Base.one(::Type{VScalar{S}}) where {S<:Number} = VScalar{S}(one(S))

Base. +(x::VScalar{S}) where {S<:Number} = VScalar{S}(+x.elt)
Base. -(x::VScalar{S}) where {S<:Number} = VScalar{S}(-x.elt)
Base. *(a::S, x::VScalar{S}) where {S<:Number} = VScalar{S}(a * x.elt)
Base. \(a::S, x::VScalar{S}) where {S<:AbstractFloat} = VScalar{S}(a \ x.elt)
Base. *(x::VScalar{S}, a::S) where {S<:Number} = VScalar{S}(x.elt * a)
Base. /(x::VScalar{S}, a::S) where {S<:AbstractFloat} = VScalar{S}(x.elt / a)

Base. +(x::VScalar{S}, y::VScalar{S}) where {S<:Number} =
    VScalar{S}(x.elt + y.elt)
Base. -(x::VScalar{S}, y::VScalar{S}) where {S<:Number} =
    VScalar{S}(x.elt - y.elt)



LinearAlgebra. ⋅(x::VScalar{S}, y::VScalar{S}) where {S<:Number} = x.elt * y.elt



LinearAlgebra.norm(x::VScalar{S}) where {S<:AbstractFloat} = sqrt(x ⋅ x)



################################################################################

export VProd
struct VProd{V1, V2}
   v1::V1
   v2::V2
   function VProd{V1, V2}(v1::V1, v2::V2) where {V1, V2}
       @assert stype(V1) === stype(V2)
       new{V1, V2}(v1, v2)
   end
end

Base.show(io::IO, x::VProd) = print(io, "($(x.v1), $(x.v2))")

Base.firstindex(x::VProd) = 1
Base.lastindex(x::VProd) = length(x)
function Base.getindex(x::VProd, i)
    l1 = length(x.v1)
    if i <= l1
        return x.v1[i - 1 + firstindex(x.v1)]
    else
        return x.v2[i - 1 + firstindex(x.v2)]
    end
end
Base.eltype(x::VProd) = eltype(x.v1)
Base.length(x::VProd) = length(x.v2) + length(x.v2)
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



stype(::Type{VProd{V1, V2}}) where {V1, V2} = stype(V1)

Base.zero(::Type{VProd{V1, V2}}) where {V1, V2} =
    VProd{V1, V2}(zero(V1), zero(V2))
Base.one(::Type{VProd{V1, V2}}) where {V1, V2} =
    VProd{V1, V2}(one(V1), one(V2))

Base. +(x::VProd{V1, V2}) where {V1, V2} =
    VProd{V1, V2}(+x.v1, +x.v2)
Base. -(x::VProd{V1, V2}) where {V1, V2} =
    VProd{V1, V2}(-x.v1, -x.v2)
function Base. *(a::S, x::VProd{V1, V2}) where {V1, V2, S}
    @assert S === stype(V1)
    VProd{V1, V2}(a*x.v1, a*x.v2)
end
function Base. \(a::S, x::VProd{V1, V2}) where {V1, V2, S}
    @assert stype(VProd{V1, V2}) <: AbstractFloat
    @assert S === stype(V1)
    VProd{V1, V2}(a\x.v1, a\x.v2)
end
function Base. *(x::VProd{V1, V2}, a::S) where {V1, V2, S}
    @assert S === stype(V1)
    VProd{V1, V2}(x.v1*a, x.v2*a)
end
function Base. /(x::VProd{V1, V2}, a::S) where {V1, V2, S}
    @assert stype(VProd{V1, V2}) <: AbstractFloat
    @assert S === stype(V1)
    VProd{V1, V2}(x.v1/a, x.v2/a)
end

Base. +(x::VProd{V1, V2}, y::VProd{V1, V2}) where {V1, V2} =
    VProd{V1, V2}(x.v1+y.v1, x.v2+y.v2)
Base. -(x::VProd{V1, V2}, y::VProd{V1, V2}) where {V1, V2} =
    VProd{V1, V2}(x.v1-y.v1, x.v2-y.v2)



LinearAlgebra. ⋅(x::VProd{V1, V2}, y::VProd{V1, V2}) where {V1, V2} =
    x.v1 ⋅ y.v1 + x.v2 ⋅ y.v2



function LinearAlgebra.norm(x::VProd{V1, V2}) where {V1, V2}
    @assert stype(VProd{V1, V2}) <: AbstractFloat
    sqrt(norm(x.v1)^2 + norm(x.v2)^2)
end



################################################################################

export VUnit
struct VUnit{S<:Number} end

Base.show(io::IO, x::VUnit) = print(io, "()")

Base.firstindex(x::VUnit) = 1
Base.lastindex(x::VUnit) = 0
Base.eltype(x::VUnit{S}) where {S} = S
Base.length(x::VUnit) = 0
Base.iterate(x::VUnit) = nothing
Base.iterate(x::VUnit, st) = @assert false



stype(::Type{VUnit{S}}) where {S} = S

Base.zero(::Type{VUnit{S}}) where {S} = VUnit{S}()
Base.one(::Type{VUnit{S}}) where {S} = VUnit{S}()

Base. +(x::VUnit{S}) where {S} = VUnit{S}()
Base. -(x::VUnit{S}) where {S} = VUnit{S}()
Base. *(a::S, x::VUnit{S}) where {S} = VUnit{S}()
Base. \(a::S, x::VUnit{S}) where {S} = VUnit{S}()
Base. *(x::VUnit{S}, a::S) where {S} = VUnit{S}()
Base. /(x::VUnit{S}, a::S) where {S} = VUnit{S}()

Base. +(x::VUnit{S}, y::VUnit{S}) where {S} = VUnit{S}()
Base. -(x::VUnit{S}, y::VUnit{S}) where {S} = VUnit{S}()



LinearAlgebra. ⋅(x::VUnit{S}, y::VUnit{S}) where {S} = S(0)



LinearAlgebra.norm(x::VUnit{S}) where {S} = S(0)

end
