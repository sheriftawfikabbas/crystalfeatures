using MacroTools: postwalk, striplines, isexpr, @forward, @capture, animals, striplines
using Random

"Fast zip+splat"
zips(a::Vector{T}) where T = map(t-> map(x->x[t], a), 1:length(first(a)))

zips(a::Vector{<:Tuple}) = tuple(map(t-> map(x->x[t], a), 1:length(first(a)))...)

zips(a::T) where {T<:Tuple} = map(t-> map(x->x[t], a), 1:length(first(a)))

""" cat+splat

    julia> cats([rand(1, 2) for _=1:3], dims=1)
    3×2 Matrix{Float64}:
     0.701238  0.704747
     0.464978  0.769693
     0.800235  0.947824
"""
cats(xs; dims=ndims(xs[1])+1) = cat(xs..., dims=dims)
vcats(xs) = vcat(xs...)

macro extract(m, vs)
  rhs = Expr(:tuple)
  for v in vs.args
    push!(rhs.args, :($m.$v))
  end
  ex = :($vs = $rhs) |> striplines
  esc(ex)
end

"""
Extends Lazy's @>
"""
macro >(exs...)
  @assert length(exs) > 0
  callex(head, f, x, xs...) = ex = :_ in xs ? Expr(:call, Expr(head, f, xs...), x) : Expr(head, f, x, xs...)
  thread(x) = isexpr(x, :block) ? thread(rmlines(x).args...) : x
  thread(x, ex) =
    if isexpr(ex, :call, :macrocall)
      callex(ex.head, ex.args[1], x, ex.args[2:end]...)
    elseif isexpr(ex, :tuple)
      Expr(:tuple,
           map(ex -> isexpr(ex, :call, :macrocall) ?
               callex(ex.head, ex.args[1], x, ex.args[2:end]...) :
               Expr(:call, ex, x), ex.args)...)
    elseif @capture(ex, f_.(xs__))
      :($f.($x, $(xs...)))
    elseif isexpr(ex, :block)
      thread(x, rmlines(ex).args...)
    else
      Expr(:call, ex, x)
    end
  thread(x, exs...) = reduce(thread, exs, init=x)
  esc(thread(exs...))
end

"Extend @>"
macro >=(x, exs...)
  esc(macroexpand(Main, :($x = @> $x $(exs...))))
end
# alias
var"@≥" = var"@>="

function copystruct!(a::T, b::U) where {T, U}
    for f in fieldnames(T)
        setfield!(a, f, getfield(b, f))
    end
    b
end

call(f, x) = f(x)

using PyCall
r2_score = pyimport("sklearn.metrics").r2_score

rsquared = r2_score

