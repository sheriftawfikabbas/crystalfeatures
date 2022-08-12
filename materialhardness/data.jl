include("utils.jl")

using BSON
using CSV
using DataFrames
using Dates
using JLD2
using MLDatasets
using Plots
using Printf
using PyCall
using Random
using Statistics

datadir = ENV["HOME"] * "/datasets/materials"

colid(name::Symbol) = colid(string(name))

function colid(name::String)
    name = lowercase(name)
    b = 1
    r = 0
    while length(name) > 0
        c = name[end]
        r += Int(c - 'a' + 1) * b
        name = name[1:end-1]
        b = b * 26
    end
    r
end

@assert colid("AB") == 28
const col = colid

features = Dict(
                :SDFT => [0],
                :Atomic => col(:AXJ):col(:BJX),
                :Geometry => col(:BJY):col(:CGZ),
                :SDFT0 => collect(col(:B):col(:DF)),
                :SDFT_minus5_plus5 => col(:HL):col(:AXI),
                :All => col(:B):col(:CGZ),
               )

i, j =features[:SDFT0], features[:SDFT_minus5_plus5]

features[:SDFT] = j[1:end÷2] ∪ i ∪ j[end÷2+1:end]

features[:AllNoDefective] = features[:SDFT] ∪ features[:Atomic] ∪ features[:Geometry]

function normalize(X, y; normx=:√, normy=:noop) # :log, :√, :std, :noop
    @show normx, normy
    if normx == :log
        @. X = log(X ^ 2 + 1e-7)

    elseif normx == :√
        @. X = sign(X) * √abs(X)

    elseif normx == :std
        S = std(X, dims=2)
        i = vec(S .< 0.5)
        sum(i)
        x = view(X, i, :)
        @. x = log(x ^ 2 + 1e-7)
    end

    # NORMALIZE
    μ, σ = @> X mean(dims=2), std(dims=2)
    @. X = (X - μ) / σ
    @. X[isnan(X)] = 0
    @. X[isinf(X)] = 0

    if normy == :√
        @. y = sign(y) * √abs(y)
    end
    μ, σ, m, M = @> y mean, std, minimum, maximum
    y = @. (y - μ) / σ

    return X, y
    # using Plots
    # histogram(y)
end

label = "K_VRH"
feature=:AllNoDefective
feature=:SDFT0
function data_raw(; label = "K_VRH", feature=:AllNoDefective, removed_outliers=false, normalized=true, normx=:√, normy=:noop)  # "G_VRH"
    X0 = CSV.read("$datadir/full_dataset.csv", DataFrame)
    y0 = CSV.read("$datadir/dataset_elasticity.csv", DataFrame)
    xid = X0[:,1]
    yid = y0[:,2]
    # y = @> y0[indexin(xid, yid), ["K_VRH", "G_VRH"]] Array transpose Array
    # y = @> y0[indexin(xid, yid), ["K_VRH"]] Array transpose Array
    y = @> y0[indexin(xid, yid), [label]] Array transpose Array
    X = @> X0[:, features[feature]] Array transpose Array
    os = @> isinf.(X) findall
    for o in os
        i, j = o.I
        X[i, j] = X[i, j-1]
    end
    if removed_outliers
        outlier(f2) = abs.(f2) .> 3std(f2)
        o = outlier(y)
        sum(o)
        id = .!o |> vec
        X = X[:, id]
        y = y[:, id]
    end
    if normalized
        X, y = normalize(X, Float64.(y); normx, normy)
    end
    X, y
end

function data_updated()
    X = CSV.read("$datadir/full_dataset_updated.csv", DataFrame)
    Z = CSV.read("$datadir/full_dataset.csv", DataFrame)
    A = CSV.read("$datadir/dataset_shsdft_elasticity.csv", DataFrame)
    Xid = X[:, "material_id"]
    Zid = Z[:, "material_id"]
    Aid = A[:, "material_id"]
    j = indexin(Xid, Aid)
    sum(j .!= nothing)
    k = indexin(Zid, Aid)
    sum(k .!= nothing)
end

label = "G_VRH"
feature=:AllNoDefective
removed_outliers=false
normalized=true
normx = :√
normy = :√
# feature=:SDFT0
function data_raw_updated(; label = "G_VRH", feature=:AllNoDefective, removed_outliers=false, normalized=true, normx, normy)  # "G_VRH"
    X0 = CSV.read("data/tmp/full_dataset_updated.csv", DataFrame)
    y0 = CSV.read("data/dataset_elasticity.csv", DataFrame)
    a = CSV.read("data/tmp/dataset_shsdft_elasticity.csv", DataFrame)
    xid = X0[:,1]
    yid = y0[:,2]
    # y = @> y0[indexin(xid, yid), ["K_VRH", "G_VRH"]] Array transpose Array
    # y = @> y0[indexin(xid, yid), ["K_VRH"]] Array transpose Array
    y = @> y0[indexin(xid, yid), [label]] Array transpose Array
    X = @> X0[:, features[feature]] Array transpose Array
    os = @> isinf.(X) findall
    for o in os
        i, j = o.I
        X[i, j] = X[i, j-1]
    end
    if removed_outliers
        outlier(f2) = abs.(f2) .> 3std(f2)
        o = outlier(y)
        sum(o)
        id = .!o |> vec
        X = X[:, id]
        y = y[:, id]
    end
    if normalized
        X, y = normalize(X, Float64.(y); normx, normy)
    end
    X, y
end

function getdata(cols)
    x = CSV.read("data/full_dataset.csv", DataFrame)
    y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    xid = x[:,1]
    yid = y[:,2]
    y2 = y[indexin(xid, yid), ["K_VRH"]]
    x2 = x[:, cols]
    y2
    X = Array(x2)
    y = vec(Array(y2))
    sum(std(X, dims=1) .== 0)
    X = X[:, vec(std(X, dims=1) .!= 0)]
    # m, M = @> X minimum(dims=1), maximum(dims=1)
    # @. X = (X - m) / (M - m)
    μ, σ = @> X mean(dims=1), std(dims=1)
    @. X = (X - μ) / σ
    y = y ./ maximum(y)
    nancols = @> findall(isnan.(X)) getindex.(2) unique
    X = @> X[:, setdiff(1:size(X, 2), nancols)]
    X, y
end

function data_xgboost(; label, feature, removed_outliers)
    X = CSV.read("data/full_dataset_updated.csv", DataFrame)
    Y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    global features
    X2 = innerjoin(X, Y, on = :material_id)
    y = @> X2[:, label] Float64.()
    x = @> X2[:, features[feature]] Array
    if removed_outliers
        outlier(f2) = abs.(f2) .> 3std(f2)
        o = outlier(y)
        sum(o)
        id = .!o |> vec
        x = x[id, :]
        y = y[id]
    end
    x, y
end

function data_update(; label, feature, removed_outliers)
    Z = CSV.read("$datadir/full_dataset.csv", DataFrame)
    A = CSV.read("$datadir/dataset_shsdft_elasticity.csv", DataFrame)
    Y = CSV.read("$datadir/dataset_elasticity.csv", DataFrame)
    names(A)
    global features
    Z2 = innerjoin(Z, A, Y, on = :material_id)
    y = Z2[:, label]
    features[:Updated] = indexin(string.(1:109), names(Z2))
    features[:AllUpdated] = features[:AllNoDefective] ∪ features[:Updated]
    # x1 = @> Z2[:, n1] Array
    # x2 = @> Z2[:, n1 ∪ n2] Array
    x = @> Z2[:, features[feature]] Array
    if removed_outliers
        outlier(f2) = abs.(f2) .> 3std(f2)
        o = outlier(y)
        sum(o)
        id = .!o |> vec
        x = x[id, :]
        y = y[id]
    end
    x, y
end

function data_update_local_full(; feature, removed_outliers)
    X = CSV.read("data/full_dataset_updated.csv", DataFrame)
    Xid = X[:, :material_id]
    Y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    Yid = Y[:, :material_id]
    y = Y[:, :G_VRH]
    X2 = innerjoin(X, Y, on=:material_id)
    x = @> X2[:, features[feature]] Array
    y = X2[:, :G_VRH]
    if removed_outliers
        outlier(f2) = abs.(f2) .> 3std(f2)
        o = outlier(y)
        sum(o)
        id = .!o |> vec
        x = x[id, :]
        y = y[id]
    end
    x, y
end

function data_local_full(; feature, removed_outliers)
    X = CSV.read("data/full_dataset_updated.csv", DataFrame)
    Xid = X[:, :material_id]
    Y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    Yid = Y[:, :material_id]
    y = Y[:, :G_VRH]
    X2 = innerjoin(X, Y, on=:material_id)
    x = @> X2[:, features[feature]] Array
    y = X2[:, :G_VRH]
    if removed_outliers
        outlier(f2) = abs.(f2) .> 3std(f2)
        o = outlier(y)
        sum(o)
        id = .!o |> vec
        x = x[id, :]
        y = y[id]
    end
    x, y
end

