using BSON
using CSV
using CUDA
using DataFrames
using Dates
using Flux
using Flux.Data: DataLoader
using Flux.Losses: logitbinarycrossentropy, logitcrossentropy
using Flux: @functor, chunk, onehotbatch, onecold, throttle, unsqueeze, flatten
using Images
using JLD2
using MLDatasets
using Parameters: @with_kw
using Random
using Statistics

CUDA.seed!(1)

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

include("utils.jl")
include("dataloader2.jl")

features = Dict(
            :SDFT0 => col(:B):col(:DF),
            :SDFT_minus5_plus5 => col(:HL):col(:AXI),
            :Atomic => col(:AXJ):col(:BJX),
            :Geometry => col(:BJY):col(:CGZ),
            :All => col(:B):col(:CGZ),
           )

testbatchsize = 109

function get_data_material(args; f=args.device)
    @extract args batchsize, device
    x = CSV.read("data/full_dataset.csv", DataFrame)
    y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    i, j =features[:SDFT0], features[:SDFT_minus5_plus5]
    cols = j[1:end÷2] ∪ i ∪ j[end÷2+1:end]
    xid = x[:,1]
    yid = y[:,2]
    y2 = y[indexin(xid, yid), ["K_VRH"]]
    x2 = x[:, cols]
    y2
    X = Array(x2)
    y = vec(Array(y2))
    m, M = @> X minimum(dims=1), maximum(dims=1)
    @. X = (X - m) / (M - m)
    # μ, σ = @> X mean(dims=1), std(dims=1)
    # @. X = (X - μ) / σ
    y = y ./ maximum(y)
    isnan.(X) |> sum
    X[isnan.(X)] .= 0
    X = @> X permutedims([2, 1])
    y = Array(y')
    X, y
    n = size(X)[end]
    # itrain = 1:8n÷10
    # itest = 8n÷10+1:n
    JLD2.@load "data/split3.jld2" itrain itest
    xtrain, xtest = X[:, itrain], X[:, itest]
    ytrain, ytest = y[:, itrain], y[:, itest]
    xtrain = @> xtrain shape
    xtest = @> xtest shape
    trainset = DataLoader2((xtrain, ytrain); batchsize=batchsize, f, shuffle=true)
    testset = DataLoader2((xtest, ytest); batchsize=testbatchsize, f)
    return trainset, testset
end

function normalize(X)
    m, M, S = @> X minimum(dims=1), maximum(dims=1), std(dims=1)
    i = vec(S .< 0.5)
    @. X[:, i] = log(X[:, i] ^ 2)
    μ, σ = @> X mean(dims=1), std(dims=1)
    @. X = (X - μ) / σ
    # m, M = @> X3 minimum(dims=1), maximum(dims=1)
    # @. X3 = (X3 - m) / (M - m)
    @. X[isnan(X)] = 0
    @. X[isinf(X)] = 0
    X = @> X permutedims([2, 1])
    X
end

function normalize_img(X)
    @≥ X permutedims([2, 1]) reshape(109, 11, :)
    _, M, S = @> X minimum(dims=[2, 3]), maximum(dims=[2, 3]), std(dims=[2, 3])
    μ, s = @> X mean(dims=[2, 3]), std(dims=[2, 3])
    @. X = (X - μ) / s
    @. X[isnan(X)] = 0
    @. X[isinf(X)] = 0
    X
end

function get_data_fusion(args; f=args.device, position=true)
    @extract args (batchsize,)
    x = CSV.read("data/full_dataset.csv", DataFrame)
    y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    i, j =features[:SDFT0], features[:SDFT_minus5_plus5]
    cols = j[1:end÷2] ∪ i ∪ j[end÷2+1:end]
    # remaining = features[:Atomic] ∪ features[:Geometry]
    remaining = setdiff(features[:All], cols)
    xid = x[:,1]
    yid = y[:,2]
    y2 = y[indexin(xid, yid), ["K_VRH"]]
    x2 = x[:, cols]
    x3 = x[:, remaining]
    y2
    X2 = @> x2 Array normalize
    shape = x -> @> x reshape(109, 11, 1, :)
    if position
        X2 = reshape(X2, 109, 11, :)
        k = 11
        p = sin.((1:k) ./ k)
        p = p'
        p = repeat(p, outer=(1, 1, size(X2, 3)));
        X2 = vcat(X2, p);
        X2 = flatten(X2)
        shape = x -> @> x reshape(110, 11, 1, :)
    end
    X3 = @> x3 Array normalize
    y = @> y2 Array vec transpose Array
    y = y ./ maximum(y)
    # n = size(X2)[end]
    # itrain = 1:8n÷10
    # itest = 8n÷10+1:n
    JLD2.@load "data/split3.jld2" itrain itest
    x2train, x2test = X2[:, itrain], X2[:, itest]
    x3train, x3test = X3[:, itrain], X3[:, itest]
    ytrain, ytest = y[:, itrain], y[:, itest]
    x2train = @> x2train shape
    x2test = @> x2test shape
    trainset = DataLoader2((x2train, x3train, ytrain); batchsize=batchsize, f, shuffle=true)
    testset = DataLoader2((x2test, x3test, ytest); batchsize=testbatchsize, f)
    return trainset, testset
end

function get_data_fusion2(args; f=args.device)
    @extract args (batchsize,)
    x = CSV.read("data/full_dataset.csv", DataFrame)
    y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    i, j =features[:SDFT0], features[:SDFT_minus5_plus5]
    cols = j[1:end÷2] ∪ i ∪ j[end÷2+1:end]
    # remaining = features[:Atomic] ∪ features[:Geometry]
    remaining = setdiff(features[:All], cols)
    xid = x[:,1]
    yid = y[:,2]
    y2 = y[indexin(xid, yid), ["K_VRH"]]
    x2 = x[:, cols]
    x3 = x[:, remaining]
    y2
    X2 = @> x2 Array normalize_img
    X3 = @> x3 Array normalize
    y = @> y2 Array vec transpose Array
    y = y ./ maximum(y)
    JLD2.@load "data/split3.jld2" itrain itest
    x2train, x2test = X2[:, :, itrain], X2[:, :, itest]
    x3train, x3test = X3[:, itrain], X3[:, itest]
    ytrain, ytest = y[:, itrain], y[:, itest]
    trainset = DataLoader2((x2train, x3train, ytrain); batchsize=batchsize, f, shuffle=true)
    testset = DataLoader2((x2test, x3test, ytest); batchsize=testbatchsize, f)
    return trainset, testset
end

function get_data_transformer(args; f=args.device)
    @extract args (batchsize,)
    x = CSV.read("data/full_dataset.csv", DataFrame)
    y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    i, j =features[:SDFT0], features[:SDFT_minus5_plus5]
    cols = j[1:end÷2] ∪ i ∪ j[end÷2+1:end]
    # remaining = features[:Atomic] ∪ features[:Geometry]
    remaining = setdiff(features[:All], cols)
    xid = x[:,1]
    yid = y[:,2]
    y2 = y[indexin(xid, yid), ["K_VRH"]]
    x2 = x[:, cols]
    x3 = x[:, remaining]
    y2
    X2 = @> x2 Array normalize
    shape = x -> @> x reshape(109, 11, :)
    X3 = @> x3 Array normalize
    y = @> y2 Array vec transpose Array
    y = y ./ maximum(y)
    # n = size(X2)[end]
    # itrain = 1:8n÷10
    # itest = 8n÷10+1:n
    JLD2.@load "data/split3.jld2" itrain itest
    x2train, x2test = X2[:, itrain], X2[:, itest]
    x3train, x3test = X3[:, itrain], X3[:, itest]
    ytrain, ytest = y[:, itrain], y[:, itest]
    x2train = @> x2train shape
    x2test = @> x2test shape
    trainset = DataLoader2((x2train, x3train, ytrain); batchsize=batchsize, f, shuffle=true)
    testset = DataLoader2((x2test, x3test, ytest); batchsize=testbatchsize, f)
    return trainset, testset
end

function normalize3(X)
    shape = size(X)
    X = Flux.flatten(X);
    l, u, s = @> X minimum(dims=2), maximum(dims=2), std(dims=2)
    i = vec(s .< 0.5)
    @. X[i, :] = log(X[i, :] ^ 2)
    # Z = view(X, i, :)
    # l = minimum(Z[.!isinf.(Z)])
    # @. Z[isinf(Z)] = l
    # @. X[isinf(X)]
    μ, s = @> X mean(dims=2), std(dims=2)
    @. X = (X - μ) / s
    # l, u = @> X3 minimum(dims=1), maximum(dims=1)
    # @. X3 = (X3 - l) / (u - l)
    @. X[isnan(X)] = 0
    @. X[isinf(X)] = 0
    X = @> X reshape(shape);
    X
end

function get_data_transformer_fusion(args; f=args.device)
    @extract args (batchsize,)
    JLD2.@load "data/4-features.jld2" sdft atomic geometry defective y
    @≥ sdft, atomic, geometry normalize3.();
    y = @> y transpose Array
    y = y ./ maximum(y)
    JLD2.@load "data/split3.jld2" itrain itest
    x1train, x1test = sdft[:, :, itrain], sdft[:, :, itest];
    x2train, x2test = atomic[:, :, itrain], atomic[:, :, itest];
    x3train, x3test = geometry[:, :, itrain], geometry[:, :, itest];
    ytrain, ytest = y[:, itrain], y[:, itest]
    trainset = DataLoader2((x1train, x2train, x3train, ytrain); batchsize=batchsize, f, shuffle=true)
    testset = DataLoader2((x1test, x2test, x3test, ytest); batchsize=testbatchsize, f)
    return trainset, testset
end

function get_data_transformer_fusion_old(args; f=args.device)
    @extract args (batchsize,)
    JLD2.@load "data/4-features.jld2" sdft atomic geometry defective y
    sdft = (sdft .- mean(sdft)) ./ std(sdft);
    atomic = (atomic .- mean(atomic)) ./ std(atomic);
    geometry = (geometry .- mean(geometry)) ./ std(geometry);
    y = @> y transpose Array
    y = y ./ maximum(y)
    JLD2.@load "data/split3.jld2" itrain itest
    x1train, x1test = sdft[:, :, itrain], sdft[:, :, itest];
    x2train, x2test = atomic[:, :, itrain], atomic[:, :, itest];
    x3train, x3test = geometry[:, :, itrain], geometry[:, :, itest];
    ytrain, ytest = y[:, itrain], y[:, itest]
    trainset = DataLoader2((x1train, x2train, x3train, ytrain); batchsize=batchsize, f, shuffle=true)
    testset = DataLoader2((x1test, x2test, x3test, ytest); batchsize=testbatchsize, f)
    return trainset, testset
end

function rsquared(ypred, y)
    SSₜ = sum((y .- mean(y)).^2)
    SSᵣ = sum((ypred .- y).^2)
    R² = 1 .- SSᵣ / SSₜ
end

function data2(cluster = 1)
    JLD2.@load "data/2clusters.jld2" x1 y1 x2 y2
    x, y = cluster == 1 ? (x1, y1) : (x2, y2)
    μ, σ = @> x mean(dims=2), std(dims=2)
    @. x = (x - μ) / σ
    y = y ./ maximum(y)
    x, y
end

function data2_20(cluster = 1)
    JLD2.@load "data/2clusters-20.jld2" x1 y1 x2 y2
    x, y = cluster == 1 ? (x1, y1) : (x2, y2)
    μ, σ = @> x mean(dims=2), std(dims=2)
    @. x = (x - μ) / σ
    y = y ./ maximum(y)
    x, y
end

function data2_cosine(cluster = 1)
    JLD2.@load "data/2clusters-cosine.jld2" x1 y1 x2 y2
    x, y = cluster == 1 ? (x1, y1) : (x2, y2)
    μ, σ = @> x mean(dims=2), std(dims=2)
    @. x = (x - μ) / σ
    y = y ./ maximum(y)
    x, y
end

function data_xgboost()
    @extract args batchsize, device
    x = CSV.read("data/full_dataset.csv", DataFrame)
    y = CSV.read("data/dataset_elasticity.csv", DataFrame)
    i, j =features[:SDFT0], features[:SDFT_minus5_plus5]
    cols = j[1:end÷2] ∪ i ∪ j[end÷2+1:end]
    xid = x[:,1]
    yid = y[:,2]
    y2 = y[indexin(xid, yid), ["K_VRH"]]
    x2 = x[:, cols]
    y2
    X = Array(x2)
    y = vec(Array(y2))
    m, M = @> X minimum(dims=1), maximum(dims=1)
    @. X = (X - m) / (M - m)
    # μ, σ = @> X mean(dims=1), std(dims=1)
    # @. X = (X - μ) / σ
    y = y ./ maximum(y)
    isnan.(X) |> sum
    X[isnan.(X)] .= 0
    X = @> X permutedims([2, 1])
    y = Array(y')
    X, y
    n = size(X)[end]
    # itrain = 1:8n÷10
    # itest = 8n÷10+1:n
    JLD2.@load "data/split3.jld2" itrain itest
    xtrain, xtest = X[:, itrain], X[:, itest]
    ytrain, ytest = y[:, itrain], y[:, itest]
    xtrain = @> xtrain shape
    xtest = @> xtest shape
    trainset = DataLoader2((xtrain, ytrain); batchsize=batchsize, f, shuffle=true)
    testset = DataLoader2((xtest, ytest); batchsize=testbatchsize, f)
    return trainset, testset
end

