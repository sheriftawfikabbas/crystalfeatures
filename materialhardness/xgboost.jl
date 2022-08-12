include("data.jl")
using PyCall
using Printf
using Random
Random.seed!(1)

datasets = pyimport("sklearn.datasets")
xgb = pyimport("xgboost")
train_test_split = pyimport("sklearn.model_selection").train_test_split
classification_report = pyimport("sklearn.metrics").classification_report
XGBRegressor = pyimport("xgboost").XGBRegressor
cross_val_score = pyimport("sklearn.model_selection").cross_val_score
metrics = pyimport("sklearn.metrics")

normx = :√
normy = :noop
seed = 1
label = "K_VRH"  # bulk_modulus
feature = :AllNoDefective
removed_outliers = false

function xgboost_each_fold(; get_data)
    @show seed, removed_outliers, normx, normy
    max_depth = 5
    subsample = 0.9
    learning_rate = 0.1
    n_estimators = 200
    df = DataFrame(Seed=Int[], Fold=Int[], N=Int[], Feature=Symbol[], Norm=String[], R2=Float64[])
    for fold = 1:5
        X, y = get_data(; label, feature, removed_outliers)
        y = if normy == :√
            @. sign(y) * √(abs(y))
        elseif normy == :log
            @. sign(y) * log(1 + abs(y))
        else
            y
        end
        isna(x) = isinf(x) | isnan(x)
        X[isna.(X)] .= 0
        # using Plots
        # plot(histogram(vec(y)))
        @show size(X) size(y)
        n = size(X)[end]
        # KFold = pyimport("sklearn.model_selection").KFold
        # skf = KFold(n_splits=5, random_state=seed, shuffle=true)
        # (fold, (itrain, itest)) = collect(enumerate(skf.split(X, y)))[fold]
        StratifiedKFold = pyimport("sklearn.model_selection").StratifiedKFold
        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=true)
        ybin = round.(Int, 10y)
        (_, (itrain, itest)) = collect(enumerate(skf.split(X, ybin)))[fold]
        itrain .+= 1
        itest .+= 1
        X_train, X_test = X[itrain, :], X[itest, :]
        y_train, y_test = y[itrain], y[itest]
        # @show seed, fold, length(y_train), length(y_test)
        reg = XGBRegressor(objective="reg:squarederror";
                           max_depth,
                           subsample,
                           learning_rate,
                           n_estimators,
                          )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = @> r2_score(y_test, y_pred) round(digits=3)
        push!(df, (seed, fold, n, feature, "$normx$normy", r2))
        @show feature, r2
    end
    df
end

function xgboost_each_feature(; get_data, fold=1)
    @show seed, fold, removed_outliers, normx, normy
    max_depth = 5
    subsample = 0.9
    learning_rate = 0.1
    n_estimators = 200
    df = DataFrame(Seed=Int[], Fold=Int[], N=Int[], Feature=Symbol[], Norm=String[], R2=Float64[])
    all_features = [:Atomic, :SDFT_minus5_plus5, :SDFT0, :SDFT, :AllNoDefective, :Geometry, ] ∪ (get_data == data_update ? [:Updated, :AllUpdated] : [])
    for feature in all_features
        X, y = get_data(; label, feature, removed_outliers)
        y = if normy == :√
            @. sign(y) * √(abs(y))
        elseif normy == :log
            @. sign(y) * log(1 + abs(y))
        else
            y
        end
        # using Plots
        # plot(histogram(vec(y)))
        @show size(X) size(y)
        n = size(X)[end]
        ybin = round.(Int, 10y)
        # KFold = pyimport("sklearn.model_selection").KFold
        # skf = KFold(n_splits=5, random_state=seed, shuffle=true)
        # (fold, (itrain, itest)) = collect(enumerate(skf.split(X, y)))[fold]
        StratifiedKFold = pyimport("sklearn.model_selection").StratifiedKFold
        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=true)
        (_, (itrain, itest)) = collect(enumerate(skf.split(X, ybin)))[fold]
        itrain .+= 1
        itest .+= 1
        X_train, X_test = X[itrain, :], X[itest, :]
        y_train, y_test = y[itrain], y[itest]
        # @show seed, fold, length(y_train), length(y_test)
        reg = XGBRegressor(objective="reg:squarederror";
                           max_depth,
                           subsample,
                           learning_rate,
                           n_estimators,
                          )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = @> r2_score(y_test, y_pred) round(digits=3)
        push!(df, (seed, fold, n, feature, "$normx$normy", r2))
        @show feature, r2
    end
    df
end

# get_data = data_xgboost
get_data = data_update  # function to get data (X, y)
d2 = xgboost_each_feature(; get_data)  # run on a 5-fold split and report R2 score for each feature mention above. This is for reporting feature importance
d1 = xgboost_each_fold(; get_data)  # run 5 fold CV and report R2 score for each fold

