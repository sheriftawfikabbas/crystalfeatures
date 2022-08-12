include("data.jl")

datasets = pyimport("sklearn.datasets")
xgb = pyimport("xgboost")
train_test_split = pyimport("sklearn.model_selection").train_test_split
GridSearchCV = pyimport("sklearn.model_selection").GridSearchCV
classification_report = pyimport("sklearn.metrics").classification_report
XGBRegressor = pyimport("xgboost").XGBRegressor
cross_val_score = pyimport("sklearn.model_selection").cross_val_score
cross_val_score = pyimport("sklearn.model_selection").cross_val_score
metrics = pyimport("sklearn.metrics")
r2_score = pyimport("sklearn.metrics").r2_score

X, y = data_raw()  # geometry
X = X';
y = y';
@show size(X) size(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

best_r2 = 0
for max_depth = [3, 5, 7]
    for learning_rate = [0.05, 0.1, 0.15]
        for subsample = [0.7, 0.9, 1]
            for n_estimators = [100, 200]
                reg = XGBRegressor(objective="reg:squarederror";
                                   max_depth,
                                   subsample,
                                   learning_rate,
                                   n_estimators,
                                  )
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                r2 = @> r2_score(y_test, y_pred) round(digits=3)
                if best_r2 < r2
                    @show max_depth, learning_rate, subsample, n_estimators, r2
                    best_r2 = r2
                end
            end
        end
    end
end

