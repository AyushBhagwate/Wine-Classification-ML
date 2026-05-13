from sklearn.model_selection import GridSearchCV


def train_model(model, X_train, Y_train):

    param_grid = {
        'model__n_estimators' : [50, 100, 200],
        'model__max_depth' :  [None, 5, 10, 20],
        'model__min_samples_split' : [2, 5, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1

    ) 

    grid.fit(X_train, Y_train)

    print('Best_params:', grid.best_params_)
    print('Best_scores:',grid.best_score_)

    best_model = grid.best_estimator_


    return best_model