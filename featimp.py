import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

def spearman_correlation(x,y):

    n = x.shape[0]
    rank_x = np.arange(1,n+1,1)
    y_sortx =y[np.argsort(x)]
    rank_y = np.argsort(y_sortx)+1

    spearm_corr = (np.cov(rank_x,rank_y)[0][1])/(np.std(rank_x,ddof=1)*np.std(rank_y,ddof=1))

    return spearm_corr

def featureimp_spearman(X,y):
    ranking = dict()

    for col in X.columns:
        x = np.array(X[col].values)
        corr = abs(spearman_correlation(x,y))
        ranking[col]=corr

    ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1], reverse=True)}

    return ranking


def featureimp_mrmr(X,y,factor):
    k = X.shape[1]

    selected = dict()

    candidates = list(X.columns)
    #first element
    dict_import_spearman=featureimp_spearman(X,y)
    key_f=next(iter(dict_import_spearman.keys()))
    val_f=next(iter(dict_import_spearman.values()))

    selected[key_f]=val_f
    candidates.remove(key_f)


    for i in range(k): #until we get all k features in order

        mrmr = dict()

        if len(candidates)>0:
            for col in candidates:
                # maximum relevance
                relevance = abs(spearman_correlation(np.array(X[col]),y))
                # redundance
                subset = X[selected.keys()]
                redundancy = 0

                for j, col_s in enumerate(subset.columns):
                    redundancy += abs(spearman_correlation(np.array(X[col]),np.array(subset[col_s])))

                mrmr[col] = relevance - (redundancy/subset.shape[1])

            mrmr = {k: v for k, v in sorted(mrmr.items(), key=lambda item: item[1], reverse=True)}

            # best next feature
            selected[next(iter(mrmr.keys()))] = next(iter(mrmr.values()))
            candidates.remove(next(iter(mrmr.keys())))

    return selected

def featureimp_dropcolumn(model_rf,X_train, y_train,X_val,y_val):
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_val)
    baseline = mean_squared_error(y_val, y_pred)
    importances = dict()

    for col in X_train.columns:
        # dropping and retraining
        X_new = X_train.drop(col, axis=1)
        model_new = clone(model_rf)
        model_new.fit(X_new, y_train)

        # difference in metric in validation set
        X_new_val = X_val.drop(col, axis=1)
        y_pred = model_new.predict(X_new_val)
        metric = mean_squared_error(y_val, y_pred)


        importances[col]= metric - baseline

    importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
    return importances

def featureimp_permutation(model_rf,X_train, y_train,X_val,y_val):
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_val)
    baseline = mean_squared_error(y_val, y_pred)
    importances = dict()
    X_new=X_train.copy()

    for col in X_val.columns:
        # shuffle the column
        X_new=X_val.copy()
        X_new[col] = np.random.permutation(X_val[col])

        # difference in metric in validation set
        y_pred = model_rf.predict(X_new)
        metric = mean_squared_error(y_val, y_pred)

        importances[col]= metric - baseline

    importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
    return importances
