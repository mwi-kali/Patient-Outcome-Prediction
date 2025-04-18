
import optuna

import numpy as np

from .config import RANDOM_STATE
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


        
def objective_gb(trial, X, y, preprocessor):
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    n = trial.suggest_int('n', 50, 300)
    sub = trial.suggest_float('sub', 0.6, 1.0)
    clf = GradientBoostingClassifier(
        learning_rate=lr, n_estimators=n,
        subsample=sub, random_state=RANDOM_STATE
    )
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipe, X, y, cv=cv, scoring='f1').mean()


def objective_logreg(trial, X, y, preprocessor):
    C = trial.suggest_float('C', 1e-3, 1e2, log=True)
    clf = LogisticRegression(
        solver='saga', C=C, penalty='l2', max_iter=5000,
        random_state=RANDOM_STATE
    )
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipe, X, y, cv=cv, scoring='f1').mean()


def objective_rf(trial, X, y, preprocessor):
    n = trial.suggest_int('n', 50, 200)
    depth = trial.suggest_int('depth', 3, 10)
    feat = trial.suggest_categorical('feat', ['sqrt', 'log2'])
    clf = RandomForestClassifier(
        n_estimators=n, max_depth=depth,
        max_features=feat, random_state=RANDOM_STATE
    )
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipe, X, y, cv=cv, scoring='f1').mean()


def tune_and_train(X_train, y_train, preprocessor):
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    def multi_objective(trial):
        model_choice = trial.suggest_categorical('model', ['logreg', 'rf', 'gb'])
        if model_choice == 'logreg':
            return objective_logreg(trial, X_train, y_train, preprocessor)
        elif model_choice == 'rf':
            return objective_rf(trial, X_train, y_train, preprocessor)
        else:
            return objective_gb(trial, X_train, y_train, preprocessor)

    study.optimize(multi_objective, n_trials=60)

    best = study.best_params
    if best['model'] == 'logreg':
        final_clf = LogisticRegression(
            solver='saga', C=best['C'], penalty='l2',
            max_iter=5000, random_state=RANDOM_STATE
        )
    elif best['model'] == 'rf':
        final_clf = RandomForestClassifier(
            n_estimators=best['n'], max_depth=best['depth'],
            max_features=best['feat'], random_state=RANDOM_STATE
        )
    else:
        final_clf = GradientBoostingClassifier(
            learning_rate=best['lr'],
            n_estimators=best['n'],
            subsample=best['sub'],
            random_state=RANDOM_STATE
        )

    uncal_pipeline = Pipeline([('pre', preprocessor), ('clf', final_clf)])
    cal_pipeline   = CalibratedClassifierCV(uncal_pipeline, cv=5)
    cal_pipeline.fit(X_train, y_train)
    uncal_pipeline.fit(X_train, y_train)
    return uncal_pipeline, cal_pipeline, study