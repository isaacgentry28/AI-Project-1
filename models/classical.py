from dataclasses import dataclass
from typing import Dict, Any

from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR


@dataclass
class ModelSpec:
    name: str
    task: str # 'classification' or 'regression'
    params: Dict[str, Any]
    

def make_model(spec: ModelSpec):
    n = spec.name
    p = spec.params or {}
    if spec.task == 'classification':
        if n == 'logistic':
            return LogisticRegression(max_iter=1000, **p)
        if n == 'tree':
            return DecisionTreeClassifier(**p)
        if n == 'random_forest':
            return RandomForestClassifier(**p)
        if n == 'boosting':
            return GradientBoostingClassifier(**p)
        if n == 'svm':
            return SVC(probability=True, **p)
        raise ValueError(f'Unknown classification model: {n}')
    else:
        if n == 'linear':
            return LinearRegression(**p)
        if n == 'tree':
            return DecisionTreeRegressor(**p)
        if n == 'random_forest':
            return RandomForestRegressor(**p)
        if n == 'boosting':
            return GradientBoostingRegressor(**p)
        if n == 'svm':
            return SVR(**p)
        raise ValueError(f'Unknown regression model: {n}')

