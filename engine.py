import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def train(df, target, features, test_size=.2):
    data=df[features+[target]].replace([np.inf,-np.inf],np.nan).dropna(subset=[target]).copy()
    X=data[features]; y=data[target].astype(float)
    if len(y)<30 or y.nunique()<2: raise ValueError('Need at least 30 valid rows and a varying numeric target.')
    if X.isna().all().any(): raise ValueError('Remove entirely empty features.')
    nums=X.select_dtypes(include='number').columns.tolist(); cats=[c for c in features if c not in nums]
    for c in cats: X[c]=X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    if any(X[c].nunique()>100 for c in cats): raise ValueError('Remove high-cardinality text or ID columns (over 100 categories).')
    a,b,c,d=train_test_split(X,y,test_size=test_size,random_state=42)
    cv=KFold(5,shuffle=True,random_state=42)
    models={}; rows=[]
    for name,est in [('Linear Regression',LinearRegression()),('Decision Tree',DecisionTreeRegressor(max_depth=5,random_state=42)),('Random Forest',RandomForestRegressor(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=1))]:
        prep=ColumnTransformer([('num',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cats)])
        model=Pipeline([('prep',prep),('model',est)])
        scores=-cross_val_score(model,a,c,cv=cv,scoring='neg_mean_absolute_error',error_score='raise')
        model.fit(a,c); p=model.predict(b); models[name]=model
        rows.append({'Model':name,'CV MAE':scores.mean(),'CV MAE SD':scores.std(),'Test MAE':mean_absolute_error(d,p),'Test RMSE':np.sqrt(mean_squared_error(d,p)),'Test R²':r2_score(d,p)})
    results=pd.DataFrame(rows).sort_values('CV MAE'); best=results.iloc[0]['Model']; model=models[best]
    importance=permutation_importance(model,b,d,scoring='neg_mean_absolute_error',n_repeats=5,random_state=42)
    return results,best,model,pd.DataFrame({'Actual':d.values,'Predicted':model.predict(b)}),pd.DataFrame({'Feature':features,'Importance':importance.importances_mean}).sort_values('Importance'),X
