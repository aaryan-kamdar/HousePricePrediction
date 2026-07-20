# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
import pickle

#Importing the data
data=pd.read_csv("AmesHousing-selected-columns.csv")

### Handle missing values first
numeric_columns=["Total Bsmt SF","Garage Cars","Garage Area"]
data[numeric_columns]=data[numeric_columns].fillna(data[numeric_columns].median())
data["Bsmt Qual"]=data["Bsmt Qual"].fillna("None")

### Fix skewed distributions
skewed_features=["Lot Area","1st Flr SF","2nd Flr SF","Gr Liv Area","SalePrice"]
for col in skewed_features:
    data[col] = np.log1p(data[col])
Nominal_features=["Neighborhood"]
### Encode categorical variables correctly
x=pd.get_dummies(data[Nominal_features],dtype=int,drop_first=True)
df=pd.concat([data,x],axis=1)
df=df.drop("Neighborhood",axis=1)
count_dict = df['Bsmt Qual'].value_counts().to_dict()
encoder = OrdinalEncoder(categories=[['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])
df['Bsmt Qual'] = encoder.fit_transform(df[['Bsmt Qual']])

# Create derived features
df["TotalSF"]=df["Total Bsmt SF"]+df["1st Flr SF"]+df["2nd Flr SF"]
df = df.drop(["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"], axis=1)
df=df.drop("Garage Cars",axis=1)

x=df.drop("SalePrice",axis=1)
y=df["SalePrice"]


### Scale numeric features
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)   # fit only on train
x_test_scaled = scaler.transform(x_test)          # transform test using train's stats

#baseline 
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(baseline_pred)))
print("Baseline (predict mean) RMSE ($):", baseline_rmse)

#Linear Regression 

model=LinearRegression()
model.fit(x_train_scaled,y_train)

# Reverse the log1p transform
preds_actual = np.expm1(model.predict(x_test_scaled))
y_test_actual = np.expm1(y_test)

rmse_dollars_lr = np.sqrt(mean_squared_error(y_test_actual, preds_actual))
mae_dollars_lr = np.mean(np.abs(y_test_actual - preds_actual))

print("RMSE ($):", rmse_dollars_lr)
print("MAE ($):", mae_dollars_lr)
print("Average SalePrice:", y_test_actual.mean())
print("RMSE as % of average:", (rmse_dollars_lr / y_test_actual.mean()) * 100)
print("MAE as % of average:", (mae_dollars_lr / y_test_actual.mean()) * 100)

print("RandomForest regressor") 
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(x_train_scaled, y_train)

# random forest check
predicted_actual_rf=np.expm1(rf.predict(x_test_scaled))
y_test_actual_rf=np.expm1(y_test)

rmse_dollars_rf = np.sqrt(mean_squared_error(predicted_actual_rf, y_test_actual_rf))
mae_dollars_rf = np.mean(np.abs(predicted_actual_rf - y_test_actual_rf))

print("RMSE ($):", rmse_dollars_rf)
print("MAE ($):", mae_dollars_rf)

print("Average SalePrice:", y_test_actual.mean())
print("RMSE as % of average:", (rmse_dollars_rf / y_test_actual_rf.mean()) * 100)
print("MAE as % of average:", (mae_dollars_rf / y_test_actual_rf.mean()) * 100)

# XGBOOST
print("XG BOOST")
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4)
xgb.fit(x_train_scaled, y_train)

predicted_actual_xgb=np.expm1(xgb.predict(x_test_scaled))
y_test_actual_xgb=np.expm1(y_test)

rmse_dollars_xgb = np.sqrt(mean_squared_error(predicted_actual_xgb, y_test_actual_xgb))
mae_dollars_xgb = np.mean(np.abs(predicted_actual_xgb - y_test_actual_xgb))

print("RMSE ($):", rmse_dollars_xgb)
print("MAE ($):", mae_dollars_xgb)

print("XG BOOST")
print("Average SalePrice:", y_test_actual.mean())
print("RMSE as % of average:", (rmse_dollars_xgb / y_test_actual_xgb.mean()) * 100)
print("MAE as % of average:", (mae_dollars_xgb / y_test_actual_xgb.mean()) * 100)


# Using grid search cv on xg boost

param_grid={'n_estimators': [300, 500, 800],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,                                   # 5-fold cross-validation
    scoring='neg_root_mean_squared_error',  # sklearn wants "higher is better," so RMSE is negated
    n_jobs=-1,                              # use all CPU cores
    verbose=1                                # prints progress so you can see it's working
)
grid_search.fit(x_train_scaled, y_train)   # y_train is still np.log1p(SalePrice)

print("Best parameters:", grid_search.best_params_)
print("Best CV RMSE (log scale):", -grid_search.best_score_)

best_xgb = grid_search.best_estimator_  # already refit on full training data with best params

predicted_actual_best = np.expm1(best_xgb.predict(x_test_scaled))
y_test_actual = np.expm1(y_test)

rmse_best = np.sqrt(mean_squared_error(y_test_actual, predicted_actual_best))
mae_best = np.mean(np.abs(predicted_actual_best - y_test_actual))

print("Tuned XGBoost RMSE ($):", rmse_best)
print("Tuned XGBoost MAE ($):", mae_best)

print("XG BOOST IMPROVED GRID SEARCH CV")
print("Average SalePrice:", y_test_actual.mean())
print("RMSE as % of average:", (rmse_best / y_test_actual_xgb.mean()) * 100)
print("MAE as % of average:", (mae_best / y_test_actual_xgb.mean()) * 100)


filename = 'xg_boost_best.sav'
pickle.dump(best_xgb, open(filename, 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))
pickle.dump(encoder, open('bsmt_qual_encoder.sav', 'wb'))
pickle.dump(x_train.columns.tolist(), open("columns.pkl", "wb"))