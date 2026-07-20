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
import argparse


# ---- 2. load saved artifacts ----
model = pickle.load(open("xg_boost_best.sav", "rb"))
scaler = pickle.load(open("scaler.sav", "rb"))
bsmt_encoder = pickle.load(open("bsmt_qual_encoder.sav", "rb"))
columns=pickle.load(open("columns.pkl","rb"))

# ---3. Parse CLI ARGUMENTS
parser = argparse.ArgumentParser(
    description="Predict house price using trained XGBoost model."
)

parser.add_argument("--lot_area", type=float, required=True)
parser.add_argument("--overall_qual", type=int, required=True)
parser.add_argument("--overall_cond", type=int, required=True)
parser.add_argument("--year_built", type=int, required=True)
parser.add_argument("--year_remod_add", type=int, required=True)

parser.add_argument("--gr_liv_area", type=float, required=True)
parser.add_argument("--full_bath", type=int, required=True)
parser.add_argument("--bedroom_abvgr", type=int, required=True)
parser.add_argument("--totrms_abvgrd", type=int, required=True)

parser.add_argument("--total_bsmt_sf", type=float, required=True)
parser.add_argument("--first_flr_sf", type=float, required=True)
parser.add_argument("--second_flr_sf", type=float, required=True)

parser.add_argument("--garage_cars", type=int, required=True)
parser.add_argument("--garage_area", type=float, required=True)

parser.add_argument("--bsmt_qual", type=str, required=True)
parser.add_argument("--neighborhood", type=str, required=True)
args = parser.parse_args()

## creating a dataframe out of arguments

data = pd.DataFrame({
    "Lot Area": [args.lot_area],
    "Neighborhood": [args.neighborhood],
    "Overall Qual": [args.overall_qual],
    "Overall Cond": [args.overall_cond],
    "Year Built": [args.year_built],
    "Year Remod/Add": [args.year_remod_add],
    "Bsmt Qual": [args.bsmt_qual],
    "Total Bsmt SF": [args.total_bsmt_sf],
    "1st Flr SF": [args.first_flr_sf],
    "2nd Flr SF": [args.second_flr_sf],
    "Gr Liv Area": [args.gr_liv_area],
    "Full Bath": [args.full_bath],
    "Bedroom AbvGr": [args.bedroom_abvgr],
    "TotRms AbvGrd": [args.totrms_abvgrd],
    "Garage Cars": [args.garage_cars],
    "Garage Area": [args.garage_area]
})

# ---- 3. rebuild the row exactly like training ----
# - log1p the skewed features
for col in ["Lot Area","1st Flr SF","2nd Flr SF","Gr Liv Area"]:
    data[col] = np.log1p(data[col])
    
# - one-hot the neighborhood by hand (all 0s except the chosen one)
Nominal_features=["Neighborhood"]
x=pd.get_dummies(data[Nominal_features],dtype=int)
df=pd.concat([data,x],axis=1)
df=df.drop("Neighborhood",axis=1)

# - encoder.transform() for bsmt_qual (NOT fit_transform)
df["Bsmt Qual"] = bsmt_encoder.transform(df[["Bsmt Qual"]])

# - compute TotalSF
df["TotalSF"]=df["Total Bsmt SF"]+df["1st Flr SF"]+df["2nd Flr SF"]
df = df.drop(["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"], axis=1)
df = df.drop("Garage Cars", axis=1)
# match the train columns 
for col in columns:
    if col not in df.columns:
        df[col] = 0
df = df[columns]
# -----------------------------
# Scale
# -----------------------------

scaled_input = scaler.transform(df)

# -----------------------------
# Predict
# -----------------------------

prediction = model.predict(scaled_input)

# Undo log transform on SalePrice
predicted_price = np.expm1(prediction[0])

print(f"\nPredicted House Price: ${predicted_price:,.2f}")