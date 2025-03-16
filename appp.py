import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
import sklearn
sklearn.set_config(transform_output="pandas")


st.title("Приложение для предсказания")

test_df = st.file_uploader("загрузи файл", type="csv")

if test_df is not None:
    test_df = pd.read_csv(test_df)
    train  = pd.read_csv("train.csv")
    model = joblib.load('model.pkl')

    df = pd.concat([train,test_df],axis=0)
    df = df.drop(columns=["Id", "Alley", "PoolQC", "Fence", "MiscFeature", "MoSold", "3SsnPorch", 
                        "BsmtFinSF2", "BsmtHalfBath", "MiscVal", "LowQualFinSF", "YrSold"])

    df["MSZoning"] = df.groupby("Neighborhood")["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Utilities"] = df.groupby("SaleCondition")["Utilities"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Exterior1st"] = df.groupby("Neighborhood")["Exterior1st"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Exterior2nd"] = df.groupby("Neighborhood")["Exterior2nd"].transform(lambda x: x.fillna(x.mode()[0]))
    df.loc[df["MasVnrType"].isnull(), "MasVnrType"] = "None"
    df["MasVnrArea"] = df.groupby("LotFrontage")["MasVnrArea"].transform(lambda x: x.fillna(x.mean()))
    df.loc[df["BsmtQual"].isnull(), "BsmtQual"] = "Na"
    df.loc[df["BsmtCond"].isnull(), "BsmtCond"] = "Na"
    df.loc[df["BsmtExposure"].isnull(), "BsmtExposure"] = "Na"
    df.loc[df["BsmtFinType1"].isnull(), "BsmtFinType1"] = "Na"
    df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
    df["BsmtFinType2"] = df.groupby("BsmtCond")["BsmtFinType2"].transform(lambda x: x.fillna(x.mode()[0]))
    df["BsmtUnfSF"] = df["BsmtUnfSF"].transform(lambda x: x.fillna(x.mean()))
    df["TotalBsmtSF"] = df.groupby("Neighborhood")["TotalBsmtSF"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Electrical"] = df["Electrical"].transform(lambda x: x.fillna(x.mode()[0]))
    df["BsmtFullBath"] = df["BsmtFullBath"].transform(lambda x: x.fillna(x.mode()[0]))
    df["KitchenQual"] = df.groupby("Neighborhood")["KitchenQual"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Functional"] = df["Functional"].transform(lambda x: x.fillna(x.mode()[0]))
    df['FireplaceQu'] = df['FireplaceQu'].fillna("Na")
    df['GarageType'] = df['GarageType'].fillna("Na")
    df["GarageYrBlt"] = df.groupby("Neighborhood")["GarageYrBlt"].transform(lambda x: x.fillna(x.mean()))
    df['GarageFinish'] = df['GarageFinish'].fillna("Na")
    df['GarageCars'] = df['GarageCars'].fillna(0)
    df["GarageArea"] = df["GarageArea"].transform(lambda x: x.fillna(x.mean()))
    df['GarageQual'] = df['GarageQual'].fillna("Na")
    df['GarageCond'] = df['GarageCond'].fillna("Na")
    df["SaleType"] = df["SaleType"].transform(lambda x: x.fillna(x.mode()[0]))

    street = {'Grvl': 0, 'Pave': 1}
    df['Street'] = df['Street'].map(street)
    lotshape = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
    df['LotShape'] = df['LotShape'].map(lotshape)
    landcontrol = {'HLS': 0, 'Bnk': 1, 'Low': 2, 'Lvl': 3}
    df['LandContour'] = df['LandContour'].map(landcontrol)
    utilites = {'NoSeWa': 0, 'AllPub': 1}
    df['Utilities'] = df['Utilities'].map(utilites)
    landslope = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
    df['LandSlope'] = df['LandSlope'].map(landslope)
    manstype = {'None': 0, 'BrkCmn': 1, 'Stone': 2, 'BrkFace': 3}
    df['MasVnrType'] = df['MasVnrType'].map(manstype)
    extra = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
    df['ExterQual'] = df['ExterQual'].map(extra)
    extracond = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df['ExterCond'] = df['ExterCond'].map(extracond)
    fundament = {'Wood': 0, 'Stone': 1, 'CBlock': 2, 'BrkTil': 3, 'PConc': 4, 'Slab':5}
    df['Foundation'] = df['Foundation'].map(fundament)
    bsmt = {'Na': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df['BsmtQual'] = df['BsmtQual'].map(bsmt)
    bsmtcond = {'Na': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}
    df['BsmtCond'] = df['BsmtCond'].map(bsmtcond)
    bsmtexp = {'Na': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    df['BsmtExposure'] = df['BsmtExposure'].map(bsmtexp)
    BsmtFin = {'Na': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4,'ALQ':5, 'GLQ':6}
    df['BsmtFinType1'] = df['BsmtFinType1'].map(BsmtFin)
    BsmtFin2 = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4,'GLQ':5}
    df['BsmtFinType2'] = df['BsmtFinType2'].map(BsmtFin2)
    heatingqs = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
    df['HeatingQC'] = df['HeatingQC'].map(heatingqs)
    air = {'N': 0, 'Y': 1}
    df['CentralAir'] = df['CentralAir'].map(air)
    Kitchen = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}
    df['KitchenQual'] = df['KitchenQual'].map(Kitchen)
    fire = {'Na': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd':4, 'Ex':5}
    df['FireplaceQu'] = df['FireplaceQu'].map(fire)
    garage = {'Na': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 'Basment':4, 'Attchd':5, '2Types':6}
    df['GarageType'] = df['GarageType'].map(garage)
    garage2 = {'Na': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    df['GarageFinish'] = df['GarageFinish'].map(garage2)
    garage3 = {'Na': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd':4, 'Ex':5}
    df['GarageQual'] = df['GarageQual'].map(garage3)
    df['GarageCond'] = df['GarageCond'].map(garage3)
    paved = {'N': 0, 'P': 1, 'Y': 2}
    df['PavedDrive'] = df['PavedDrive'].map(paved)

    standard_scaler_columns = ['MSSubClass','LotFrontage','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1', 'TotalBsmtSF',
                            'GrLivArea','GarageYrBlt', 'GarageArea','ScreenPorch','EnclosedPorch','PoolArea']
    RobustScaler_columns = ['LotArea','BsmtUnfSF','1stFlrSF', '2ndFlrSF','WoodDeckSF','OpenPorchSF']


    my_scaler = ColumnTransformer(
    [
            ('standard_scaler', StandardScaler(), standard_scaler_columns),
            ('RobustScaler', RobustScaler(), RobustScaler_columns)
        ],
        verbose_feature_names_out = False,
        remainder = 'passthrough')

    df = my_scaler.fit_transform(df)

    y = train["SalePrice"]
    y = np.log1p(y)

    test = df.iloc[len(train):]
    test = test.drop('SalePrice', axis=1)
    train_x = df.iloc[:len(train)]
    train_x = train_x.drop('SalePrice', axis=1)


    test_preds = model.predict(test)
    test_preds = np.expm1(test_preds)
    pred = pd.DataFrame(test_preds)
    st.write("Предсказания:")
    st.dataframe(pred)