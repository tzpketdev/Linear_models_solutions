from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def torque_cleaner(test_tor, coeff_1 = 1):
    list_split = []
    dict_kgm_nm = {'kgm': 1, 'nm': 9.80665}
    str_test = ''
    for i in test_tor:
        i = i.lower()
        if i.isdigit() or i=='.':
            str_test += i
        if (i.isdigit() == 0) and (i != '.') and (str_test != ''):
            list_split.append(float(str_test))
            str_test = ''
    for key in dict_kgm_nm.keys():
        if key in test_tor:
            coeff_1 = dict_kgm_nm[key]

    return [np.round(list_split[0]*coeff_1, 2), list_split[-1]]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    test_item = dict(item)
    input_data = pd.DataFrame.from_dict([test_item])

    input_data['mileage'] = input_data['mileage'].apply(
        lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
    input_data['engine'] = input_data['engine'].apply(
        lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
    input_data['max_power'] = input_data['max_power'].apply(
        lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)

    input_data['torque_new'] = (input_data['torque'] + '@') \
        .fillna('0.0@').replace(',', '', regex=True).apply(torque_cleaner)

    input_data['torque'] = input_data['torque_new'].apply(lambda x: float(x[0])).replace(0.0, np.nan)
    input_data['max_torque'] = input_data['torque_new'].apply(lambda x: float(x[1])).replace(0.0, np.nan)

    input_data = input_data.drop('torque_new', axis=1)

    input_data[['engine', 'seats']] = input_data[['engine', 'seats']].astype(int)

    with open('best_model.pickle', 'rb') as f:
        model = pickle.load(f)
    num_cols = list(model['scaler'].feature_names_in_)
    cat_cols = list(model['ohe'].feature_names_in_)

    X_cat = pd.DataFrame(model['ohe'].transform(input_data[cat_cols]).toarray())
    X_cat.columns = X_cat.columns.astype(str)

    X_num = pd.DataFrame(model['scaler'].transform(input_data[num_cols]), columns=num_cols)

    X = pd.concat([X_cat, X_num], axis=1)
    feat_order = list(model['best_model'].feature_names_in_)

    return model['best_model'].predict(X[feat_order])

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    prediction_list = []
    for item in items:
        test_item = dict(item)
        input_data = pd.DataFrame.from_dict([test_item])

        input_data['mileage'] = input_data['mileage'].apply(
            lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
        input_data['engine'] = input_data['engine'].apply(
            lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)
        input_data['max_power'] = input_data['max_power'].apply(
            lambda x: float(str(x).split()[0]) if len(str(x).split()) == 2 else np.nan)

        input_data['torque_new'] = (input_data['torque'] + '@') \
            .fillna('0.0@').replace(',', '', regex=True).apply(torque_cleaner)

        input_data['torque'] = input_data['torque_new'].apply(lambda x: float(x[0])).replace(0.0, np.nan)
        input_data['max_torque'] = input_data['torque_new'].apply(lambda x: float(x[1])).replace(0.0, np.nan)

        input_data = input_data.drop('torque_new', axis=1)

        input_data[['engine', 'seats']] = input_data[['engine', 'seats']].astype(int)

        with open('best_model.pickle', 'rb') as f:
            model = pickle.load(f)
        num_cols = list(model['scaler'].feature_names_in_)
        cat_cols = list(model['ohe'].feature_names_in_)

        X_cat = pd.DataFrame(model['ohe'].transform(input_data[cat_cols]).toarray())
        X_cat.columns = X_cat.columns.astype(str)

        X_num = pd.DataFrame(model['scaler'].transform(input_data[num_cols]), columns=num_cols)

        X = pd.concat([X_cat, X_num], axis=1)
        feat_order = list(model['best_model'].feature_names_in_)

        prediction = model['best_model'].predict(X[feat_order])
        prediction_list.append(prediction)
    return prediction_list
