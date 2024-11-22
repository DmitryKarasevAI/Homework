from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
import re
import pickle
import numpy as np

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

with open('best_model.pickle', 'rb') as best_model: # Загружаем модель
    model = pickle.load(best_model)

with open('columns.pickle', 'rb') as model_cols: # Загружаем названия колонок
    columns = pickle.load(model_cols)

def trunc_max_power(value): # Вспоминаем, как предобрабатывать строки в признаки
  if not value is np.nan and value != '0' and value != ' bhp':
    return float(value[:-4])
  elif value == '0':
    return 0.
  else:
    return np.nan

def trunc_mileage(value):
  if not value is np.nan and value[-2:] == 'pl':
    return float(value[:-5])
  elif not value is np.nan and value[-2:] == 'kg':
    return float(value[:-6])
  elif value == '0':
    return 0.
  else:
    return np.nan

def trunc_engine(value):
  if not value is np.nan:
    return float(value[:-3])
  else:
    return np.nan

def find_force(value):
  if value is np.nan:
    return np.nan
  value = value.lower()
  value = value.replace(',', '')
  if 'nm' in value:
    return float(re.findall('\d+', value)[0])
  elif 'kgm' in value:
    return float(re.findall('\d+', value)[0]) / 0.101972 # Converting to nm
  elif len(str(int(re.findall('\d+', value)[0]))) == 3:
    return float(re.findall('\d+', value)[0])
  elif len(str(int(re.findall('\d+', value)[0]))) == 2:
    return float(re.findall('\d+', value)[0]) / 0.101972
  else:
    print(value)
    raise ValueError # Check that all values were found correctly

def find_max_rpm(value):
  if value is np.nan:
    return np.nan
  value = value.lower()
  value = value.replace(',', '')
  if 'rpm' in value:
    return float(re.findall('\d+', value)[-1])
  elif len(re.findall('\d+', value)) > 1:
    return float(re.findall('\d+', value)[-1])
  else:
    return np.nan

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_dict = dict(item.copy())

    # Делаем предобработку признаков как раньше
    item_dict['name'] = item_dict['name'].split()[0] + ' ' + item_dict['name'].split()[1]
    item_dict.pop('selling_price')
    item_dict['mileage'] = trunc_mileage(item_dict['mileage'])
    item_dict['max_power'] = trunc_max_power(item_dict['max_power'])
    item_dict['engine'] = trunc_engine(item_dict['engine'])
    item_dict['max_torque_rpm'] = find_max_rpm(item_dict['torque'])
    item_dict['torque'] = find_force(item_dict['torque'])
    item_dict['seats'] = str(item_dict['seats'])

    numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    categorial_features = ['name', 'seats', 'fuel', 'seller_type', 'transmission', 'owner']
    for feature in categorial_features:
      item_dict[feature+'_'+item_dict[feature]] = 1.
      item_dict.pop(feature)

    for col_1 in numeric_features:
      for col_2 in numeric_features:
        if col_1 == col_2:
          new_name = col_1 + '_squared'
        else:
          new_name = col_1 + '_x_' + col_2
        item_dict[new_name] = item_dict[col_1] * item_dict[col_2]

    user_df = pd.DataFrame(item_dict, columns = columns, index = [0])
    user_df.fillna(0., inplace = True)

    return model.predict(user_df)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    preds = []
    for item in items:
      preds.append(predict_item(item))
    return preds
