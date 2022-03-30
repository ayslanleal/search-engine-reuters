import pandas as pd


read = pd.read_json("./reuters.json")

print(read.describe())