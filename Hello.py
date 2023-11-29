import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Changes to the prices over the years in European Countries (Amar)
# Prices of Big Macs in different countries (Amar)
df_bigmac = pd.read_csv('BigmacPrice.csv')
df_bigmac['date'] = pd.to_datetime(df_bigmac['date'])
one_year_ago = df_bigmac['date'].max() - timedelta(days=365)
df_bigmac = df_bigmac[df_bigmac['date'] >= one_year_ago]

df_bigmac = df_bigmac.drop(['date', 'dollar_ex', 'local_price', 'currency_code'], axis=1)
df_bigmac = df_bigmac.dropna()
fig, ax = plt.subplots()

counts = df_bigmac.groupby('name')['dollar_price'].mean()
counts = counts.sort_index()
counts.plot(kind='bar')

fig.autofmt_xdate()
plt.ylabel('Price in USD')
plt.title('Prices of Big Macs in different countries')