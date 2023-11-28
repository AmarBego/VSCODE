import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Changes to the prices over the years in European Countries (Amar)
df_bigmac = pd.read_csv('BigmacPrice.csv')
df_bigmac = df_bigmac.drop(['dollar_ex', 'currency_code'], axis=1)
df_bigmac = df_bigmac.dropna()

european_countries = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City']

df_bigmac = df_bigmac[df_bigmac['name'].isin(european_countries)]

df_bigmac['date'] = pd.to_datetime(df_bigmac['date'])
df_bigmac['date'] = df_bigmac['date'].dt.year

countries = df_bigmac['name'].unique()

plt.figure(figsize=(10, 6))

for country in countries:
    country_data = df_bigmac[df_bigmac['name'] == country]
    counts = country_data.groupby('date')['dollar_price'].mean()
    plt.plot(counts.index, counts.values, label=country)

plt.xticks(np.arange(min(df_bigmac['date']), max(df_bigmac['date'])+1, 1.0))

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Big Mac Prices in $USD')
plt.legend()
plt.show()