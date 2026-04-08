import pandas as pd

df = pd.read_csv("fed_speeches.csv")

print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df['text'].iloc[0][:1000])
