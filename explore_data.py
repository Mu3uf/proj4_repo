import pandas as pd


df = pd.read_csv('data/Tweets.csv')


print(df.shape)
print(df.columns)
print(df.head())


df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'sentiment']

# توزيع الفئات
print(df['sentiment'].value_counts())


print(df.isnull().sum())

df.to_csv('data/clean_raw.csv', index=False)