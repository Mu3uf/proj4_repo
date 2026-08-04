import pandas as pd

# قراءة الداتا
df = pd.read_csv('data/Tweets.csv')

# نظرة عامة
print(df.shape)
print(df.columns)
print(df.head())

# الأعمدة اللي بتهمنا هي: text و airline_sentiment
df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'sentiment']

# توزيع الفئات
print(df['sentiment'].value_counts())

# فحص القيم الناقصة
print(df.isnull().sum())

df.to_csv('data/clean_raw.csv', index=False)