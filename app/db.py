from sqlalchemy import create_engine


DATABASE_URL = "postgresql://postgres:123456@localhost:5432/sentiment_db"


engine = create_engine(DATABASE_URL, echo=False)