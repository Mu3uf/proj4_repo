from sqlalchemy import create_engine, MetaData


DATABASE_URL = "sqlite:///sentiment.db"


engine = create_engine(
    DATABASE_URL,
    echo=True
)


metadata = MetaData()