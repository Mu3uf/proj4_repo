from database import engine, metadata
import models


metadata.create_all(engine)


print("Database created successfully!")