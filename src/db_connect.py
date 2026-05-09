import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://sepsis_user:sepsis123@localhost:5432/mimic4')

with engine.connect() as conn:
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'mimiciv_hosp' LIMIT 10;"))
    tables = [row[0] for row in result]

print("Connected successfully!")
print("Tables found:", tables)