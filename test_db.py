import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="parola",
    database="mentions_test",
    charset="utf8mb4",
)

cursor = conn.cursor()

cursor.execute("SELECT * FROM soundiiz LIMIT 10")

for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
