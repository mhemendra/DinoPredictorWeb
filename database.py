import mysql.connector

def get_conn():
    config = {
        'user': 'root',
        'password': 'Password123',
        'host': '34.93.126.116',
        'database': 'dino_db'
    }

    # now we establish our connection
    conn = mysql.connector.connect(**config)
    return conn

def create_db(conn):
    cursor = conn.cursor()  # initialize connection cursor
    cursor.execute('CREATE DATABASE dino_db')  # create a new 'testdb' database
    conn.close()  # close connection because we will be reconnecting to testdb

def create_table(conn):
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE dino_names ("
               "Id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,"
               "dino_name VARCHAR(255))")
    conn.commit()

def insert_into_table(conn, dino_names):
    query = (
        "INSERT INTO dino_names (dino_name) VALUES (%s)")
    cursor = conn.cursor()
    for dino_name in dino_names:
        cursor.execute(query, [dino_name])
    conn.commit()

def select_data(conn):
    cursor = conn.cursor()
    cursor.execute("select * from dino_names")
    out = cursor.fetchall()
    print(len(out)/3)
    for row in out:
        print(row)
    print("Total Generated Names:",int(len(out)/3))

def delete_data(conn):
    cursor = conn.cursor()
    cursor.execute("delete from dino_names")
    conn.commit()

if __name__ == '__main__':
    conn = get_conn()
    #delete_data(conn)
    select_data(conn)
