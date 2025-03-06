import sys
import psycopg2

def execute_sql_file(file_path):
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname='dream11', 
        user='postgres', 
        password='Ak@123', 
        host='db'
    )
    cur = conn.cursor()

    # Open and read the SQL file
    with open(file_path, 'r') as file:
        sql = file.read()

    # Execute the SQL
    cur.execute(sql)
    conn.commit()

    # Close the connection
    cur.close()
    conn.close()

if __name__ == '__main__':
    sql_file = sys.argv[1]
    execute_sql_file(sql_file)