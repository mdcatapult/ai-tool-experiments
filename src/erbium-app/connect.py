import psycopg2
from config.psql_config import load_config


def connect(config):
    """Connect to the PostgreSQL database server"""
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            print("Connected to the PostgreSQL server.")

            cursor = conn.cursor()
            query = "SELECT * FROM coshh.chemical;"
            cursor.execute(query)
            # Fetch all rows
            rows = cursor.fetchall()

            # Iterate over the rows and print the results
            for row in rows:
                print(row)

        cursor.close()
        conn.close()
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)


if __name__ == "__main__":
    config = load_config()
    connect(config)
