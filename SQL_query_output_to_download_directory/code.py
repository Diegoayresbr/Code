import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyodbc


def run_sql():
    try:
        server_name = ""
        dbname = ""
        user_SQL = ""
        password_SQL = ""

        connection = pyodbc.connect(
            "Driver={SQL Server Native Client 11.0};"
            "Server=" + str(server_name) + ";"  # server_name
            "Database=" + str(dbname) + ";"  # db name
            # "uid=User;" + str(user_SQL) + ";" # User
            # "pwd=password" + str(password_SQL) + ";" # password
            "Trusted_Connection=yes;"
        )

        query = """
                    Select * 
                    from 
                 """

        cursor = connection.cursor()  # cursor
        cursor.execute(query)  # without parameters

        # add Data to pd.DataFrame
        sql_data = pd.DataFrame(
            [tuple(row) for row in cursor.fetchall()],
            columns=[desc[0] for desc in cursor.description],
        )

        print("\nQuery is completed")
        return sql_data

    except NameError:
        print("SQL Connection not completed")

    except (Exception, pyodbc.Error) as error:
        print("Error connecting to SQL Database: \n {}".format(error))

    finally:
        if connection:
            connection.close()  # close connection


def saving_df(df):
    file_path = str(os.path.join(Path.home(), "Downloads"))  # get Download folder
    currentDate = datetime.now().strftime("%Y-%m-%d__%Hh-%Mm")
    file_path_csv = file_path + "\Query_output" + "__" + currentDate + ".csv"

    df.to_csv(file_path_csv, index=False)
    print("\nFile is saved in: \n" + file_path_csv)


df = run_sql()

saving_df(df)
