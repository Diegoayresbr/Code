import pandas as pd
from sqlalchemy import create_engine
import pyodbc
import urllib
import time


def get_data(file, headers = []):
    if len(headers) == 0:
        df = pd.read_csv(file
                         , delimiter=","
                         , low_memory=False
                         )
    else:
        df = pd.read_csv(file, header = None)
        df.columns = headers
    for r in range(len(df.columns)):
        try:
            df.rename( columns={'Unnamed: {0}'.format(r):'Unnamed{0}'.format(r)},    inplace=True )
        except:
            pass
    return df


def SQL_connection_engine():
    server_name = "serve_name"
    db_name = "df_name"
    user_SQL = ""
    password_SQL = ""

    # CREATE Engine
    try:
        params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                         "SERVER=" + str(server_name) + ";"
                                        "DATABASE=" + str(db_name) + ";"
                                         # "UID=;"
                                         # "PWD="
                                        "Trusted_Connection=yes;" # needs it when you do not provide the user and password
                                         )
        engine_sql = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
        print("\nEngine Completed in: {}".format(str(db_name)))

    except (Exception, pyodbc.Error) as error:
        print("Error while creating Engine: \n {}".format(error))


    finally:
        return engine_sql, db_name


def load_df_sql(df, engine_sql, table_name):

    try:
        df.to_sql(table_name
                  , schema='dbo'
                  , con = engine_sql
                  , if_exists='append' # 'replace'
                  , index=False
                  )
        print("\nDataFrame Adjusted == True")

    except (Exception, pyodbc.Error) as error:
        print("\nDataFrame Adjusted == False")
        print(f"Error: '{error}'")
    return


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


start_time = time.time()

file_path = r'C:\......csv'
df = get_data(file_path)

engine_sql, db_name = SQL_connection_engine()

table_name = 'new_table'
load_df_sql(df, engine_sql, table_name) 


end_time = time.time()
duration = end_time - start_time
duration = convert(duration)

print('\nThe running time was : \n' + str(duration))

