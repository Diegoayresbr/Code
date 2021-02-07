import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from fbprophet import Prophet


def yfinance_data(company_id):

    tickers = yf.Ticker(company_id)

    series_data_historical = tickers.history(period="max")
    df_1 = pd.DataFrame(series_data_historical)

    df_1["company_id"] = company_id
    df_1["series_type"] = "historical_data"

    df_1 = df_1.sort_index(ascending=True)

    historical_price = df_1["Close"]

    return historical_price


def prophet_model(df, holidays=True):

    model_x = Prophet()
    if holidays == True:
        model_x.add_country_holidays(country_name="US")

    model_x.fit(df)

    future_dates = model_x.make_future_dataframe(periods=365)
    prediction = model_x.predict(future_dates)
    print("predictions DF:")
    print(prediction.head().to_string())

    model_x.plot(prediction)  # data point as scatter; rolling avg; confidence interval
    plt.title(f"{company_id}: Future price prediction")
    plt.tight_layout()
    plt.show()

    model_x.plot_components(prediction)
    plt.show()


company_id = "GOOGL"
series = yfinance_data(company_id)

df = series.to_frame()
df.reset_index(drop=False, inplace=True)
df.columns = ["ds", "y"]


prophet_model(df, holidays=False)
