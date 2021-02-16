import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def ADF_DickyFuller_test(df_data):
    print("--" * 40)
    if not isinstance(df_data, pd.DataFrame):
        df_data = df_data.to_frame()

    for i in range(0, len(df_data.columns)):
        series_data = df_data.iloc[:, i]
        print("ADF Check Stationarity: ", str(series_data.name), "\n")

        test_adf = adfuller(series_data, autolag="AIC")

        results_df = pd.Series(
            test_adf[0:4],
            index=[
                "test statistic",
                "p-value",
                "lags_used",
                "numbers of obeservations",
            ],
        )

        for key, value in test_adf[4].items():
            results_df["Critical value (%s)" % key] = value
        print(results_df)

        if test_adf[1] < 0.005:
            print(
                "{} (P-value {:5f}): \033[1;30;46mStationary\033[m".format(
                    series_data.name, test_adf[1]
                )
            )
        else:
            print(
                "{} (P-value {:5f}): \033[1;30;41mNOT Stationary\033[m".format(
                    series_data.name, test_adf[1]
                )
            )
        print("\n")

    return


def cointegration_test_ADF_Resids_methodology(df_data1):
    print("--" * 40, "Cointegration Test")
    df_data = df_data1.copy()

    # ADF of Resids methodology
    print("\nADF of Resids methodology")

    comb_list = list(itertools.combinations(df_data.columns, 2))

    for x in comb_list:
        series_1 = df_data.loc[:, str(x[0])]
        series_2 = df_data.loc[:, str(x[1])]

        # OLS regression between variable > test if residual is stationary
        model = sm.OLS(series_1, series_2)
        model_fit = model.fit()
        residual_x = model_fit.resid

        test_adf = adfuller(residual_x, maxlag=None, autolag="AIC")
        print("Results ADF of Resids from OLS Regression : ", str(x))
        if test_adf[1] < 0.05:
            print(
                "P-value {:5f}: \033[1;30;46mStationary. Therefore: cointegrated\033[m".format(
                    test_adf[1]
                )
            )
        else:
            print(
                "P-value {:5f}: \033[1;30;41mNOT Stationary. Therefore: NOT cointegrated\033[m".format(
                    test_adf[1]
                )
            )

        # plot OLS Resids
        mean_z = np.mean(residual_x)
        std_z = np.std(residual_x)
        fig_2, (ax6) = plt.subplots(
            nrows=1, ncols=1, figsize=(9, 5), dpi=80, sharex=False
        )
        fig_2.subplots_adjust(hspace=0.2, wspace=0.2)
        residual_x.plot(label="Residual", ax=ax6)
        ax6.axhline(
            y=mean_z, linestyle="--", color="gray", alpha=0.4
        )  # create the central line
        ax6.axhline(y=mean_z + (std_z) * 2, linestyle="--", color="gray", alpha=0.8)
        ax6.axhline(y=mean_z - (std_z) * 2, linestyle="--", color="gray", alpha=0.8)
        ax6.legend().set_draggable(True)
        ax6.set_title("Resids from OLS regression Y = Bx + C")
        # ax6.grid(True, linestyle='--', color='c', alpha=0.4)  # to add grind
        fig_2.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    return


def johansen_cointegration_test(df, alpha=0.05):
    print("--" * 40)
    print("Johansen Cointegration Test")

    # Johanson's Cointegration
    johansen_test = coint_johansen(
        df,
        det_order=-1,  # 1 - linear trend
        k_ar_diff=4,  # Number of lagged differences in the model
    )

    d = {"0.90": 0, "0.95": 1, "0.99": 2}  # dictionary with critical values
    cvts = johansen_test.trace_stat_crit_vals[:, d[str(1 - alpha)]]
    traces = johansen_test.trace_stat

    # Summary
    print(
        "Name   ::  Test Stat > C({:.0%})    =>   Signif  \n".format(1 - alpha),
        "--" * 20,
    )
    count_vector = len(df.columns)
    i = 0
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(
            "I(" + str(count_vector - i) + ") :: ",
            round(trace, 2),
            "--" * 2,
            cvt,
            "--" * 2,
            "Reject Ho: ",
            trace > cvt,
        )
        i = i + 1
    if not trace > cvt:
        print("At least 1 cointegration vector")
    else:
        print("NO cointegration vector")

    return


class drag_workaround(object):
    def __init__(self, artists):
        self.artists = artists
        artists[0].figure.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        for artist in self.artists:
            artist.pick(event)


def pairs_trading(df_w):
    print("--" * 40, "\nPairs trade")
    df_data1 = df_w.pct_change().dropna()
    df_diff = df_data1.iloc[:, 0] - df_data1.iloc[:, 1]
    # print(df_diff)
    df_data2 = df_w.pct_change().cumsum().dropna()

    fig_w, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(11, 8), dpi=80, sharex=False, sharey=False
    )
    fig_w.subplots_adjust(hspace=0.2, wspace=0.2)
    df_data2.plot(kind="line", ax=ax1, label="pct_accumulate")
    ax1.set_title("Percentage Accumulated")
    leg1 = ax1.legend(
        loc="best", fontsize="x-small", title=None, bbox_to_anchor=None
    ).set_draggable(
        True
    )  #

    # Plotting the rolling AVG
    rol_avg = df_data2.rolling(window=5).mean()
    ax3 = ax1.twinx()
    rol_avg = rol_avg.rename(
        columns=lambda x: x + " rolling AVG"
    )  # great to adjust column names
    rol_avg.plot(ax=ax3, linestyle="--", alpha=0.9, linewidth=0.9)

    leg1 = ax3.legend(
        loc="lower right", fontsize="x-small", title=None, bbox_to_anchor=None
    )

    df_diff.plot(
        kind="line",
        ax=ax2,
        label="pct_change_diff (Series_1 minus Series_2)",
        color="k",
    )
    ax2.set_title("Difference Analysis")
    leg2 = ax2.legend(
        loc="upper left", fontsize="x-small", title=None, bbox_to_anchor=None
    )

    ax2.axhline(y=0, linestyle="--", color="gray", alpha=0.4)  # create the central line
    ax2.axhline(
        y=np.std(df_diff) * 2, linestyle="--", color="gray"
    )  # upper confidence level
    ax2.axhline(
        y=-np.std(df_diff) * 2, linestyle="--", color="gray"
    )  # lower confidence level
    ax2.tick_params(axis="x", rotation=90, labelsize=7, pad=8)
    an1 = ax2.annotate(
        "High " + df_data2.columns[0] + " _and_ Low " + df_data2.columns[1],
        xy=(0.4, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
        clip_on=True,
        bbox=dict(boxstyle="round,pad=0.4", fc="g", ec="b", alpha=0.7),
    )

    an2 = ax2.annotate(
        "High " + df_data2.columns[1] + " _and_ Low " + df_data2.columns[0],
        xy=(0.4, 0.08),
        xycoords="axes fraction",
        ha="left",
        va="top",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
        clip_on=True,
        bbox=dict(boxstyle="round,pad=0.4", fc="g", ec="b", alpha=0.7),
    )
    an1.draggable()
    an2.draggable()
    leg1.set_draggable(True)
    leg2.set_draggable(True)
    drag_workaround([an1, an2])

    plt.grid(True, linestyle="--", color="c", alpha=0.4)
    fig_w.autofmt_xdate()
    plt.tight_layout(pad=3.0)
    # plt.savefig("plot1.png")
    plt.show()

    return


df_x = pd.read_csv("Series_pribor_discount_rate_CZ.csv", parse_dates=["date"])
df_x.set_index("date", inplace=True)


ADF_DickyFuller_test(df_x)  # function to check stationarity

johansen_cointegration_test(df_x)  # johansen to check cointegration

cointegration_test_ADF_Resids_methodology(df_x)  # function to check cointegration

pairs_trading(df_x)  # Plot visualization
