import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm


def normality_test(test_data):

    test_data = np.array(test_data)
    test_data = np.ravel(test_data)

    import scipy.stats as stats

    asymmetry = stats.skew(test_data)
    print("Skewness: {0:.4f}".format(asymmetry))

    kurtosis = stats.kurtosis(test_data)
    print("Kurtosis: {0:.4f}".format(kurtosis))

    # D’Agostino, R. and Pearson
    stat, p1 = stats.normaltest(test_data)
    print("\nD’Agostino, R. and Pearson Statistics= %.3f, p=%.3f" % (stat, p1))
    alpha = 0.05
    if p1 < alpha:  # (reject null hypothesis)
        print("\033[1;30;41mNot Normal distribution\33[m")
    else:
        print("\033[1;30;46mNormal distribution\33[m")

    # Shapiro - Wilk test
    W, p2 = stats.shapiro(test_data)
    print("\nShapiro-Wilk: W:{0:.4f}    P={1:.4f}".format(W, p2))
    alpha = 0.05
    if p2 < alpha:  # (reject null hypothesis)
        print("\033[1;30;41mNot Normal distribution\33[m")
    else:
        print("\033[1;30;46mNormal distribution\33[m")

    sns.set()  # give a better layout

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(13, 6), dpi=80, sharex=False
    )
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    sns.histplot(
        test_data,
        bins=40,
        stat="density",
        color="c",
        alpha=0.6,
        label="histogram",
        ax=ax1,
    )

    sns.kdeplot(
        test_data, color="darkblue", label="KDE - kernel density estimation", ax=ax1
    )

    import scipy.stats as stats

    avg = test_data.mean()
    sigma = math.sqrt(test_data.var())
    x = np.linspace(avg - 3 * sigma, avg + 3 * sigma, 100)
    ax1.plot(
        x,
        stats.norm.pdf(x, avg, sigma),
        color="yellow",
        alpha=0.8,
        linewidth=2,
        label="normal distribution",
    )
    ax1.legend(
        loc="best", fontsize="x-small", title=None, bbox_to_anchor=None
    ).set_draggable(True)

    # Theoretical Quantiles
    import statsmodels.api as sm

    sm.qqplot(test_data, line="s", fit=True, ax=ax2)

    ax2.set_title("(QQ) Quantile-Quantile Plot")

    plt.tight_layout()
    plt.show()

    return


mu, sigma = 30, 1
normal_dist = s = np.random.normal(mu, sigma, 1000)
normality_test(normal_dist)


df = pd.read_csv("test_data.csv", delimiter=";")
normality_test(df)
