import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import colors


def heatmap_correlation(df):

    df.reset_index(drop=True, inplace=True) 
    correlation_x = df.corr(method='pearson')  # pearson correlation for columns in df
    print(correlation_x.to_string())

    # create personalized color Maps – Good for replacing those 'OrRd'
    color_list = ['#FF0000', '#FFFFFF', '#FF0000']
    cmap_personalized = colors.LinearSegmentedColormap.from_list('mycmap', color_list, gamma=1.0)  #

    # creating a correlation heatmap with seaborn
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=120, figsize=(10, 6))
    sb.heatmap(correlation_x
               , annot=True # , fmt='.2%'
               , cmap=cmap_personalized
               , center = 0, vmax = 1, vmin = -1
               , ax=ax
               )

    # Vertical alignment of the y axis
    plt.setp(ax.yaxis.get_majorticklabels() , va="center", size=8)
    plt.setp(ax.xaxis.get_majorticklabels() , size=8)

    # Set the tickers from charts in the center
    for label in ax.xaxis.get_major_ticks():
        label.label1.set_horizontalalignment('center')


    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    return


df = pd.read_csv("real_state_data.csv")

heatmap_correlation(df)

