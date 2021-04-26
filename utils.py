## Load standard libraries
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from scipy import stats

# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# from IPython.core.display import display, HTML

# plt.rcParams["figure.figsize"] = (12,8)


# ## Load auxiliar packages
# try:
    # import sidetable
# except:
    # %pip install sidetable
    # import sidetable

# importar pgs da wikipedia
# try:
#   import wikipedia as wp
# except:
#   !pip install wikipedia
#   import wikipedia as wp
# wp.set_lang("pt")
    
# Dashboards
# Static figures to show on GitHub
# !pip install kaleido==0.0.1
# !pip install psutil==5.7.2
# # !pip install plotly==4.9.0
# !pip install -U plotly
# import plotly.express as px

# from IPython.core.display import display, HTML


# https://stackoverflow.com/questions/7261936/convert-an-excel-or-spreadsheet-column-letter-to-its-number-in-pythonic-fashion
excel_col_name = lambda n: '' if n <= 0 else excel_col_name((n - 1) // 26) + chr((n - 1) % 26 + ord('A'))
excel_col_num = lambda a: 0 if a == '' else 1 + ord(a[-1]) - ord('A') + 26 * excel_col_num(a[:-1])


def minimum_example_df():
    """ generate minimum example
    
    src: https://stackoverflow.com/questions/20109391/how-to-make-good-reproducible-pandas-examples
    
    
    # ex:
    df = minimum_example_df()
    
    # other ideas
    # stocks = pd.DataFrame({ 
#     'ticker':np.repeat( ['aapl','goog','yhoo','msft'], 50 ),
#     'date':np.tile( pd.date_range('1/1/2011', periods=50, freq='D'), 4 ),
#     'price':(np.random.randn(200).cumsum() + 10) })
    """ 
    import numpy as np
    import pandas as pd

    np.random.seed(123)

    df = pd.DataFrame({ 

        # some ways to create random data
        'a':np.random.randn(6),
        'b':np.random.choice( [5,7,np.nan], 6),
        'c':np.random.choice( ['panda','python','shark'], 6),

        # some ways to create systematic groups for indexing or groupby
        # this is similar to r's expand.grid(), see note 2 below
        'd':np.repeat( range(3), 2 ),
        'e':np.tile(   range(2), 3 ),

        # a date range and set of random dates
        'f':pd.date_range('1/1/2011', periods=6, freq='D'),
        'g':np.random.choice( pd.date_range('1/1/2011', periods=365, 
                              freq='D'), 6, replace=False) 
        })
    return df

def move_legend(ax, new_loc, **kws):
    """
    moves a sns legend
    
    # examples:
        # move_legend(ax, "upper left")
    
    # out of the graph area
        move_legend(ax, "upper left", bbox_to_anchor=(1.04,1))
    
    # src: https://github.com/mwaskom/seaborn/issues/2280
    """
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()   
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
    pass

def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
        
    [Thanks Anton Golubev](https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side)
    
    # Minimum example
        import pandas as pd
        
        
        col = ['A','B','C','D']
        df1 = pd.DataFrame(np.arange(12).reshape((3,4)),columns=col)
        df2 = pd.DataFrame(np.arange(16).reshape((4,4)),columns=col)
        display_side_by_side([df1, df2], ['DF1', 'DF2'])
    
    # output: 
    DF1                      DF2
        A	B	C	D            A	B	C	D
    0	0	1	2	3        0	0	1	2	3
    1	4	5	6	7        1	4	5	6	7
    2	8	9	10	11       2	8	9	10	11
                             3	12	13	14	15  
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'") \
                    .set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))
    pass


def summarize_data(df, max_categories=10):
    """ Summarize tables (repetition patterns)
    
    # Minimum example
        import pandas as pd
        
        
        df = pd.DataFrame([[1, 2], [1, 3], [4, 6]], columns=['A', 'B'])
        df_list, capt_list = summarize_data(df, max_categories=2)
        df_list, capt_list
    # output:
    ([      A  count    percent  cumulative_count  cumulative_percent
        0   1      2  66.666667                 2           66.666667
        1   4      1  33.333333                 3          100.000000,
            B_Repetitions  count  percent  cumulative_count  cumulative_percent
        0              1      3    100.0                 3               100.0],
    ['A', 'B_Repetitions'])
    """  
    try:
        import sidetable
    except:
        # %pip install sidetable
        # !pip install sidetable
        import sidetable
   
    
    df_list=[]
    capt_list=[]
    
    for col in list(df.columns):
        df_freq = df.stb.freq([col])
        if len(df_freq) > max_categories:
            col_name = col + '_Repetitions'
            df_freq.rename(columns={'count':col_name}, inplace=True)
            # display(df_freq.stb.freq([col_name], style=True))
            capt_list.append(col_name)
            df_list.append(df_freq.stb.freq([col_name]))
        else:
            # display(df.stb.freq([col], style=True))
            capt_list.append(col)
            df_list.append(df_freq)
    return df_list, capt_list


def plot_data(df, max_categories=10, plot_col=2, size=6):
    """
        Plot histograms and bar charts
        if unique_values > max_categories:
            histograms with Repetitions
        else:
            barplots with Ocurrences

    Args:
        df (pandas Dataframe): data in a tab form
        max_categories (int, optional): threshold to barplot. Defaults to 10.
        plot_col (int, optional): number of figures per row. Defaults to 2.
        size (int, optional): dimension of the figure. Defaults to 6.
    
    # Example
        df = minimum_example_df()
        plot_data(df, max_categories=10, plot_col=3, size=6)
    """   
    import matplotlib.pyplot as plt


    plot_row = df.shape[1] // plot_col# + df.shape[1] % plot_col    
    if df.shape[1] % plot_col != 0:
        plot_row += 1 
  
    fig, axes = plt.subplots(plot_row, plot_col,
                             figsize=(size*plot_col,size*plot_row/2))

    count = 0
    for col in list(df.columns):
        ax_row = count // plot_col
        ax_col = count % plot_col
        
        try:
            ax = axes[ax_row, ax_col]
        except:  # single row
            ax = axes[ax_col]
        
        df_freq = df.stb.freq([col])
        if len(df_freq) > max_categories:
            col_name = col + '_Repetitions'
            df_freq.rename(columns={'count':col_name}, inplace=True)
            
            if len(df_freq[col_name].unique()) > max_categories:
                df_freq.hist(
                    column=col_name
                    , ax=ax
                    , sharex=False
                    , sharey=False
                )
            else:
                df_freq[col_name].value_counts().plot.bar(
                    ax=ax
                    , rot=0
                )
            ax.set_title(col_name)
        else:
            df[col].value_counts().plot.bar(
                ax=ax
                , rot=90
            )
            ax.set_title(col)
        count += 1
    fig.tight_layout()
    pass


def func(pct, total):
    """
        Format label to pie chart: pct% (occurence/total)
    """
    ocurrence = int(pct/100*total)
    return "{:.1f}%\n({:d}/{:d})".format(pct, ocurrence, total)


def plot_pie_data(df, max_categories=5, plot_col=2, size=6):
    """ Plot pie charts
        
        if unique_values > max_categories:
            pie chart
        else:
            pass

    Args:
        df (pandas Dataframe): data in a tab form
        max_categories (int, optional): threshold to pie plot. Defaults to 5.
        plot_col (int, optional): number of figures per row. Defaults to 2.
        size (int, optional): dimension of the figure. Defaults to 6.
        
    # Example:       
        df = minimum_example_df()
        plot_pie_data(df, max_categories=5, plot_col=2, size=6)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    nr_plots = 0
    list_col = list(df.columns)   
    mask = np.array([False]*len(list_col))
    count = 0
    for col in list_col:
        if len(list(df[col].unique())) < max_categories:
            nr_plots += 1
            mask[count] = True
        count += 1

    pie_list = [i for (i, v) in zip(list_col, mask) if v]

    plot_row = nr_plots // plot_col + nr_plots % plot_col
  
    fig, axes = plt.subplots(plot_row, plot_col,
                             figsize=(size*plot_col,size*plot_row/2))
    count = 0    
    for col in pie_list:
        ax_row = count // plot_col
        ax_col = count % plot_col
        
        ax = axes[ax_row, ax_col]
        
        wedges, texts, autotexts = ax.pie(
            df[col].value_counts()
            , autopct=lambda pct: func(pct, len(df))
            , textprops=dict(color="w")
        )

        ax.set_title(col)
        ax.legend(wedges
            , list(df[col].unique())
            #, title=col
            , loc="center left"
            , bbox_to_anchor=(1, 0, 0.5, 1)
        )
        plt.setp(
            autotexts
            , size=10
            , weight="bold"
        )
        
        count += 1
    fig.tight_layout()
    pass


def compare_two_variables(col1, col2, df):
    """
    col1, col2: strings with columns names to be compared
    df: dataframe

    Method: 
      compute value_counts distribution

    out:
      histogram(col1)
      plot(col1, col2)
      dataframe

    without return
    
    Example:
        df = minimum_example_df()
        compare_two_variables(col1='c', col2='d', df=df)
    Out:
        index_x	c	index_y	d
    0	python	2	    2	2
    1	shark	2	    1	2
    2	panda	2	    0	2

    """
    fig, (ax1,ax2) = plt.subplots(1, 2)
    
    # Correlation plot
    ax1.plot(
        df[col1].value_counts()
        , df[col2].value_counts()
        );
    
    # Histogram   
    df[col1].value_counts().hist(ax=ax2);
    
    fig.set_size_inches(12,4)
    
    # Compare value counts
    df1 = df[col1].value_counts().reset_index()
    df2 = df[col2].value_counts().reset_index()
    display(pd.merge(df1, df2, left_index=True, right_index=True))
    pass


def plot_dist_qq_box(df, col, fit_legend='normal_fit'):
    """
        
    Example 1:
        import numpy as np
    
        
        plot_dist_qq_box(np.arange(100), )
    
    
    Example 2:
        import pandas as pd
        
        
        df = pd.DataFrame([[1, 2], [1, 3], [4, 6]], columns=['A', 'B'])
        plot_dist_qq_box(df, col='A')
    
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.core.pylabtools import figsize
    from statsmodels.graphics.gofplots import qqplot
    figsize(12, 8)
    #TODO: deprecated distplot
    variable_to_plot = df[col]
    sns.set()
    fig, axes = plt.subplots(2, 2)
    l1 = sns.distplot(
        variable_to_plot
        , fit=stats.norm
        , kde=False
        , ax=axes[0,0]
        )
    # l1 = sns.histplot(
    #     variable_to_plot
    #     #, fit=sct.norm
    #     , kde=True
    #     , ax=axes[0,0]
    #     )
    l2 = sns.boxplot(
        variable_to_plot
        , orient='v'
        , ax=axes[0,1]
        )
    l3 = qqplot(
        variable_to_plot
        , line='s'
        , ax=axes[1,0]
        )
    l4 = sns.distplot(
        variable_to_plot
        , fit=stats.norm
        , hist=False
        , kde_kws={"shade": True}
        , ax=axes[1,1]
        )
    # l4 = sns.kdeplot(
    #     variable_to_plot
    #     #, fit=sct.norm
    #     #, hist=False
    #     #, kde_kws={"shade": True}
    #     , ax=axes[1,1]
    #     )
    axes[0,0].legend((fit_legend,'distribution'))
    axes[1,0].legend(('distribution',fit_legend))
    axes[1,1].legend((fit_legend,'kde_gaussian'))

    xlabel = col

    axes[0,0].set_xlabel(xlabel)
    axes[0,1].set_xlabel(xlabel)
    axes[1,1].set_xlabel(xlabel)
        
    plt.show()
    pass




def plot_dist_qq_box1(df, feature_name):
    """
        input: df (pd.DataFrame); feature_name (string)
        output: 3 plots
        1) distribution normal plot (Gaussian Kernel)
        2) qqplot (imput nan with mean)
        3) boxplot
    
    Thanks to [Ertuğrul Demir](https://www.kaggle.com/datafan07/titanic-eda-and-several-modelling-approaches)
    
    Example:
        import pandas as pd
        
        
        df = pd.DataFrame([[1, 2], [1, 3], [4, 6]], columns=['A', 'B'])
        plot_dist_qq_box1(df, feature_name='A')       
    
    """

    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    
    # Creating a customized chart. and giving in figsize and everything.    
    fig = plt.figure(
        constrained_layout=True
        , figsize=(12, 8)
    )
    
    # Creating a grid of 3 cols and 3 rows.    
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    
    # Plot the histogram.
    sns.distplot(
        df.loc[:, feature_name]
        , hist=True
        , kde=True
        , fit=stats.norm
        , ax=ax1
        , color='#e74c3c'
    )
    ax1.legend(labels=['Normal', 'Actual'])

    # Customizing the QQ_plot.
    ax2 = fig.add_subplot(grid[1, :2])  
    ax2.set_title('Probability Plot')
    
    # Plotting the QQ_Plot.
    stats.probplot(
        df.loc[:, feature_name].fillna(np.mean(df.loc[:, feature_name]))
        , plot=ax2
    )
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    # Customizing the Box Plot.
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    
    # Plotting the box plot.
    sns.boxplot(
        df.loc[:, feature_name]
        , orient='v'
        , ax=ax3
        , color='#e74c3c'
    )
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{feature_name}', fontsize=24)
    pass





def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=False,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is False.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    
    
    # Thanks [DTrimarchi10](https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py)
    
    Example:
        from sklearn.metrics import confusion_matrix
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        cf = confusion_matrix(y_true, y_pred)
        make_confusion_matrix(cf)
    
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value)
            for value
            in cf.flatten()/np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3
        in zip(group_labels,group_counts,group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            tn, fp, fn, tp = cf.ravel()
            
            # precision = cf[1,1] / sum(cf[:,1])
            precision = tp/(tp+fp)
            # recall    = cf[1,1] / sum(cf[1,:])
            recall    = tp/(tp+fn)
            #f1_score  = 2*precision*recall/(precision+recall)
            f1_score  = 2*tp/(2*tp+fp+fn)
            mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nMCC={:0.3f}".format(
                accuracy, precision, recall, f1_score, mcc)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf
        , annot=box_labels
        , fmt=""
        , cmap=cmap
        , cbar=cbar
        , xticklabels=categories
        , yticklabels=categories
    )

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    pass


def convert_str2int(df, start_col=1):
    """
        given a df with string columns
        convert it to float columns
            trim spaces
            remove leading zeros
        from the start_col on
    """
    for col in df.columns[start_col:]:
        try:
            df.loc[:,col] = df[col].str.replace("\s","")  # trim spaces
            df.loc[:,col] = df[col].str.replace(".","")  # remove points
            df.loc[:,col] = df[col].str.lstrip('0')  # remove leading zeros
            # df[col] = df[col].astype(int)
            df.loc[:,col] = pd.to_numeric(df[col], errors='coerce')
            # df[col] = df[col].replace(np.nan, 0, regex=True)
            # df[col] = df[col].astype(int)
            # df[col] = df[col].replace(0, np.nan)
        except:
            pass
    return df


def check_equal_means(statistic, p_value, alpha=0.05):
    """Compare two means, print statistic

    Args:
        statistic (float]): [description]
        p_value (float): [description]
        alpha (float, optional): Significance Level. Defaults to 0.05.

    Returns:
        boolean: True if the two means seem equal else False
    """
    print('Statistics=%.3f, p_value=%.3f' % (statistic, p_value))
    if p_value <= alpha/2:
        means_seems_equal = False
        print('Sample means not look equal (reject H0)')
    else:
        means_seems_equal = True
        print('Sample means look equal (fail to reject H0)')
        
    return means_seems_equal


def check_normality(statistic, p_value, alpha=0.05):
    """ Statistical report if the variables informed seems Gaussian/Normal

    Args:
        statistic (float): [description]
        p_value (float): [description]
        alpha (float, optional): significance level. Defaults to 0.05.

    Returns:
        (boolean): True if Normal else False
    """
    print('Statistics=%.3f, p_value=%.3f' % (statistic, p_value))
    if p_value <= alpha:
        seems_normal = False
        print('Sample does not look Gaussian (reject H0)')
    else:
        seems_normal = True
        print('Sample looks Gaussian (fail to reject H0)')

    return seems_normal



def anderson_darling_normality_test(result):
    """ 
        Statistical report if the variables informed seems Gaussian/Normal
        accordingly to Anderson Darling Normality Test
        plot Significance Level x Critical Values

    Args:
        result (Scipy object): Object from Scipy

    Returns:
        boolean: True if Normal else False
    """
    import matplotlib.pyplot as plt
    
    
    print('Statistic: %.3f' % result.statistic)
    p = 0
    is_normal = True
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)'% (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)'% (sl, cv))
            is_normal = False
    plt.scatter(
        result.significance_level
        ,result.critical_values
        )
    plt.xlabel('Significance Level')
    plt.ylabel('Critical Values')
    plt.title("Anderson-Darling Normality Test")
    
    return is_normal


def print_check_normality_multiple_tests(data):
    """
        Reports 4 normality tests and summarizes in a dataframe
        1) Shapiro-Wilk
        2) Jarque-Bera
        3) D'Agostino-Pearson or D'Agostino K^2
        4) Anderson-Darling
    Args:
        data ([np.array or pd.Series]): [description]

    Returns:
        pd.DataFrame: Summarize results from tests
        
    
    Example:
        import numpy as np
        
        
        print_check_normality_multiple_tests(np.arange(100))
    
    Out:
        Shapiro-Wilk Normality Test
        Statistics=0.955, p_value=0.002
        Sample does not look Gaussian (reject H0)

        Jarque-Bera Normality Test
        Statistics=6.002, p_value=0.050
        Sample does not look Gaussian (reject H0)

        D'Agostino-Pearson Normality Test
        Statistics=34.674, p_value=0.000
        Sample does not look Gaussian (reject H0)
        Statistics=34.674, p_value=0.000
        Sample does not look Gaussian (reject H0)

        Anderson-Darling Normality Test
        Statistic: 1.084
        15.000: 0.555, data does not look normal (reject H0)
        10.000: 0.632, data does not look normal (reject H0)
        5.000: 0.759, data does not look normal (reject H0)
        2.500: 0.885, data does not look normal (reject H0)
        1.000: 1.053, data does not look normal (reject H0)
        Statistics=34.674, p_value=0.000
        Sample does not look Gaussian (reject H0)
    
            Method	            Is_Normal
        0	Shapiro-Wilk	    False
        1	Jarque-Bera	        False
        2	D'Agostino-Pearson	False
        3	Anderson-Darling	False
    
    """
    import pandas as pd
    from scipy import stats
    
    
    # Shapiro-Wilk
    print("Shapiro-Wilk Normality Test")
    statistic, p_value = stats.shapiro(data)
    is_normal_shapiro_wilk = check_normality(statistic, p_value)
    
    # Jarque-Bera
    print("\nJarque-Bera Normality Test")
    statistic, p_value = stats.jarque_bera(data)
    is_normal_jarque_bera = check_normality(statistic, p_value)
    
    # D'Agostino-Pearson or D'Agostino K^2
    # check skew: pushed left or right (asymmetry)
    # check kurtosis: how much is in the tail
    print("\nD'Agostino-Pearson Normality Test")
    statistic, p_value = stats.normaltest(data)
    check_normality(statistic, p_value)
    is_normal_dagostino_pearson = check_normality(statistic, p_value)
    
    # Anderson-Darling    
    print("\nAnderson-Darling Normality Test")
    result = stats.anderson(data, dist='norm')
    anderson_darling_normality_test(result)
    is_normal_anderson_darling = check_normality(statistic, p_value)
    
    is_normal = {"Method": ["Shapiro-Wilk",
                            "Jarque-Bera",
                            "D'Agostino-Pearson",
                            "Anderson-Darling"],
                 'Is_Normal': [is_normal_shapiro_wilk,
                               is_normal_jarque_bera,
                               is_normal_dagostino_pearson,
                               is_normal_anderson_darling]
                }
    
    return pd.DataFrame(data=is_normal)



def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
        
        
    Example:
        import pandas as pd
        
        df = pd.DataFrame({
            'numbers_1to100': np.arange(100)})
        get_sample(df=df, col_name='numbers_1to100', n=10, seed=40)
    Out:
        79    79
        75    75
        63    63
        15    15
        38    38
        11    11
        40    40
        45    45
        39    39
        62    62
        Name: numbers_1to100, dtype: int32
        
    """
    import numpy as np
    
    
    np.random.seed(seed)   
    random_idx = np.random.choice(
        df[col_name].dropna().index
        , size=n
        , replace=False
        )
    
    return df.loc[random_idx, col_name]


def plot_top(df, col, ax, n=10, normalize=True):
    """ Plot nlargest frequencies in a dataframe column

    Args:
        df (pandas Dataframe): data in a tab form
        col (str): name of the column to plot
        ax (axis object): axis to plot the figure
        n (int, optional): Number of results to plot. Defaults to 10.
        normalize (bool, optional): pct if True else absolute. Defaults to True.
        
    Ex:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        
        df = pd.DataFrame({
            'c':np.random.choice( [100,5,3], 100)
            })
        fig, ax = plt.subplots()
        plot_top(df=df, col='c', ax=ax, n=2, normalize=False)
    """

    df[col].value_counts(
        normalize=normalize
    ).nlargest(n).plot.barh(
        x=col
        , legend=False
        , ax=ax
        , title=f'Top {n} - {col}'
        )
    ax.invert_yaxis()
    if normalize:
        ax.axis(xmin=0, xmax=1)

    pass


def df_total_row_col(df,row=True,col=True):
    """ Add total row and col to a daftaframe
    """

    df_out = df.copy()
    df_out.loc['Column_Total']= df_out.sum(numeric_only=True, axis=0)
    df_out.loc[:,'Row_Total'] = df_out.sum(numeric_only=True, axis=1)

    return df_out



def cross_tab(df, col1, col2, total_row_col=True, plot=True, table=True):
    """ Compare two (categorical) variables by a stacked barplot and cross table

    Args:
        df (pandas Dataframe): data in a tab form
        col_1 (str): name of the first column to compare (x_axis)
        col_2 (str): name of the second column to compare (y_axis)

    Returns:
        (pandas Dataframe): cross table with data between col1 and col2
        
    Thanks [Piush Vaish](https://adataanalyst.com/data-analysis-resources/visualise-categorical-variables-in-python/)
    
    Ex:
        import pandas as pd
        
        
        df = pd.DataFrame({
            'ABC':np.random.choice( ['a','b','c'], 100),
            'DEF':np.random.choice( ['d','e','f'], 100)
            })
        df_cross = cross_tab(df, col1='ABC', col2='DEF')
     
    Out:
        DEF	d	e	f
        ABC			
        a	17	12	9
        b	8	18	12
        c	7	10	7
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    
    cross_col1_col2 = pd.crosstab(
        index=df[col1]
        , columns=df[col2]
        )
    
    if plot:
        cross_col1_col2.plot(
            kind="bar"
            , stacked=True
        )
        ax = plt.gca()
        move_legend(ax, "upper left", bbox_to_anchor=(1.04,1))
    # plt.show()
    
    cross_col1_col2 = df_total_row_col(cross_col1_col2,row=True,col=True)
    
    if table:
        display(cross_col1_col2)
    
    return cross_col1_col2



def sorted_box_plot(by_col, plot_col, df, hue=False):
    """
        #TODO: WRITE DOCSTRING
        
        #TODO: BUILD TEST
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # import numpy as np
        
        
        # tips = sns.load_dataset("tips")
        # df = pd.DataFrame({
        #     'ABC':np.random.choice( ['a','b','c'], 100),
        #     'DEF':np.random.choice( ['d','e','f'], 100)
        #     })
        # sorted_box_plot(by_col='ABC', plot_col='DEF', df=df, hue=False)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    plt.figure(figsize=(10, 8))
    lst_order = sorted(list(dct[by_col].keys()))
    if hue:
        ax = sns.boxplot(
            x=by_col
            , y=plot_col
            , data=df
            , order=lst_order
            , hue=hue
        )
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, dct[hue].values(), title=descr[hue])
    else:
        ax = sns.boxplot(
            x=by_col
            , y=plot_col
            , data=df
            , order=lst_order
        )

    ax.set_xticklabels([dct[by_col][k] for k in lst_order])
    plt.xticks(rotation='vertical')
    plt.title(descr[plot_col] + " por " + descr[by_col])
    plt.xlabel(descr[by_col])
    plt.ylabel(descr[plot_col])
    plt.show()
    pass





def plot_missing_data(df, lst, name, figsize):
    """ 
    Detecting missing data:
    yellow means no available data, black means we have data
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(f'{name} Missing Values')
    sns.heatmap(
        df[lst].T.isna()
        , cbar=False
        , cmap='magma'
    )
    xtl=[item.get_text()[:7] for item in ax.get_xticklabels()]
    _=ax.set_xticklabels(xtl)
    plt.xticks(rotation=90)
    plt.show()
    
    return df[lst].isna().sum().to_frame(name=name.lower())



def check_miss_data(df):
    import pandas as pd
    
    
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
  
    return missing_data

def get_initial_dates(df, index, data_started_lin=0):
    """
    Args:
        df: pd.DataFrame
        index: col with dates
        data_started_lin: int where data starts (raw data)
        
    Make list from repeated values
    https://stackoverflow.com/questions/31796973/pandas-dataframe-combining-one-columns-values-with-same-index-into-list
    """
    missing_data = check_miss_data(df)
    missing_data.index.name = "Name"
    initial_dates_agg = missing_data.reset_index().groupby(['Total'])['Name'].unique().to_frame()
    initial_dates_agg['date'] = df[index].iloc[initial_dates_agg.index].dt.strftime('%m/%Y')
    initial_dates_agg['initial_lin'] = initial_dates_agg.index+data_started_lin
    initial_dates_agg.set_index("date",inplace=True)
    
    return initial_dates_agg


def display_full(x):
    """
    Non-truncated pandas
    https://stackoverflow.com/questions/25351968/how-to-display-full-non-truncated-dataframe-information-in-html-when-convertin
    """
    import pandas as pd


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    display(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
    pass


def print_full(x):
    """
    Non-truncated pandas
    https://stackoverflow.com/questions/25351968/how-to-display-full-non-truncated-dataframe-information-in-html-when-convertin
    """
    import pandas as pd
    
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
    pass




def plot_heatmap_nr(corr_mat, figsize=(6, 6)):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns


    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr_mat
        , mask=mask
        , cmap='RdBu_r'
        , vmin=-1.0
        , vmax=1.0
        , center=0
        , square=True
        , linewidths=.5
        , cbar_kws={"shrink": .5}
        , annot=True
        , fmt=".2f"
        );
    pass


def drop_prevalent(df, threshold=0.01):
    """
        if 99% is one single answer drop
    """
    col_to_drop_prevalent = list()
    
    for col in df.columns:
        prevalent = df[col].value_counts(normalize=True).max()
        if 1-prevalent < threshold:
            col_to_drop_prevalent.append(col)
    return col_to_drop_prevalent



def prevalent_analysis(df):
    """
    
    
    #TODO Build Example:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        
        fig = px.scatter(df_prevalent, x='Threshold', y='Dropped columns amount')
        fig.show()

        arr_bool = np.empty(shape=(len(df_prevalent),len(df.columns)))
        ii = 0
        jj = 0
        for item in drop_prevalent_list:
            for col in df.columns:
                arr_bool[ii,jj] = col in item
                jj+=1
            ii+=1
            jj=0
        
        df2 = pd.DataFrame(arr_bool)
        df2.columns = df.columns
        abs(df2.sum()-101).sort_values() # Threshold to discard each column  %
        
    
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    drop_prevalent_len = list()
    drop_prevalent_list = list()
    arr_threshold = np.linspace(0,1,101)
    for threshold in arr_threshold:
        drop_it = drop_prevalent(df, threshold)
        drop_prevalent_list.append(drop_it)
        drop_prevalent_len.append(len(drop_it))

    fig, ax = plt.subplots()
    ax.plot(arr_threshold, drop_prevalent_len)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Colums to drop')
    plt.show()

    df_prevalent = pd.DataFrame([arr_threshold,drop_prevalent_len]).T
    df_prevalent.columns = ['Threshold', 'Dropped columns amount']

    pass
    
    
def label_encode_categorical_values(df, index, plot=True):
    """
        Args:
            df: pd.DataFrame
            index: string (col_name as data index)
        Return:
            df_encoded: label encoding
            dct_encoding: dict mapping
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    import seaborn as sns
    
    
    df_encoded = df.copy()

    lst_col_string = df.select_dtypes(
        include='object',  # 'number' # 'object' # 'datetime' #
        exclude=None
    ).columns.to_list()

    dct_encoding = {}
    for idx, col in enumerate(lst_col_string):
        dct = pd.DataFrame(
            {"name":df[col],
             "code":df[col].astype('category').cat.codes}
        ).drop_duplicates().set_index("name").to_dict()['code']
        # lst_dct_strings.append({col: dct})
        # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
        dct = dict(sorted(dct.items(), key=lambda item: item[1]))      
        dct_encoding[col] = dct
        df_encoded[col] = df_encoded[col].map(dct)
        
        if plot:          
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=df,
                x=index,
                y=col,
                hue=col,
                legend=False
            )
            # move_legend(ax, "upper left", bbox_to_anchor=(1.04,1))
            plt.show()
            
            sns.countplot(
                data=df,
                y=col
            )
            plt.show()
            
#             positions = tuple(dct.values())
#             labels = dct.keys()
#             df_encoded.plot(x=index,y=col)
#             plt.yticks(positions, labels)
#             plt.title(col)
#             plt.show()

#             df[col].value_counts().plot(kind='barh')
#             plt.title(col)
#             plt.show()
    
    return df_encoded, dct_encoding


def convert_0_1(arr, threshold=0.5):
    """
    Convert a probabilistic array into binary values (0,1)
        Args:
            arr: np.array with probabilities
            threshold (optional, default=0.5: probability limit)
                if element < threshold = 0
                if element >= threshold = 1
        Return:
            arr_0_1: np.array with 0 or 1 values
    """
    import numpy as np


    arr_0_1 = np.copy(arr)
    arr_0_1[arr<threshold] = 0
    arr_0_1[arr>=threshold] = 1
    
    arr_0_1 = np.array([np.int32(x.item()) for x in arr_0_1])
    
    return arr_0_1
    
    
def standardize_df(df):
    """
    Standardize a dataframe (mean centered at 0 and unitary standard deviation)
        Args:
            df: pd.DataFrame with only numeric values
        Return:
            df: pd.DataFrame standardized
    """
    return (df-df.mean())/df.std()
    

def normalize_df(df):
    """
    Normalize a dataframe (range[0,1])
        Args:
            df: pd.DataFrame with only numeric values
        Return:
            df: pd.DataFrame normalized
    """
    return (df-df.min())/(df.max()-df.min())



def eda_plot(df, index):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    # width, height
    fig_size_base = (6,6)
    total_fig = len(df.select_dtypes('number').columns)
    cols = 6

    rows = total_fig // cols + 1

    # check if you have more space as figures to plot
    assert rows*cols > total_fig
    
    fig_size_hist_line = (cols*fig_size_base[0], 
                         rows*fig_size_base[1])
    fig_scale_heatmap = 0.75
    fig_size_heatmap = (fig_scale_heatmap*fig_size_hist_line[0], 
                        fig_scale_heatmap*fig_size_hist_line[1])
    fig_scale_boxplot = 0.25
    fig_size_boxplot = (fig_scale_boxplot*fig_size_hist_line[0], 
                        fig_scale_boxplot*fig_size_hist_line[1])

    # Histogram
    fig, ax = plt.subplots(figsize=fig_size_hist_line)
    df.drop(columns=[index]).hist(ax=ax, layout=(rows,cols))
    plt.show()

    
    # Heatmap
    plot_heatmap_nr(df.corr(), figsize=fig_size_heatmap)
    
    # Line plot
    df.plot(
        x=index,
        subplots=True,
        sharex=False,
        layout=(rows,cols),
        figsize=fig_size_hist_line
    )
    plt.show()
    
    # Boxplot
    fig, ax = plt.subplots(figsize=fig_size_boxplot)
    # df_standardized.boxplot(vert=False, ax=ax)
    sns.boxplot(
        data=standardize_df(df.drop(columns=[index]))
        , ax=ax
        , orient='h'
    )

    # sns.stripplot(
    #     data=df_standardized
    #     , ax=ax
    #     , orient='h'
    #     , color=".2"
    # )
    plt.show()
    
    pass
    
    
def replace_str_nan_by_np_nan(df_str_nan):
    """
        dealing with nan strings, since fillna handles only np.nan
        
        Args: df with string nan
        
        Return: df with np.nan
    
    Ex: 
        import pandas as pd
        import numpy as np
                
        df_str_nan = pd.DataFrame({
            'age':['np.nan',34,19], 
            'gender':['Nan',np.nan,'M'], 
            'profession':['student', 'nan', 'artist']})
        df_np_nan = replace_str_nan_by_np_nan(df_str_nan)              
        print(df_np_nan.isna())
        	age	    gender	profession
        0	True	True	False
        1	False	True	True
        2	False	False	False
    """
    import numpy as np
    
    df_np_nan = df_str_nan.copy()
    for nan in ['np.nan', 'NaN', 'Nan', 'nan']:  
        df_np_nan = df_np_nan.replace(nan, np.nan, regex=True)
        
    return df_np_nan


def join_df1_df2_repeated_col(df1, df2):
    """
        join two dataframes keeping values within repeated columns 
        dealing with nan strings, since fillna handles only np.nan
        
        Args: df1, df2 two dataframes
        
        Return: df_join joined dataframe
    
    Ex: 
        import pandas as pd
        import numpy as np
        

        df1 = pd.DataFrame({
            'age':[7,34,19], 
            'gender':['F',np.nan,'M'], 
            'profession':['student', 'CEO', 'artist']})
        df2 = pd.DataFrame({
            'age':[7,34,19], 
            'gender':['np.nan','F',np.nan], 
            'interests':['acting', 'cars', 'gardening']})

        print(join_df1_df2_repeated_col(df1, df2))
        
            age	gender	profession	interests
        0	7	F	    student	    acting
        1	34	F	    CEO	        cars
        2	19	M	    artist	    gardening
    """
    import pandas as pd
    import numpy as np
    
    
    # dealing with nan strings, since fillna handles only np.nan
    df1 = replace_str_nan_by_np_nan(df1)
    df2 = replace_str_nan_by_np_nan(df2)
    
    # join and dealing with repeated columns
    rsuffix = "_r"
    df_join = df1.join(df2, rsuffix=rsuffix)   
    mask = df_join.columns.str.endswith(rsuffix)
    lst_col_r = list(df_join.loc[:,mask].columns)
    for col_r in lst_col_r:
        col = col_r[:-len(rsuffix)]
        df_join[col] = df_join[col].fillna(df_join[col_r])   
    
    return df_join.drop(columns=lst_col_r)



def drop_outliers(df, method='std', value=3):
    """
        Drop outliers within a dataframe
        
        Args: df: pd.DataFrame with numeric entries
        method: string ['std','iqr'], default:  'std'
            'std': check if the data are within a standard range
            as a z_score in an equivalent normal distribution
            'iqr': check if the data are within a interquartile range
            iqr = quantile(0.75) - quantile(0.25)
        value: float, default 3
            for 'std' method corresponds to a standard deviation factor 
            for 'iqr' method corresponds to a interquartile factor

        Return:
            df_no_outliers: pd.DataFrame without outliers regardless the columns

    src: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
    """
    import numpy as np

    
    if method == 'std':
        df_no_outliers = df[df
              .apply(lambda x: np.abs(x - x.mean()) / x.std() < 3)
              .all(axis=1)]
    elif method == 'iqr':
        df_no_outliers = df[df
              .apply(lambda x: x.between(
                  x.quantile(0.25)-value*(x.quantile(0.75)-x.quantile(0.25)),
                  x.quantile(0.75)+value*(x.quantile(0.75)-x.quantile(0.25))))
              .all(axis=1)]

    return df_no_outliers



def detect_outliers(df, method='std', value=3):
    """
        Detect outliers within a dataframe
        
        Args: df: pd.DataFrame with numeric entries
        method: string ['std','iqr'], default:  'std'
            'std': check if the data are within a standard range
            as a z_score in an equivalent normal distribution
            'iqr': check if the data are within a interquartile range
            iqr = quantile(0.75) - quantile(0.25)
        value: float, default 3
            for 'std' method corresponds to a standard deviation factor 
            for 'iqr' method corresponds to a interquartile factor

        Return:
            df_outliers: pd.DataFrame with the outliers regardless the columns
    """
    
    import numpy as np
    import pandas as pd

    if method == 'std':
        df_outliers = df[df
              .apply(lambda x: np.abs(x - x.mean()) / x.std() >= value)
              .any(axis=1)]
    elif method == 'iqr':
        # iqr = df.quantile(0.75) - df.quantile(0.25)
        # lim_inf = df.quantile(0.25) - 1.5*iqr
        # lim_sup = df.quantile(0.75) + 1.5*iqr
        df_outliers_inf = df[df
              .apply(lambda x: x <= x.quantile(0.25) 
                                - value * (x.quantile(0.75) - x.quantile(0.25)))
              .any(axis=1)]
        df_outliers_sup = df[df
              .apply(lambda x: x >= x.quantile(0.75) 
                                + value * (x.quantile(0.75) - x.quantile(0.25)))
              .any(axis=1)]
        df_outliers = pd.concat([df_outliers_inf, df_outliers_sup]).drop_duplicates()

    return df_outliers
    
    
def get_df_stats(df):
    
    df_stats = df.describe().T
    df_stats['IQR'] = df_stats['75%'] - df_stats['25%']
    # df_stats['lim_inf_1.5IQR'] = df_stats['25%'] - 1.5 * df_stats['IQR']
    df_stats['lim_inf_1.5IQR'] = df.quantile(0.25) - 1.5*(df.quantile(0.75) - df.quantile(0.25))
    df_stats['lim_inf_3std'] = df.mean()-3*df.std()
    # df_stats['lim_sup_1.5IQR'] = df_stats['75%'] + 1.5 * df_stats['IQR']
    df_stats['lim_sup_1.5IQR'] = df.quantile(0.75) + 1.5*(df.quantile(0.75) - df.quantile(0.25))
    df_stats['lim_sup_3std'] = df.mean()+3*df.std() 
    
    return df_stats.T



def get_tscv_index(y, test_idx_start, test_size=1):
    """
    Args: 
        y: numpy array to get number of elements
        test_idx_start: int where the first test start
        test_size: int number of elements in test (default=1)
    
    Return:
        lst_train_idx: lst with np.array with indexes
            for n_splits time series
            each one starting from 0 and finishing at (test_idx_start-1) + ii
            where ii ranges from 0 to n_splits
        lst_test_idx: lst with with np.array with indexes
            for all time series cross validate sequence
            from (train_idx_start+1)+ii to (train_idx_start+1)+ii + test_size
            up to the last element in y
            
    Minimum example:
        
        import numpy as np
        
        len_y = 7
        test_size = 2
        test_idx_start = 3
        
        X = np.random.randn(len_y, 2)
        y = np.random.randint(0, 2, len_y)
        lst_train_idx, lst_test_idx = get_tscv_index(y, test_idx_start, test_size)
        
        # test_idx_start_last = len_y-test_size
        n_splits = (len_y-test_size) - test_idx_start + 1

        assert len(lst_train_idx) == len(lst_test_idx)
        assert len(lst_train_idx) == n_splits
        
        print(f"n_splits: {n_splits}\n")
        
        for idx, train_index in enumerate(lst_train_idx):
            test_index = lst_test_idx[idx]
            print(f"Train: idx: {train_index}\t\t Test: idx: {test_index}")

        '''
        Out:
        
        n_splits: 3
        
        Train: idx: [0 1 2]		     Test: idx: [3 4]
        Train: idx: [0 1 2 3]		 Test: idx: [4 5]
        Train: idx: [0 1 2 3 4]		 Test: idx: [5 6]
        '''
    
    """
    
    import numpy as np
    
    n_splits = (len(y)-test_size) - test_idx_start + 1
    lst_train_idx = [np.arange(test_idx_start+ii) for ii in range(n_splits)]
    lst_test_idx = [(test_idx_start+ii)+np.arange(test_size) for ii in range(n_splits)]
    
    return lst_train_idx, lst_test_idx


def get_tscv_X_y(X, y, lst_train_idx, lst_test_idx):
    lst_X_train = [X[ii] for ii in lst_train_idx]
    lst_X_test = [X[ii] for ii in lst_test_idx]
    
    lst_y_train = [y[ii] for ii in lst_train_idx]
    lst_y_test = [y[ii] for ii in lst_test_idx]
    
    return lst_X_train, lst_X_test, lst_y_train, lst_y_test

    
def get_X_y(df, lst_col_X, target):
    X = df[lst_col_X].values
    y = df[target].values.ravel()
    
    return X, y
    

def get_rank_col(df):
    rank = df.columns[df.columns.str.endswith('_rank')][0]
    return rank

def plot_eda_results(df, metric, hue, plot, n=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    if n:
        df = df.sort_values(by=get_rank_col(df)).head(n)
    
    fig, ax = plt.subplots()
    
    if plot=='hist':
        sns.histplot(
            data=df,
            x=metric,
            hue=hue,
            multiple="stack",
            palette='deep'
        )
        move_legend(ax, "upper left", bbox_to_anchor=(1.04,1))
    elif plot=='boxplot':
        sns.boxplot(data=df, x=metric, y=hue)
        plt.ylabel(None)
        if hue == 'n_var':
            print('#TODO: check this')
#             n=10
#             start, end = ax.get_xlim()
#             positions = np.linspace(start, end, n)
#             labels = np.linspace(df[metric].min(), df[metric].max(), n)
#             plt.xticks(positions, labels)
#             import matplotlib.ticker as ticker
#             ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    elif plot =='scatter' and n:
        sns.scatterplot(
            data=df,
            x=metric,
            y='lst_col_str',
            hue='model'
        )      

    if metric == 'matthews_corrcoef':#'f1_score':
        target_val = 1
        problema = 'classificadores'
    elif metric == 'mean_squared_error':
        target_val = 0
        problema = 'regressores'
    
    if hue == 'model':
        text_var = 'modelos'
    elif hue == 'n_var':
        text_var = 'número de variáveis'
    elif hue == 'lst_col_str':
        text_var = 'variáveis'
        
    
    plt.title(f'Desempenho de {df.shape[0]} {problema}, \n considerando {text_var}')
    plt.xlabel(f'{metric} (quanto mais próximo de {target_val} melhor)')
    plt.show()
    
    pass

def table_results(dct_base, df, metric, index, figsize, n=10):
    # TODO: ABSTRAIR
    # DEPENDE DO FRAMEWORK DO ARQUIVO
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = df.sort_values(by=get_rank_col(df)).head(n)
    df[metric] = df[metric].round(2)
    
    if index == 'lst_col_str':
        df = df[[index, 'model', metric]].set_index(index)
    elif index == 'model':
        df = df[[index, metric]+dct_base['lst_col_selected']].set_index(index).T       
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    pd.plotting.table(ax, df)
    plt.show()
    pass

def eda_results_clf_reg(dct_base, df_results, metric):
    # TODO: ABSTRAIR
    # DEPENDE DO FRAMEWORK DO ARQUIVO
    
    import pandas as pd
    
    plot_eda_results(df=df_results, metric=metric, hue='model', plot='boxplot')
    plot_eda_results(df=df_results, metric=metric, hue='model', plot='hist')
    # plot_eda_results(df=df_results, metric=metric, hue='n_var', plot='boxplot')
    plot_eda_results(df=df_results, metric=metric, hue='n_var', plot='hist')
    plot_eda_results(df=df_results, metric=metric, hue='lst_col_str', plot='hist', n=10)
    plot_eda_results(df=df_results, metric=metric, hue='model', plot='scatter', n=20)

    table_results(dct_base, df_results, metric, index='lst_col_str', figsize=(3,0.1), n=10)
    table_results(dct_base, df_results, metric, index='model', figsize=(15,0.1), n=10)
        
    # metric stats for each algorithm
    pd.options.display.latex.repr=True
    display(df_results
     .groupby('model')[[metric]]
     .agg(['min','max','median','std','mean'])
     .sort_values(by=[(metric, 'median')], ascending=False))
    pd.options.display.latex.repr=False
    
    rank_col = get_rank_col(df_results)
    
    print('Distribuição dos melhores algoritmos de cada conjunto de variáveis.')
    # best classifier for each variables set
    best_model_each_var_set = (
        df_results
        .sort_values(rank_col)
        .drop_duplicates(['lst_col_str'])
    )
    # ut.display_full(
    #     best_model_each_var_set
    #     [['clf_rank','model',metric]+lst2]
    #     .head(5)
    # )

    # maior ocorrência dos algoritmos
    pd.options.display.latex.repr=True
    display(best_model_each_var_set['model'].value_counts())
    pd.options.display.latex.repr=False
    
    pass

def get_lst_features(lst_col_selected, index, target):
    lst_features = [x for x in lst_col_selected if x not in [index, target]]
    return lst_features

def get_df_results(dct_model, dct_base):
    # TODO: ABSTRAIR
    # DEPENDE DO FRAMEWORK DO ARQUIVO
    import pandas as pd
    
    df_results = pd.DataFrame.from_dict(dct_model, orient='index')
    
    df_results['lst_col_str'] = df_results['lst_features'].apply(
        lambda x: ', '.join(map(str, x)))
    
    df_results[dct_base['lst_col_selected']] = 0
    
    for col in dct_base['lst_col_selected']:
        mask = df_results['lst_col_str'].str.contains(col, regex=False)
        df_results.loc[mask, col] = 1
    
    return df_results

def get_df_results_reg(dct_reg, dct_base, metric_reg):
    df_results_reg = get_df_results(dct_reg, dct_base)
    
    # results regression
    df_results_reg = df_results_reg.drop(
        columns=["reg","scaler_x","scaler_y",
                 "y_pred","y_test","y_pred_train",
                 "annotation_string"]
    )
    # overall best regressors
    df_results_reg['reg_rank'] = df_results_reg[metric_reg].rank()
    
    return df_results_reg

def get_pred_train_test(dct_reg, ii):
    import numpy as np    
    
    y_pred_whole_series = np.concatenate( (
            dct_reg[ii]["y_pred_train"], 
            dct_reg[ii]["y_pred"]
        ))
    return y_pred_whole_series

def plot_reg(df_no_miss_data_reg, df_results_reg, dct_reg, index, target, test_start_idx_reg, metric_reg, ii):
    # TODO: ABSTRAIR
    # DEPENDE DO FRAMEWORK DO ARQUIVO
    
    import matplotlib.pyplot as plt
    
    y_pred_whole_series = get_pred_train_test(dct_reg, ii)
    
    df_no_miss_data_reg.plot(x=index, y=target, kind="scatter")
    plt.plot(df_no_miss_data_reg[index], y_pred_whole_series, color="red")
    rank = int(df_results_reg["reg_rank"].iloc[ii])
    plt.title(f'{dct_reg[ii]["model"]}: {rank}/{len(dct_reg)}')
    plt.axvline(x=(df_no_miss_data_reg[index]
                   .iloc[test_start_idx_reg]),
                linewidth=1, color='k')
    plt.xlabel(index)
    plt.annotate(fr'{dct_reg[ii]["annotation_string"]}',
                 xy=(1.04, 0.60), xycoords='axes fraction')
    plt.annotate(f'ID: {ii}\n{metric_reg}: {dct_reg[ii][metric_reg]:.3f}',
                xy=(1.04, 0.1), xycoords='axes fraction')
    plt.show()
    
    pass

def get_dct_coef(algoritmo, lst_features, model):
    if (algoritmo in ["Decision tree", "Random forest"]) and len(lst_features)>1:               
        dct_coef = dict(zip(lst_features, model.feature_importances_))
    elif algoritmo == "Regressão linear":
        dct_coef = dict(zip(lst_features, model.coef_[0]))
        dct_coef["Constante"] = model.intercept_[0]
    else:
        dct_coef = {}
    
    return dct_coef

def get_annotation_string(algoritmo, lst_features, dct_coef):
    # TODO: ABSTRAIR
    # DEPENDE DO FRAMEWORK DO ARQUIVO
    if not dct_coef:
        iterable = lst_features
        name = "Features"
    else:
        iterable = { f'{k}: {round(v, 3)}' for k,v in dct_coef.items() } 
        
        if algoritmo == "Regressão linear":
            name = "Coeficientes"
    
        elif (algoritmo in ["Decision tree", "Random forest"]) and len(lst_features)>1:
            name = "Feature importance"        
    
    annotation_string = f'{name}:\n' + '\n'.join(iterable)
    
    return annotation_string


def plot_confusion_matrix(dct_clf, ii):
    # TODO: ABSTRAIR
    # DEPENDE DO FRAMEWORK DO ARQUIVO
    
    import matplotlib.pyplot as plt
    
    make_confusion_matrix(
        dct_clf[ii]['confusion_matrix_test'],
        group_names=[f"tn\npreviu descida\ne desceu",
                     f"fp\npreviu subida\ne desceu",
                     f"fn\npreviu descida\ne subiu",
                     f"tp\npreviu subida\ne subiu",],
        categories=["descida","subida"]
    )
    plt.text(1.32,2.49,'taxa de acerto nas previsões')
    plt.text(1.32,2.58,'taxa de acerto quando previu subida')
    plt.text(1.32,2.69,'taxa de acerto quando subiu')
    
    pass 

def plot_errors_clf(df_encoded_raw, df_results_clf, dct_clf, index, target, test_start_idx_clf, ii):
    # TODO: ABSTRAIR
    # DEPENDE DO FRAMEWORK DO ARQUIVO
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    # Errors representation
    df_encoded_raw.iloc[test_start_idx_clf:].plot(x=index, y=target, ax=ax)
    df_encoded_raw.iloc[dct_clf[ii]['fp']].plot(x=index, y=target, kind='scatter', color='green', marker='^', ax=ax)
    df_encoded_raw.iloc[dct_clf[ii]['fn']].plot(x=index, y=target, kind='scatter', color='red', marker='v', ax=ax)
    plt.legend(['real','fp: previu subida e desceu','fn: previu descida e subiu'])#, title=row['model'])
    move_legend(ax, "upper left", bbox_to_anchor=(1.04,1))

    rank = df_results_clf["clf_rank"].iloc[ii]

    rank = int(rank) if ((rank % 1) < 0.1) else rank  # keep decimal if tied

    plt.title(f'{dct_clf[ii]["model"]}: {rank}/{len(dct_clf)}')
    plt.xlabel(index)
    plt.ylabel(target)
    plt.annotate(fr'{dct_clf[ii]["annotation_string"]}',
                 xy=(1.04, 0.30), xycoords='axes fraction') 
    plt.annotate(f'ID: {ii}',
                xy=(1.04, 0.1), xycoords='axes fraction')
    plt.show()
    
    plot_confusion_matrix(dct_clf, ii)
    
    pass

 
def count_consecutive_items_n_cols(df, col_name_list, output_col):
    """
    
        source: https://stackoverflow.com/questions/25119524/pandas-conditional-rolling-count
    """
    cum_sum_list = [
        (df[col_name] != df[col_name].shift(1)).cumsum().tolist() 
        for col_name in col_name_list
    ]
    df_out = df.copy()
    df_out[output_col] = df[col_name_list].groupby(
        ["_".join(map(str, x)) for x in zip(*cum_sum_list)]
    ).cumcount() + 1
    return df_out

def add_col_max_sequence_event_count(df, col):
    """
        source: https://stackoverflow.com/questions/66822515/how-to-repeat-the-cumsum-for-previous-values-in-a-pandas-series-when-the-count
    
    """
    df_out = df.copy()
    df_out[col+'_max'] = df[col].groupby(df[col].eq(1).cumsum()).transform('max')
    return df_out 
    
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    
    
    Source:
    https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscale xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    
    return lc
 
 
 
def get_cols_date(df_in, index):
    
    df = df_in.copy()
    
    df['year'] = df[index].dt.year
    df['month'] = df[index].dt.month
       
    dct_dt = {
        'bimester': 2,
        'trimester': 3,  # quarter
        'quadrimester': 4,
        'semester': 6
    }

    for period, n_months in dct_dt.items():
        df[period] = df['month'].apply(
            lambda month: 
            (month-1)//n_months + 1 
    #         n_months_group(month, n_months)
        )
    
    return df

def get_ano_safra(month, year, safra_start_month):
    return f'{year}-{year+1}' if (month>=safra_start_month) else f'{year-1}-{year}'
    

def plot_limit_pct_change(df, col, index, pct_change):
    import matplotlib.pyplot as plt
    # import matplotlib.ticker as mtick
    from matplotlib.ticker import FuncFormatter

    ax = df[[col]].pct_change().set_index(df[index]).plot(legend=False)

    for y in [-pct_change, pct_change]:
        ax.hlines(y=y,
                  xmin=df[index].iloc[0],
                  xmax=df[index].iloc[-1],
                  colors='black',
                  linestyle='dashed'
                 )
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

    plt.ylabel('Variação percentual '+col)
    # plt.legend()
    plt.show()  
    pass
    

def get_3_classes(x, pct_change):
    import numpy as np
    

    if np.isnan(x):
        return np.nan
    elif x > pct_change:
        return 1    # increase
    elif x < -pct_change:
        return -1   # decrease
    elif (x >= -pct_change) and (x <= pct_change): 
        return 0    # stagnation
    else:
        print(x, type(x))
        raise Exception('Not should be here')
        

def plot_col_level_vs_target(df, index, target_reg, pct_change_over_name, col, bins):

    import matplotlib.pyplot as plt
    import seaborn as sns
   
    # import utils as ut
    
    col_level = col+'_'+''.join([str(x) for x in range(bins)])
    
    cm = sns.color_palette("RdBu", bins)
      
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x=index,
        y=target_reg,
        hue=col_level,
        palette=cm,
        style=pct_change_over_name,
        markers=['v','s','^']
    )
    # ut.move_legend
    move_legend(ax, "upper left", bbox_to_anchor=(1.04,1))
    plt.show()
    
    pass
    

def get_level_col(df, col, bins):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    import pprint
    
#     import utils as ut

    col_level = col+'_'+''.join([str(x) for x in range(bins)])
    
    cut = pd.cut(df[col], bins=bins)
    dct = dict(zip(sorted(set(cut.dropna().to_list())), [x for x in range(bins)]))
    df[col_level] = cut.map(dct).astype(float)
    
    pprint.pprint(dct)
       
    df[[col, col_level]].hist(figsize=(10,5))#, bins=bins)
    plt.show()
   
    return dct, col_level
    
    
def plot_feature_importance(clf, lst_col):

    import pandas as pd
    import matplotlib.pyplot as plt
    
    col = 'feature_importances'
    (pd.DataFrame(
        data=clf.feature_importances_,
        index=lst_col,
        columns=[col]).sort_values(by=[col])
     .plot(kind='barh'))
    plt.show()
    
    pass 
    

def plot_pair_vs_target_clf(df, target_clf):
    import seaborn as sns
    import matplotlib.pyplot as plt


    g = sns.PairGrid(
        df,
        diag_sharey=False, 
        corner=True,
        hue=target_clf,
        palette=['orange', 'blue']
    )
    g.map_lower(sns.scatterplot)

    # g.map_lower(sns.residplot)
    # g.map_upper(sns.regplot)

    # https://stackoverflow.com/questions/57644355/stacked-barchart-in-pairgrid-python-seaborn
    # below for the histograms on the diagonal
    d = {}
    def func(x, **kwargs):
        ax = plt.gca()

        if not ax in d.keys():
            d[ax] = {"data" : [], "color" : []}
        d[ax]["data"].append(x)
        d[ax]["color"].append(kwargs.get("color"))

    g.map_diag(func)
    for ax, dic in d.items():
        ax.hist(dic["data"], color=dic["color"], histtype="barstacked")

    # g.map_diag(sns.histplot)
    # g.map_diag(sns.kdeplot)
    g.add_legend()
    plt.show()
    
    pass


def plot_col_vs_target_time(df_standardized, index, col, target, dct_encoding):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator
    
    fig, ax = plt.subplots()
    df_standardized.plot(x=index,y=[col,target], ax=ax)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    if col in dct_encoding.keys(): 
        positions = tuple(df_standardized[col].unique())
#             labels = list(lst_dct_strings[lst_col_string.index(col)].values())[0].keys()
        # labels = list(dct_encoding[col])[0].keys()
        labels = dct_encoding[col].keys()
        plt.yticks(positions, labels)
        secax = ax.secondary_yaxis('right')
        secax.yaxis.set_major_locator(AutoLocator())
    plt.show()
    pass
    
    
def get_IQR(s):
    import numpy as np

    return np.subtract(*np.percentile(s, [75, 25]))
#    return s.quantile(.75) - s.quantile(.25)
    
def get_bins_number(s):
    # https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    iqr = get_IQR(s)
    
    def get_bins_width(iqr, n):
        return 2 * iqr * n ** (-1/3)
   
    bins_width = get_bins_width(iqr, len(s))

    return round( (max(s) - min(s)) / bins_width )
