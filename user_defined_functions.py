import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

def stacked_barplot(df_original, response_var, feature_name, plot_type, top_n_category=5, bar_width=0.3):
    """
    This function plots the count/percentage of the binary response variable within each category of a feature

    param: df_original - dataframe
    param: response_var - binary response variable
    param: col_list - a list of columns to base the barplot on, can accept 2 columns at most
    param: plot_type - the values can be 'percent' for stacked barplot of percentage,
                        or 'count' for regular stacked barplot
    param: top_n_category - top n categories to include in the plot, default is 5
    param: bar_width - the width of the bar, default is 0.3
    """
    df = df_original.copy()
    count = pd.DataFrame(df[feature_name].value_counts())
    top_index = pd.Series(count.head(top_n_category).index)
    top = df[df[feature_name].isin(top_index)]
    df_count = pd.DataFrame(top.groupby([response_var, feature_name]).size(), columns=['count'])
    df_count.reset_index(inplace=True)

    df_pivot = df_count.pivot(index=feature_name, columns=response_var, values='count')
    df_pivot['% Defaulted'] = df_pivot['Defaulted'] / (df_pivot['Defaulted'] + df_pivot['Fully Paid'])
    df_pivot["% Fully Paid"] = df_pivot['Fully Paid'] / (df_pivot['Defaulted'] + df_pivot['Fully Paid'])
    df_pivot = df_pivot.sort_values(by='% Defaulted', axis=0, ascending=False)

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes.grid(color='grey', linestyle='-', linewidth='0.3')
    if plot_type == 'count':
        del df_pivot['% Defaulted'], df_pivot['% Fully Paid']
        df_pivot.columns = ['# of Defaulted', '# of Fully Paid']
        ax = df_pivot.plot(ax=axes, kind='bar', stacked=True, width=bar_width)
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(x + width / 2.0,
                    y + height / 2.0,
                    '{:.0f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='bottom')
        axes.set_ylabel('count')

    elif plot_type == 'percent':
        del df_pivot['Defaulted'], df_pivot['Fully Paid']
        ax = df_pivot.plot(ax=axes, kind='bar', stacked=True, width=bar_width)
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(x + width / 2.0,
                    y + height / 2.0,
                    '{:.2f} %'.format(height * 100),
                    horizontalalignment='center',
                    verticalalignment='bottom')
        axes.set_ylim((0, 1.2))
        axes.set_ylabel('percent')

    axes.set_xlabel(" ".join(feature_name.split('_')))
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 1.3, box.height])
    axes.legend(loc='upper right')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')