from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as s
import matplotlib.pyplot as plt


def confusion(chars, predictions):
    size = len(chars)
    matrix = np.zeros((size, size), dtype=int)
    for prediction in predictions:
        matrix[chars.index(prediction[0]), chars.index(prediction[1])] += 1
        if prediction[0] != prediction[1]:
            print('My bad! I thought this {} was a {}'.format(prediction[0], prediction[1]))
    accuracy = round((np.trace(matrix) / len(predictions)) * 100, 1)
    plot_confusion(chars, matrix)
    return accuracy


def plot_confusion(chars, matrix):
    s.heatmap(pd.DataFrame(matrix, index=chars, columns=chars),
              annot=True, cmap="PuBu",
              linewidths=.0,
              cbar_kws={'label': 'Accuracy %'})
    plt.ylabel('Class')
    plt.xlabel('Prediction')
    plt.title('Confusion Matriz')
    plt.show()





