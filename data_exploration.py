import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

def load_dataset():
    return pd.read_csv("arruela_.csv")


def drop_useless_columns(dataset):
    columns = ["Hora", "Tamanho", "Referencia", "Delta"] # Delta removido do Dataset
    return dataset.drop(columns, axis=1)


def plot_amount_output2(dataset):
    sb.set_style("whitegrid")
    sb.countplot(x="Output2", data=dataset, palette='RdBu_r')
    plt.savefig("./output/countplot-amount2.png")


def dataset_pair_plot(dataset):
    sb.pairplot(dataset)
    plt.savefig("./output/dataset-pairplot.png")


# Identificar features que tenham grande
# correlação com a variável preditiva, porém
# pouca correlação entre si.
def heatmap(dataset):
    sb.heatmap(dataset.corr(), annot=True)
    plt.savefig("./output/dataset-heatmap.png")


if __name__ == "__main__":
    original_dataset = load_dataset()
    clean_dataset = drop_useless_columns(original_dataset)

    # plot_amount_output2(clean_dataset)
    # dataset_pair_plot(clean_dataset)
    # heatmap(clean_dataset)