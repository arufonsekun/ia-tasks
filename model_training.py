from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from data_exploration import load_dataset, drop_useless_columns, heatmap

def normalization(dataset):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    features = [
        "NumAmostra",
        "Area",
        "Delta",
        "Output1",
        "Output2"
    ]
    return pd.DataFrame(np.array(scaled_data), columns=features);


def split_dataset(scaled_dataset):
    X = scaled_dataset.drop(["Output1", "Output2"], axis=1)
    y = scaled_dataset[["Output1", "Output2"]]
    return X, y


def train_test_split(X, y):
    seed = 42
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=seed)

    return X_train, X_test, y_train, y_test


def model_setup():

    N_input = 3
    N_hidden = 8
    N_output = 2
    weights_input_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
    weights_hidden_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))

    return weights_input_hidden, weights_hidden_output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit(X_train, y_train):
    n_records, n_features = X_train.shape
    learn_rate = 0.1

    weights_input_hidden, weights_hidden_output = model_setup()
    epochs = 200000
    last_loss = None
    evolution_error = []
    index_error = []

    for e in range(epochs):
        delta_w_i_h = np.zeros(weights_input_hidden.shape)
        delta_w_h_o = np.zeros(weights_hidden_output.shape)

        for xi, yi in zip(X_train.values, y_train.values):
            hidden_layer_input = np.dot(xi, weights_input_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)

            output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)

            output = sigmoid(output_layer_in)

            # Erro
            error = yi - output
            # Gradiente da camada de saída
            output_error_term = error * output * (1 - output)
            # Calcula a distribuição da camada oculta para o erro
            hidden_error = np.dot(weights_hidden_output, output_error_term)
            # Gradiente da Camada Oculta
            hidden_error_term = hidden_error * hidden_layer_output * (1 - hidden_layer_output)
            # Calcula a variação do peso da camada de saída
            delta_w_h_o += output_error_term*hidden_layer_output[:, None]
            # Calcula a variação do peso da camada oculta
            delta_w_i_h += hidden_error_term * xi[:, None]

        # Atualização dos pesos na época atual
        weights_input_hidden += learn_rate * delta_w_i_h / n_records
        weights_hidden_output += learn_rate * delta_w_h_o / n_records

        if e % (epochs / 20) == 0:
            hidden_output = sigmoid(np.dot(xi, weights_input_hidden))
            out = sigmoid(np.dot(hidden_output, weights_hidden_output))
            loss = np.mean((out - yi) ** 2)

            if last_loss and last_loss < loss:
                print("Erro quadrático no treinamento: ", loss, " Atenção: O erro está aumentando")
            else:
                print("Erro quadrático no treinamento: ", loss)
            last_loss = loss
            evolution_error.append(loss)
            index_error.append(e)

    plot_error_evolution(index_error, evolution_error)


def plot_error_evolution(index_error, evolution_error):
    plt.plot(index_error, evolution_error, 'r') # 'r' is the color red
    plt.xlabel("")
    plt.ylabel("Erro Quadrático")
    plt.title("Evolução do Erro no treinamento da MPL")
    plt.show()


if __name__ == "__main__":
    dataset = load_dataset()
    clean_dataset = drop_useless_columns(dataset)
    scaled_dataset = normalization(clean_dataset)
    X, y = split_dataset(scaled_dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    fit(X_train, y_train)