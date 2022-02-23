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
        # "Delta",
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

    N_input = 2 # Nº de  features
    N_hidden1 = 4
    N_hidden2 = 5
    N_output = 2
    weights_input_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden1))
    weights_hidden_hidden = np.random.normal(0, scale=0.1, size=(N_hidden1, N_hidden2))
    weights_hidden_output = np.random.normal(0, scale=0.1, size=(N_hidden2, N_output))

    return weights_input_hidden, weights_hidden_hidden, weights_hidden_output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit(X_train, y_train, X_test, y_test):
    n_records, n_features = X_train.shape
    learn_rate = 0.1

    weights_input_hidden, weights_hidden_hidden, weights_hidden_output = model_setup()
    epochs = 100000
    last_loss = None
    evolution_error = []
    index_error = []

    for e in range(epochs):
        delta_w_i_h1 = np.zeros(weights_input_hidden.shape)
        delta_w_h1_h2 = np.zeros(weights_hidden_hidden.shape)
        delta_w_h2_o = np.zeros(weights_hidden_output.shape)

        for xi, yi in zip(X_train.values, y_train.values):
            hidden1_layer_input = np.dot(xi, weights_input_hidden)
            hidden1_layer_output = sigmoid(hidden1_layer_input)

            hidden2_layer_input = np.dot(hidden1_layer_output, weights_hidden_hidden)
            hidden2_layer_output = sigmoid(hidden2_layer_input)

            output_layer_input = np.dot(hidden2_layer_output, weights_hidden_output)
            output = sigmoid(output_layer_input)

            # Erro
            error = yi - output
            # Gradiente da camada de saída
            output_error_term = error * output * (1 - output)

            # Calcula a distribuição da 2º camada oculta para o erro
            hidden2_error = np.dot(weights_hidden_output, output_error_term)

            # Gradiente da 2ª Camada Oculta
            hidden2_error_term = hidden2_error * hidden2_layer_output * (1 - hidden2_layer_output)

            # Calcula a distribuição da 1º camada oculta para o erro
            hidden1_error = np.dot(weights_hidden_hidden, hidden2_error_term)

            # Gradiente da 1ª Camada Oculta
            hidden1_error_term = hidden1_error * hidden1_layer_output * (1 - hidden1_layer_output)

            delta_w_h2_o += output_error_term * hidden2_layer_output[:, None]

            delta_w_h1_h2 += hidden2_error_term * hidden1_layer_output[:, None]

            # Calcula a variação do peso da camada oculta
            delta_w_i_h1 += hidden1_error_term * xi[:, None]

        # Atualização dos pesos na época atual
        weights_input_hidden += learn_rate * delta_w_i_h1 / n_records
        weights_hidden_hidden += learn_rate * delta_w_h1_h2 / n_records
        weights_hidden_output += learn_rate * delta_w_h2_o / n_records

        if e % (epochs / 20) == 0:
            hidden2_input = sigmoid(np.dot(xi, weights_input_hidden))
            hidden2_output = sigmoid(np.dot(hidden2_input, weights_hidden_hidden))
            # print("hidden_output.shape", hidden_output.shape)
            # print("weights_input_hidden.shape", weights_input_hidden.shape)
            # return
            out = sigmoid(np.dot(hidden2_output, weights_hidden_output))
            loss = np.mean((out - yi) ** 2)

            if last_loss and last_loss < loss:
                print("Erro quadrático no treinamento: ", loss, " Atenção: O erro está aumentando", "Época: {0}".format(e))
            else:
                print("Erro quadrático no treinamento: ", loss, "Época: {0}".format(e))
            last_loss = loss
            evolution_error.append(loss)
            index_error.append(e)
        else:
            print("Época: {0}".format(e))


    plot_error_evolution(index_error, evolution_error, epochs)
    evaluate(X_test, y_test, weights_input_hidden, weights_hidden_hidden, weights_hidden_output)

def evaluate(X_test, y_test, weights_input_hidden, weights_hidden_hidden, weights_hidden_output):
    
    n_records, n_features = X_test.shape
    predictions = 0

    for xi, yi in zip(X_test.values, y_test.values):
        hidden1_layer_input = np.dot(xi, weights_input_hidden)

        hidden1_layer_output = sigmoid(hidden1_layer_input)

        hidden2_layer_input = np.dot(hidden1_layer_output, weights_hidden_hidden)

        output_layer_in = np.dot(hidden2_layer_input, weights_hidden_output)

        output = sigmoid(output_layer_in)

        if (output[0] > output[1]):
            if (yi[0] > yi[1]):
                predictions += 1

        if (output[1] > output[0]):
            if (yi[1] > yi[0]):
                predictions += 1

    accuracy = predictions / n_records
    print("A acurácia do modelo é de {:.3f}".format(accuracy))


def plot_error_evolution(index_error, evolution_error, epochs):
    plt.plot(index_error, evolution_error, "r")
    plt.xlabel("")
    plt.ylabel("Erro Quadrático")
    plt.title("Evolução do Erro no treinamento da MPL - {0} Épocas".format(epochs))
    plt.savefig("./output/mean-square-error-evolution.png")


if __name__ == "__main__":
    dataset = load_dataset()
    clean_dataset = drop_useless_columns(dataset)
    scaled_dataset = normalization(clean_dataset)
    print(scaled_dataset.head())
    X, y = split_dataset(scaled_dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    fit(X_train, y_train, X_test, y_test)