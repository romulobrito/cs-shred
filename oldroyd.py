import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
import json
import os
import time

import models
from processdata import TimeSeriesDataset

# Caminho do arquivo .npy
# npy_file_path = r"/home/romulo/Downloads/prmsl_data.npy"
# npy_file_path = r"/home/romulo/Downloads/qmax.2m.1836_data_mavg.npy"
npy_file_path = r"/home/romulo/migoogledrive/shred-jan/pyshred/Data/16_roll7_Re1_Wi3.5_beta0.6666/fields.npy"

save_path = r'./results/csshred/oldroyd/'
# save_path = r'./results/shred/testes'

# Verifica a disponibilidade de GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Função para carregar dados de um arquivo .npy
def load_data(npy_file_path, time_slice):
    data_array = data = np.load(
    npy_file_path,allow_pickle=True
    )
    data_array = data_array.item()["trace_A"]
    data_array = data_array[:,:,time_slice:]
    data_array = np.transpose(data_array, (2,0,1))
    print("Loaded data dimensions:", data_array.shape)
    return data_array



# Visualização dos dados 2D ou 3D
def visualize_data(matrix, subsampled):
    # Plot para o último slice temporal da matriz
    plt.imshow(matrix[-1, :, :], cmap="Spectral", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


    # Plot para o último slice temporal da matriz subsample
    plt.imshow(subsampled[:, :, -1], cmap="Spectral", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice (Subsampled)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


#Subamostragem dos dados
def subsample(snapshot, num_cols_subsample, num_snapshots_subsample):
    
    np.random.seed(1001)

    print('snapshot', snapshot.shape)

    snapshot = np.transpose(snapshot, (1, 2, 0))
    dim_x, dim_y, dim_t = snapshot.shape
    snapshot_subsampled = snapshot.copy()

    # Escolha aleatória das colunas a serem mantidas (não subamostradas)
    cols_to_keep = np.random.choice(dim_y, size=dim_y - num_cols_subsample, replace=False)
    cols_to_keep = sorted(cols_to_keep)

    # Escolha aleatória dos snapshots a serem mantidos (não subamostrados)
    snapshots_to_keep = np.random.choice(dim_t, size=dim_t - num_snapshots_subsample, replace=False)

    # Criar uma máscara de uns
    mask = np.ones((dim_x, dim_y, dim_t), dtype=bool)

    # Definir os valores a serem subamostrados como False na máscara
    mask[:, cols_to_keep, :] = False
    mask[:, :, snapshots_to_keep] = False

    # Aplicar a máscara
    snapshot_subsampled[mask] = 0

    print("Forma do snapshot após subamostragem:", snapshot_subsampled.shape)
    return snapshot_subsampled



# def subsample(snapshot, percent_subsample):
#     np.random.seed(1001)
#     print('snapshot', snapshot.shape)

#     snapshot = np.transpose(snapshot, (1, 2, 0))
#     num_cols_subsample = int(snapshot.shape[1] * percent_subsample)
#     snapshot_subsampled = snapshot.copy()

#     # Gera os índices aleatórios uma única vez
#     random_indices = np.random.choice(
#         snapshot.shape[1], size=num_cols_subsample, replace=False
#     )
#     random_indices = sorted(random_indices)
#     diff_indices = np.diff(random_indices)
#     for j in range(len(random_indices) - 1):
#         if diff_indices[j] > 4:
#             random_indices[j + 1] = random_indices[j] + 3

#     # Aplica a subamostragem usando os mesmos índices para todos os snapshots
#     for i in range(snapshot.shape[2]):
#         snapshot_subsampled[:, random_indices, i] = 0

#     print("Forma do snapshot após subamostragem:", snapshot_subsampled.shape)
#     return snapshot_subsampled

# Configuração dos sensores
def plot_dynamics_at_sensors(
    trace_A, num_sensors, locations="c", show_plot=False, save_plot=True, save_path=save_path, file_name="plot_din.png", seed=101,
    auto_close_time=5
):
    np.random.seed(seed)

    dim_x, dim_y, dim_t = trace_A.shape

    print("Forma do snapshot após subamostragem:", trace_A.shape)

    if locations == "a":
        central_x, central_y = 0.5, 0.5
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

        sensor_positions_x = np.append(sensor_positions_x, central_x)
        sensor_positions_y = np.append(sensor_positions_y, central_y)

    elif locations == "b":
        sensor_positions_x, sensor_positions_y = [0.5], [0.5]

    else:
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

    sensor_temperature_history = []
    for t in range(dim_t):
        sensor_temperatures = [
            trace_A[int(dim_x * x), int(dim_y * y), t]
            for x, y in zip(sensor_positions_x, sensor_positions_y)
        ]
        sensor_temperature_history.append(sensor_temperatures)
    if show_plot or save_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        y = np.linspace(0, 1, int(dim_x))
        x = np.linspace(0, 1, int(dim_y))
        X, Y = np.meshgrid(x, y)

        cmap = ax1.pcolormesh(
            X, Y, trace_A[:, :, -1].real, shading="auto", cmap="coolwarm"
        )
        fig.colorbar(cmap, ax=ax1, label=r"$Tr(C)$")
        ax1.set_title("Espatial Distribution of The $Tr(C)$")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        ax1.scatter(
            sensor_positions_x,
            sensor_positions_y,
            color="k",
            label="Sensor Positions",
        )
        ax1.legend()

        sensor_temperature_history = np.array(sensor_temperature_history)
        for i, sensor_data in enumerate(sensor_temperature_history.T):
            ax2.plot(range(dim_t), sensor_data, label=f"Sensor {i+1}")

        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Dynamics at Sensor Positions")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_plot:
            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file = os.path.join(save_path, file_name)
            else:
                save_file = file_name
            plt.savefig(save_file)
            print(f"Plot saved to {save_file}")

        if show_plot:
            plt.show()
            # time.sleep(auto_close_time)  # Espera por um tempo específico
            # plt.close(fig)
        else:
            plt.close(fig)

    return sensor_locations, sensor_positions_x, sensor_positions_y


# Preparação dos dados para treinamento e validação
def prepare_datasets(trace_A, trace_A_ori, num_sensors, sensor_locations, lags):
    trace_A_ori = np.transpose(trace_A_ori, (1, 2, 0))
    num_sensors = num_sensors

    dim_x, dim_y, dim_t = trace_A.shape

    lags = lags
    train_size = int(0.7 * trace_A.shape[2])
    val_size = int(0.2 * trace_A.shape[2])
    test_size = int(0.1 * trace_A.shape[2])

    print("train_size", train_size)
    print("val_size", val_size)
    print("test_size", test_size)

    print("trace_A", trace_A.shape)
    print("trace_A", trace_A_ori.shape)

    load_X = trace_A.reshape(dim_x * dim_y, dim_t).T
    load_X_test = trace_A_ori.reshape(dim_x * dim_y, dim_t).T

    load_X_shape_0, load_X_shape_1 = load_X.shape
    print(load_X_shape_0, load_X_shape_1)

    # Garantindo que os tamanhos não ultrapassem o tamanho real dos dados
    total_size = train_size + val_size + test_size
    if total_size > load_X_shape_0 - lags:
        train_size = (train_size * (load_X_shape_0 - lags)) // total_size
        val_size = (val_size * (load_X_shape_0 - lags)) // total_size
        test_size = (test_size * (load_X_shape_0 - lags)) // total_size

    train_indices = np.random.choice(
        load_X_shape_0 - lags, size=train_size, replace=False
    )
    mask = np.ones(load_X_shape_0 - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, load_X_shape_0 - lags)[np.where(mask != 0)[0]]

    valid_indices = valid_test_indices[:val_size]
    test_indices = valid_test_indices[val_size : val_size + test_size]

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    sc_test = sc.fit(load_X_test[train_indices])
    transformed_X_test = sc.transform(load_X_test)

    all_data_in = np.zeros((load_X_shape_0 - lags, lags, num_sensors))
    all_data_in_test = np.zeros((load_X_shape_0 - lags, lags, num_sensors))
    for i in range(load_X_shape_0 - lags):
        for j, loc in enumerate(sensor_locations):
            all_data_in[i, :, j] = transformed_X[i : i + lags, loc]
            all_data_in_test[i, :, j] = transformed_X_test[i : i + lags, loc]

    print("device:", device)
    print("Número de sensores:{}".format(num_sensors))

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(
        device
    )
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(
        device
    )
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(
        device
    )
    test_data_in_test = torch.tensor(
        all_data_in_test[test_indices], dtype=torch.float32
    ).to(device)

    train_data_out = torch.tensor(
        transformed_X[train_indices + lags - 1], dtype=torch.float32
    ).to(device)
    valid_data_out = torch.tensor(
        transformed_X[valid_indices + lags - 1], dtype=torch.float32
    ).to(device)
    test_data_out = torch.tensor(
        transformed_X[test_indices + lags - 1], dtype=torch.float32
    ).to(device)
    test_data_out_test = torch.tensor(
        transformed_X_test[test_indices + lags - 1], dtype=torch.float32
    ).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    test_dataset_test = TimeSeriesDataset(test_data_in_test, test_data_out_test)

    return train_dataset, valid_dataset, test_dataset_test, sc, load_X_shape_1


# Treinamento e validação do modelo
def train_and_validate_model(
    type_model,
    model,
    train_dataset,
    valid_dataset,
    num_epochs,
    batch_size,
    lr,
    lambL2,
    lambL1,
    lambdaSNR,
    step_epoch,
    verbose,
    patience,
):
    if type_model == "CS-SHRED":
        train_error, validation_errors = models.fit_csshred_model(
            model,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lambL2=lambL2,
            lambL1=lambL1,
            lambdaSNR=lambdaSNR,
            step_epoch=step_epoch,
            verbose=verbose,
            patience=patience,
        )
        return train_error, validation_errors
    else:
        validation_errors = models.fit(
            model,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            step_epoch=step_epoch,
            verbose=True,
            patience=patience,
        )
        return validation_errors


def evaluate_model(model, test_dataset, sc, json_save_path=save_path +'/error_results.json'):
    """
    Avalia o modelo calculando o erro normalizado e o SSIM entre as previsões e o ground truth.

    :param model: O modelo treinado.
    :param test_dataset: O conjunto de dados de teste.
    :param sc: O scaler usado para normalizar os dados.
    :param json_save_path: Caminho para salvar os resultados em um arquivo JSON.
    :return: Dados reconstruídos, dados verdadeiros e o erro normalizado.
    """
    # Realiza a previsão com o modelo e transforma os dados de volta ao formato original
    test_recons = sc.inverse_transform(model(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    
    # Verifica as dimensões dos arrays
    if test_recons.ndim != test_ground_truth.ndim:
        raise ValueError("As dimensões dos dados reconstruídos e ground truth não correspondem.")
    
    # Calcula o erro normalizado
    error_norm = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    
    # Calcula o SSIM para cada snapshot
    ssim_scores = []
    for i in range(test_recons.shape[0]):
        ssim_score = ssim(test_ground_truth[i], test_recons[i], data_range=test_ground_truth[i].max() - test_ground_truth[i].min())
        ssim_scores.append(ssim_score)
    mean_ssim = np.mean(ssim_scores)
    
    print("Mean SSIM:", mean_ssim)
    print("Normalized Error:", error_norm)
    
    # Cria o diretório se não existir
    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    
    # Salva os resultados em um arquivo JSON
    results = {
        'Normalized_Error': float(error_norm),
        'SSIM': {
            'overall': float(mean_ssim),
            'snapshots': ssim_scores
        }
    }
    with open(json_save_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return test_recons, test_ground_truth, error_norm


def add_model_info_to_json(json_file_path, model_type, model_params, config_params):
    """
    Adiciona informações do modelo e configuração a um arquivo JSON existente ou cria um novo.

    :param json_file_path: Caminho para o arquivo JSON.
    :param model_type: Tipo do modelo (CS-SHRED ou SHRED).
    :param model_params: Dicionário com os parâmetros do modelo.
    :param config_params: Dicionário com os parâmetros de configuração.
    """
    # Verifica se o arquivo JSON já existe
    if os.path.exists(json_file_path):
        # Se existir, carrega o conteúdo
        with open(json_file_path, 'r') as json_file:
            results = json.load(json_file)
    else:
        # Se não existir, cria um dicionário vazio
        results = {}

    # Adiciona as novas informações
    results['Model_Type'] = model_type
    results['Model_Parameters'] = model_params
    results['Configuration_Parameters'] = config_params

    # Salva o arquivo JSON atualizado
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Model information added to {json_file_path}")



# Parâmetros comuns
seed = 915
verbose = True
patience = 10
step_epoch = 50
# Escolha do modelo CS-SHRED/SHRED
model_type = "CS-SHRED"

# Carregamento dos dados
matrix = load_data(npy_file_path, time_slice=0)

begin_time = time.time()


# Subamostragem e visualização dos dados
num_cols_subsample = int(matrix.shape[2] * 0.9)  # % das colunas serão subamostradas
num_snapshots_subsample = int(matrix.shape[0] * 0.7)  #  % dos snapshots serão subamostrados
snapshot = subsample(matrix, num_cols_subsample, num_snapshots_subsample)
# visualize_data(matrix, snapshot)


# hidden_size= 512
# hidden_layers= 1
# batch_size= 64
# lr= 0.0023024787914655426
# lambL2= 0.1607657651820694
# lambL1= 0.07515465378909521
# lambdaSNR= 0.04495920482255287
# l1= 500
# l2= 600
# lags= 24

# hidden_size= 128    #  SHRED
# hidden_layers= 2
# batch_size= 512
# lr= 0.004838932184239264
# lambL2= 0.9725017970672707
# lambL1= 0.2568758900116353
# lambdaSNR= 0.0023009683873444417
# l1= 600
# l2= 600
# lags= 16


# Parâmetros de treinamento  CS-SHRED SST
# hidden_size=64
# hidden_layers=2
# batch_size=128
# lr=0.002606434698023075
# lambL2=0.592069825195236
# lambL1=0.002999642862729624
# lambdaSNR=0.01574493140076599
# l1=600
# l2=300
# lags=24

# Parâmetros de treinamento  CS-SHRED qmax
# hidden_size= 256
# hidden_layers= 1
# batch_size= 512
# lr= 0.00037706714496823075
# lambL2= 0.37697815730679735
# lambL1= 0.013110681923458177
# lambdaSNR= 0.0037632278472859095
# l1= 500
# l2= 600
# lags= 24

# Parâmetros de treinamento  CS-SHRED oldroyd old
# hidden_size= 256
# hidden_layers= 1
# batch_size= 512
# lr= 0.00037706714496823075
# lambL2= 0.37697815730679735
# lambL1= 0.013110681923458177
# lambdaSNR= 0.0037632278472859095
# l1= 500
# l2= 600
# lags= 24
# num_sensors = 2
# num_epochs = 2000


# Parâmetros de treinamento  CS-SHRED oldroyd
hidden_size=256
hidden_layers=2
batch_size=512
lr=0.005063016934642184
lambL2=0.7400965498783587
lambL1=0.003139927307990767
lambdaSNR=0.011825041569135032
l1=300
l2=400
lags=10
num_sensors=1
num_epochs=1497



# Parâmetros de treinamento  SHRED oldroyd
# hidden_size= 128
# hidden_layers= 1
# batch_size= 128
# lr= 0.03420381377030703
# lambL2= 0.15932806526755558
# lambL1= 0.006647369864904643
# lambdaSNR= 0.04274742006188003
# l1= 300
# l2= 400
# lags= 20
# num_sensors= 1
# num_epochs= 665

# hidden_size=32
# hidden_layers=1
# batch_size=512
# lr=0.04948382512132273
# lambL2=0.011565873149469828
# lambL1=0.03302436801060761
# lambdaSNR=0.03846504873833428
# l1=500
# l2=300
# lags=3
# num_sensors=5
# num_epochs=2739


# Configuração dos sensores
sensor_locations, sensor_positions_x, sensor_positions_y = plot_dynamics_at_sensors(
    snapshot, num_sensors, locations="c", show_plot=False, save_plot=True, save_path=save_path, file_name=f"plot_din_{model_type}.png", seed=seed
)

# Preparação dos conjuntos de dados
train_dataset, valid_dataset, test_dataset, sc, load_X_shape_1 = prepare_datasets(
    snapshot, matrix, num_sensors, sensor_locations, lags
)


# Treinamento e validação do modelo
if model_type == "CS-SHRED":
    # Instanciação e configuração do modelo CS-SHRED
    model = models.CSSHRED(
        num_sensors,
        load_X_shape_1,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        l1=l1,
        l2=l2,
        dropout=0.0,
        l1_tol=1e-4,
        opt_tol= 1e-6,
        ls_tol= 1e-6,
        n_sparsity_threshold=num_snapshots_subsample,
        verbosity=0,
        show_plot=False,
).to(device)
    train_error, validation_errors = train_and_validate_model(
        model_type,
        model,
        train_dataset,
        valid_dataset,
        num_epochs,
        batch_size,
        lr,
        lambL2,
        lambL1,
        lambdaSNR,
        step_epoch,
        verbose,
        patience,
    )
else:
    # Instanciação e configuração do modelo SHRED
    model = models.SHRED(  # 64
        num_sensors,
        load_X_shape_1,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        l1=l1,
        l2=l2,
        dropout=0.0,
    ).to(device)
    validation_errors = train_and_validate_model(
        model_type,
        model,
        train_dataset,
        valid_dataset,
        num_epochs,
        batch_size,
        lr,
        lambL2,
        lambL1,
        lambdaSNR,
        step_epoch,
        verbose,
        patience,
    )


# Avaliação do modelo
test_recons, test_ground_truth, error_norm = evaluate_model(model, test_dataset, sc, json_save_path=save_path+ r'error_results.json')



end_time = time.time()
total_time = (end_time - begin_time)/60
print(f"Tempo de execução: {total_time:.2f} minutos")   



# Definição dos parâmetros do modelo
model_params = {
    'hidden_size': hidden_size,
    'hidden_layers': hidden_layers,
    'batch_size': batch_size,
    'lr': lr,
    'lambL2': lambL2,
    'lambL1': lambL1,
    'lambdaSNR': lambdaSNR,
    'l1': l1,
    'l2': l2,
    'lags': lags,
    'num_sensors': num_sensors,
    'num_epochs': num_epochs,
    'step_epoch': step_epoch,

}

# Definição dos parâmetros de configuração
config_params = {
    'seed': seed,
    'verbose': verbose,
    'patience': patience,
    'num_cols_subsample': num_cols_subsample,
    'num_snapshots_subsample': num_snapshots_subsample,
    'total_time': total_time
}

# Caminho para o arquivo JSON
json_file_path = save_path + r'/error_results.json'


add_model_info_to_json(json_file_path, model_type, model_params, config_params)


if model_type == 'CS-SHRED':

    def save_to_numpy(test_recons, test_ground_truth, matrix, snapshot, sensor_positions_x, sensor_positions_y, train_error, validation_errors, model, directory=save_path):
        # Verifica se o diretório 'results' existe, senão cria
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Salva cada conjunto de dados em um arquivo .npy separado
        np.save(os.path.join(directory, "test_recons.npy"), test_recons)
        np.save(os.path.join(directory, "test_ground_truth.npy"), test_ground_truth)
        np.save(os.path.join(directory, "matrix.npy"), matrix)
        np.save(os.path.join(directory, "snapshot.npy"), snapshot)
        np.save(os.path.join(directory, "sensor_positions_x.npy"), sensor_positions_x)
        np.save(os.path.join(directory, "sensor_positions_y.npy"), sensor_positions_y)
        np.save(os.path.join(directory, "train_error.npy"), train_error)
        np.save(os.path.join(directory, "validation_errors.npy"), validation_errors)
        
        print(f"Results saved in {directory}")

    save_to_numpy(test_recons, test_ground_truth, matrix, snapshot, sensor_positions_x, sensor_positions_y, train_error, validation_errors, model)

else:
    
    def save_to_numpy(test_recons, test_ground_truth, matrix, snapshot, sensor_positions_x, sensor_positions_y, validation_errors, model, directory=save_path):
        # Verifica se o diretório 'results' existe, senão cria
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Salva cada conjunto de dados em um arquivo .npy separado
        np.save(os.path.join(directory, "test_recons.npy"), test_recons)
        np.save(os.path.join(directory, "test_ground_truth.npy"), test_ground_truth)
        np.save(os.path.join(directory, "matrix.npy"), matrix)
        np.save(os.path.join(directory, "snapshot.npy"), snapshot)
        np.save(os.path.join(directory, "sensor_positions_x.npy"), sensor_positions_x)
        np.save(os.path.join(directory, "sensor_positions_y.npy"), sensor_positions_y)
        np.save(os.path.join(directory, "validation_errors.npy"), validation_errors)
        
        print(f"Results saved in {directory}")

    save_to_numpy(test_recons, test_ground_truth, matrix, snapshot, sensor_positions_x, sensor_positions_y, validation_errors, model)





