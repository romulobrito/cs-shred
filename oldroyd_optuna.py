import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
import optuna
import json
import os

import models
from processdata import TimeSeriesDataset

# Verifica a disponibilidade de GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Função para carregar dados de um arquivo .npy
def load_data(npy_file_path, time_slice):
    data_array = np.load(npy_file_path)
    data_array = data_array[:,:,: time_slice]
    data_array = np.transpose(data_array, (2,0,1))
    print("Loaded data dimensions:", data_array.shape)
    return data_array


# Visualização dos dados 2D ou 3D
def visualize_data(matrix, subsampled, results_dir):
    plt.imshow(matrix[-1, :, :], cmap="Spectral", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    plt.savefig(os.path.join(results_dir, f"last_temporal_slice.png"))

    plt.imshow(subsampled[:, :, -1], cmap="Spectral", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice (Subsampled)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    plt.savefig(os.path.join(results_dir, f"last_temporal_subsampled_slice.png"))



# def subsample(snapshot, percent_subsample):
#     """
#     Realiza a subamostragem de um snapshot tridimensional, preservando regras específicas.
    
#     Parâmetros de entrada:
#     snapshot: numpy array, snapshot tridimensional
#     percent_subsample: float, percentual de subamostragem desejado
    
#     Retorna:
#     snapshot_subsampled: numpy array, snapshot subamostrado
#     """
#     num_cols_subsample = int(snapshot.shape[1] * percent_subsample)
#     snapshot_subsampled = snapshot.copy()

#     for i in range(snapshot.shape[2]):
#         random_indices = np.random.choice(
#             snapshot.shape[1], size=num_cols_subsample, replace=False
#         )

#         # Garante que os extremos da lista não sejam subamostrados e não há lacunas maiores que 3 índices
#         random_indices = sorted(random_indices)
#         diff_indices = np.diff(random_indices)
#         for j in range(len(random_indices) - 1):
#             if diff_indices[j] > 4:
#                 random_indices[j + 1] = random_indices[j] + 3

#         snapshot_subsampled[:, random_indices, i] = 0

#     return snapshot_subsampled

def subsample(snapshot, num_cols_subsample, num_snapshots_subsample):
    np.random.seed(1001)

    print('snapshot', snapshot.shape)

    snapshot = np.transpose(snapshot, (1, 2, 0))
    # print('snapshot after transpose', snapshot.shape)
    dim_x, dim_y, dim_t = snapshot.shape
    snapshot_subsampled = snapshot.copy()

    # Garantir que num_snapshots_subsample seja menor que dim_t
    num_snapshots_subsample = min(num_snapshots_subsample, dim_t - 1)
    
    # Escolha aleatória das colunas a serem subamostradas
    # Garantindo que não subamostre todas as colunas
    num_cols_subsample = min(num_cols_subsample, dim_y - 1)  # Sempre manter pelo menos uma coluna
    cols_to_subsample = np.random.choice(dim_y, size=num_cols_subsample, replace=False)
    cols_to_subsample = np.sort(cols_to_subsample)

    # Escolha aleatória dos snapshots, excluindo o último inicialmente
    available_snapshots = np.arange(dim_t - 1)
    snapshots_to_subsample = np.random.choice(
        available_snapshots, 
        size=num_snapshots_subsample - 1, 
        replace=False
    )
    # Adiciona o último snapshot
    snapshots_to_subsample = np.append(snapshots_to_subsample, dim_t - 1)
    snapshots_to_subsample = np.sort(snapshots_to_subsample)

    # Criar uma máscara inicialmente False (manter todos os dados)
    mask = np.zeros((dim_x, dim_y, dim_t), dtype=bool)

    # Aplicar subamostragem apenas nas colunas e snapshots selecionados
    for t in snapshots_to_subsample:
        mask[:, cols_to_subsample, t] = True

    # Verificar se não estamos zerando dados demais
    total_points = dim_x * dim_y * dim_t
    masked_points = np.sum(mask)
    if masked_points / total_points > 0.95:  # Se mais de 95% dos pontos forem mascarados
        print("Aviso: Muitos pontos sendo mascarados. Ajustando...")
        return snapshot_subsampled  # Retorna sem subamostragem

    # Aplicar a máscara
    snapshot_subsampled[mask] = 0

    # Verificação final
    if np.all(snapshot_subsampled == 0):
        print("Aviso: Todos os valores foram zerados. Retornando dados originais...")
        return snapshot

    print("Forma do snapshot após subamostragem:", snapshot_subsampled.shape)
    print(f"Porcentagem de dados mantidos: {100 * (np.sum(mask)/mask.size):.2f}%")
    return snapshot_subsampled


# Configuração dos sensores
def plot_dynamics_at_sensors(
    trace_A, num_sensors, trial_number, results_dir, locations="c", show_plot=False, seed=101
):
    np.random.seed(seed)

    dim_x, dim_y, dim_t = trace_A.shape

    print("Forma do snapshot após subamostragem:", trace_A.shape)

    if locations == "a":
        central_x = 0.5
        central_y = 0.5
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

        sensor_positions_x = np.append(sensor_positions_x, central_x)
        sensor_positions_y = np.append(sensor_positions_y, central_y)

    elif locations == "b":
        sensor_positions_x = [0.5]
        sensor_positions_y = [0.5]

    else:
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

    # sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
    # sensor_positions_x = sensor_locations % dim_x / dim_x
    # sensor_positions_y = sensor_locations // dim_y / dim_y


    sensor_temperature_history = []
    for t in range(dim_t):
        sensor_temperatures = [
            trace_A[int(dim_x * x), int(dim_y * y), t]
            for x, y in zip(sensor_positions_x, sensor_positions_y)
        ]
        sensor_temperature_history.append(sensor_temperatures)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.linspace(0, 1, int(dim_x))
    y = np.linspace(0, 1, int(dim_y))
    X, Y = np.meshgrid(x, y)

    cmap = ax1.pcolormesh(X, Y, trace_A[:, :, -1].real, shading="auto", cmap="hot")
    fig.colorbar(cmap, ax=ax1, label=r"$tr(C)$")
    ax1.set_title("Espatial Distribution oF The $tr(C)$ Oldroyd-B Tensor")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax1.scatter(
        sensor_positions_x, sensor_positions_y, color="blue", label="Sensor Positions"
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
    plt.show()
    plt.savefig(os.path.join(results_dir, f"{trial_number}_{num_sensors}_sensors_dynamics.png"))
    plt.close()

    return sensor_locations, sensor_positions_x, sensor_positions_y




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
            verbose=verbose,
            patience=patience,
        )
        return validation_errors



def evaluate_model(model, test_dataset, test_dataset_test, sc):
    test_recons = sc.inverse_transform(model(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    test_ground_truth_test = sc.inverse_transform(test_dataset_test.Y.detach().cpu().numpy())
    
    # Check the dimensions of the arrays
    if test_recons.ndim != test_ground_truth_test.ndim:
        raise ValueError(
            "The dimensions of the reconstructed and ground truth data do not correspond."
        )
    
    # Calculate the normalized error using original data
    error_norm = np.linalg.norm(test_recons - test_ground_truth_test) / np.linalg.norm(test_ground_truth_test)
    
    # Calculate the SSIM for each snapshot using original data
    ssim_scores = []
    for i in range(test_recons.shape[0]):
        ssim_score = ssim(
            test_ground_truth_test[i],
            test_recons[i],
            data_range=test_ground_truth_test[i].max() - test_ground_truth_test[i].min(),
        )
        ssim_scores.append(ssim_score)
    
    # SSIM médio de todas as amostras
    mean_ssim = np.mean(ssim_scores)
    
    # SSIM do último snapshot (mais importante para avaliar qualidade final)
    last_snapshot_ssim = ssim_scores[-1]  # Último elemento da lista
    
    print("Normalized Error:", error_norm)
    print("Mean SSIM (all samples):", mean_ssim)
    print("Last Snapshot SSIM:", last_snapshot_ssim)
    
    return test_recons, test_ground_truth, test_ground_truth_test, error_norm, mean_ssim, last_snapshot_ssim


def objective(trial):
    # Caminho do arquivo .npy
    npy_file_path =  r'/home/romulo/Documentos/cs-shred-turb-flow/turb-rot-csshred/data/turb_vy_combined.npy'
    # Criação do diretório para armazenar os resultados
    results_dir = r"/home/romulo/Documentos/cs-shred-turb-flow/turb-rot-csshred/results/csshred/turb"
    os.makedirs(results_dir, exist_ok=True)

    # Carregamento dos dados
    matrix = load_data(npy_file_path, time_slice=650)
    

    # model_type = 'CSSHRED'
    model_type = 'SHRED'
    seed = 915

    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    hidden_layers = trial.suggest_categorical("hidden_layers", [1, 2, 3])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    lambL2 = trial.suggest_float("lambL2", 1e-5, 1, log=True)
    lambL1 = trial.suggest_float("lambL1", 1e-5, 1, log=True)
    lambdaSNR = trial.suggest_float("lambdaSNR", 1e-5, 1, log=True)
    l1 = trial.suggest_categorical("l1", [300, 400, 500])
    l2 = trial.suggest_categorical("l2", [300, 400, 500])
    lags = trial.suggest_categorical("lags", [15, 20, 40])
    num_sensors = trial.suggest_categorical("num_sensors", [5])
    num_epochs = trial.suggest_int("num_epochs", 500, 2000)
    step_epoch = trial.suggest_int("step_epoch", 10, 50)
    # Novos parâmetros para otimização
    l1_tol = trial.suggest_float("l1_tol", 1e-5, 1, log=True)
    opt_tol = trial.suggest_float("opt_tol", 1e-5, 1, log=True)
    ls_tol = trial.suggest_float("ls_tol", 1e-5, 1, log=True)
    dropout = trial.suggest_float("dropout", 0.01, 0.011) 
    

    # Adicionar print dos novos parâmetros
    print("l1_tol=", l1_tol)
    print("opt_tol=", opt_tol)
    print("ls_tol=", ls_tol)
    print("dropout=", dropout)  

    print("hidden_size=", hidden_size)
    print("hidden_layers=", hidden_layers)
    print("batch_size=", batch_size)
    print("lr=", lr)
    print("lambL2=", lambL2)
    print("lambL1=", lambL1)
    print("lambdaSNR=", lambdaSNR)
    print("l1=", l1)
    print("l2=", l2)
    print("lags=", lags)
    print("num_sensors=", num_sensors)
    print("num_epochs=", num_epochs)
    print("step_epoch=", step_epoch)


    # Atualize as variáveis globais ou crie novos dados com os novos lags
    global all_data_in, train_data_in, valid_data_in, test_data_in
    global train_data_out, valid_data_out, test_data_out, train_dataset, valid_dataset, test_dataset

    # Subamostragem e visualização dos dados
    snapshot = matrix.copy()
    num_cols_subsample = int(snapshot.shape[2] * 0.3)  # % das colunas serão subamostradas
    num_snapshots_subsample = int(snapshot.shape[0] * 0.3)  #  % dos snapshots serão subamostrados
    snapshot = subsample(snapshot, num_cols_subsample, num_snapshots_subsample)
    # visualize_data(matrix,snapshot, results_dir)
    
    sensor_locations, sensor_positions_x, sensor_positions_y = plot_dynamics_at_sensors(
        snapshot, 
        num_sensors,
        trial_number=trial.number,
        results_dir=results_dir,
        locations="c", 
        show_plot=False, 
        seed=seed
    )

    trace_A = snapshot.copy()
    trace_A_ori = matrix.copy()  # CORREÇÃO: usar dados originais sem subamostragem

    print("trace_A", trace_A.shape)
    print("trace_A_ori", trace_A_ori.shape)

    dim_x, dim_y, dim_t = trace_A.shape
    train_size = int(0.7 * trace_A.shape[2])
    val_size = int(0.2 * trace_A.shape[2])
    test_size = int(0.1 * trace_A.shape[2])

    load_X = trace_A.reshape(dim_x * dim_y, dim_t).T
    load_X_test = trace_A_ori.reshape(dim_x * dim_y, dim_t).T

    print("load_X", load_X.shape)
    print("load_X_test", load_X_test.shape)

    n = load_X.shape[0]
    m = load_X.shape[1]
    print(n, m)

    if n - lags <= 0:
        raise ValueError("Invalid lags: exceeds data length.")

    total_size = train_size + val_size + test_size
    if total_size > n - lags:
        train_size = (train_size * (n - lags)) // total_size
        val_size = (val_size * (n - lags)) // total_size
        test_size = (test_size * (n - lags)) // total_size

    train_indices = np.random.choice(n - lags, size=train_size, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]

    valid_indices = valid_test_indices[:val_size]
    test_indices = valid_test_indices[val_size : val_size + test_size]

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    # CORREÇÃO: usar o mesmo scaler para evitar data leakage
    transformed_X_test = sc.transform(load_X_test)

    all_data_in = np.zeros((n - lags, lags, num_sensors))
    all_data_in_test = np.zeros((n - lags, lags, num_sensors))
    for i in range(n - lags):
        for j, loc in enumerate(sensor_locations):
            all_data_in[i, :, j] = transformed_X[i : i + lags, loc]
            all_data_in_test[i, :, j] = transformed_X_test[i : i + lags, loc]

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    if model_type == 'CSSHRED':
        model = models.CSSHRED(
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
            l1_tol=l1_tol,
            opt_tol=opt_tol, 
            ls_tol=ls_tol,
            verbosity=0,
            show_plot=False,
        ).to(device)

        train_error, validation_errors = models.fit_csshred_model(
            model,
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,  # Incluindo num_epochs
            lr=lr,
            lambL2=lambL2,
            step_epoch=step_epoch,
            lambL1=lambL1,
            lambdaSNR=lambdaSNR,
            verbose=False,
            patience=15,
        )
    else:
        model = models.SHRED(  # 64
            num_sensors,
            m,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            l1=l1,
            l2=l2,
            dropout=dropout,
        ).to(device)
        validation_errors = models.fit(
            model,
            train_dataset,
            valid_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            step_epoch=step_epoch,
            verbose=False,
            patience=15,
        )
    
    test_recons = sc.inverse_transform(model(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    test_ground_truth_test = sc_test.inverse_transform(
        test_dataset_test.Y.detach().cpu().numpy()
    )
    print("test_recons shape:", test_recons.shape)
    print("test_ground_truth_test shape:", test_ground_truth_test.shape)


    
    # Convertendo valores para tipos serializáveis antes de salvar
    validation_errors = [float(val) for val in validation_errors]

    # ssim_score = calculate_ssim(test_recons, test_ground_truth_test)  

    test_recons, test_ground_truth, test_ground_truth_test, error_norm, mean_ssim, last_snapshot_ssim = evaluate_model(model, test_dataset, test_dataset_test, sc)

    with open(os.path.join(results_dir, f"{trial.number}_results.json"), "w") as f:
        results = {
            "trial_number": trial.number,
            "hidden_size": hidden_size,
            "hidden_layers": hidden_layers,
            "batch_size": batch_size,
            "lr": float(lr),
            "lambL2": float(lambL2),
            "lambL1": float(lambL1),
            "lambdaSNR": float(lambdaSNR),
            "dropout": float(dropout), 
            "l1_tol": float(l1_tol),
            "opt_tol": float(opt_tol),
            "ls_tol": float(ls_tol),
            "l1": l1,
            "l2": l2,
            "lags": lags,
            "num_sensors": num_sensors,
            "num_epochs": num_epochs,  
            "step_epoch": step_epoch,
            "error_norm": float(error_norm),
            "validation_errors": np.mean(validation_errors),           
            "ssim_score_mean": float(mean_ssim),
            "ssim_score_last": float(last_snapshot_ssim),
            
        }
        json.dump(results, f, indent=4)

    # Função objetivo multicriterial: combina erro normalizado e SSIM
    # Objetivo: minimizar erro E maximizar SSIM
    # Formula: error_norm + (1 - mean_ssim) 
    # Peso para SSIM pode ser ajustado (alpha)
    alpha = 0.5  # peso para SSIM (0.5 = igual importância)
    
    # Normalizando os valores para ter escalas similares
    # error_norm já está entre 0-1, SSIM também está entre 0-1
    objective_value = alpha * error_norm + (1 - alpha) * (1 - mean_ssim)
    
    print(f"Objective components - Error: {error_norm:.4f}, SSIM: {mean_ssim:.4f}")
    print(f"Combined Objective: {objective_value:.4f}")
    
    # Adicionar o valor objetivo ao JSON
    with open(os.path.join(results_dir, f"{trial.number}_results.json"), "r") as f:
        results = json.load(f)
    
    results["objective_value"] = float(objective_value)
    results["objective_alpha"] = alpha
    
    with open(os.path.join(results_dir, f"{trial.number}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return float(objective_value)


# Early stopping callback for Optuna
class EarlyStoppingCallback:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.counter = 0

    def __call__(self, study, trial):
        if self.best_value is None or study.best_value < self.best_value:
            self.best_value = study.best_value
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"Early stopping triggered: no improvement in {self.patience} trials.")
            study.stop()

# Configuração do estudo do Optuna 
study = optuna.create_study(direction="minimize")
early_stopping = EarlyStoppingCallback(patience=10)
study.optimize(objective, n_trials=55, callbacks=[early_stopping])

# Salvando os resultados do estudo
results_dir = r"/home/romulo/Documentos/cs-shred-turb-flow/turb-rot-csshred/results/csshred/turb"
os.makedirs(results_dir, exist_ok=True)

# Extrair e imprimir o melhor ensaio
print("Best trial:")
trial = study.best_trial

# Obter os parâmetros do melhor ensaio
best_params = {
    "trial_number": trial.number,
    "hidden_size": trial.params.get("hidden_size"),
    "hidden_layers": trial.params.get("hidden_layers"),
    "batch_size": trial.params.get("batch_size"),
    "lr": float(trial.params.get("lr")),
    "lambL2": float(trial.params.get("lambL2")),
    "lambL1": float(trial.params.get("lambL1")),
    "lambdaSNR": float(trial.params.get("lambdaSNR")),
    "l1": trial.params.get("l1"),
    "l2": trial.params.get("l2"),
    "lags": trial.params.get("lags"),
    "num_sensors": trial.params.get("num_sensors"),
    "num_epochs": trial.params.get("num_epochs"),
    "dropout": float(trial.params.get("dropout")), 
    "l1_tol": float(trial.params.get("l1_tol")),
    "opt_tol": float(trial.params.get("opt_tol")),
    "ls_tol": float(trial.params.get("ls_tol")),
    "step_epoch": trial.params.get("step_epoch"),
}

# Adicionar os resultados das métricas
results_file = os.path.join(results_dir, f"{trial.number}_results.json")
with open(results_file, "r") as f:
    results = json.load(f)

best_params.update({
    "error_norm": results.get("error_norm"),
    "ssim_score_mean": results.get("ssim_score_mean"),
    "ssim_score_last": results.get("ssim_score_last"),
})

# Salvar os parâmetros do melhor ensaio em um arquivo JSON
with open(os.path.join(results_dir, "best_trial_params.json"), "w") as f:
    json.dump(best_params, f, indent=4)

print("Best trial parameters and metrics saved to 'best_trial_params.json'")

