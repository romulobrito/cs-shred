import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from math import log10
import optuna
import json
import os
from datetime import datetime

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
def visualize_data(matrix, subsampled, results_dir, save_plots=True):
    """
    Visualize the last temporal slice of the original and subsampled data arrays,
    saving the plots as PNG files.
    
    Parameters
    ----------
    matrix : np.ndarray
        Original data array of shape (time, x, y).
    subsampled : np.ndarray
        Subsampled data array of shape (x, y, time).
    results_dir : str
        Directory to save the plots.
    save_plots : bool
        Whether to save the plots to disk.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix[-1, :, :], cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice")
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_plots:
        plt.savefig(os.path.join(results_dir, "last_temporal_slice.png"))
        plt.savefig(os.path.join(results_dir, "last_temporal_slice.pdf"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.imshow(subsampled[:, :, -1], cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("Last Temporal Slice (Subsampled)")
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_plots:
        plt.savefig(os.path.join(results_dir, "last_temporal_subsampled_slice.png"))
        plt.savefig(os.path.join(results_dir, "last_temporal_subsampled_slice.pdf"))
    plt.close()



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
    trace_A, num_sensors, locations="c", show_plot=False, seed=101,
    save_plot=False, save_path=None, file_name="plot_din.png", model_type="CS-SHRED"
):
    """
    Select sensor locations and optionally plot their temporal dynamics and spatial positions.
    
    Parameters
    ----------
    trace_A : np.ndarray
        3D array (x, y, time) representing the field.
    num_sensors : int
        Number of sensors to select.
    locations : str
        Sensor placement strategy ('a', 'b', or 'c').
    show_plot : bool
        Whether to display the plot interactively.
    seed : int
        Random seed for reproducibility.
    save_plot : bool
        Whether to save the plot to disk.
    save_path : str
        Directory to save the plot.
    file_name : str
        Name of the plot file.
    model_type : str
        Model type for file naming.
    
    Returns
    -------
    sensor_locations : np.ndarray
        Indices of selected sensor locations.
    sensor_positions_x : np.ndarray
        X coordinates of sensors (normalized).
    sensor_positions_y : np.ndarray
        Y coordinates of sensors (normalized).
    """
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
        sensor_locations = None

    else:
        sensor_locations = np.random.choice(dim_x * dim_y, num_sensors, replace=False)
        sensor_positions_x = sensor_locations % dim_x / dim_x
        sensor_positions_y = sensor_locations // dim_y / dim_y

    # Plot and save if requested
    if save_plot and save_path:
        sensor_temperature_history = []
        for t in range(dim_t):
            sensor_temperatures = [
                trace_A[int(dim_x * x), int(dim_y * y), t]
                for x, y in zip(sensor_positions_x, sensor_positions_y)
            ]
            sensor_temperature_history.append(sensor_temperatures)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        y = np.linspace(0, 1, int(dim_x))
        x = np.linspace(0, 1, int(dim_y))
        X, Y = np.meshgrid(x, y)

        cmap = ax1.pcolormesh(
            X, Y, trace_A[:, :, -1].real, shading="auto", cmap="viridis"
        )
        fig.colorbar(cmap, ax=ax1, label=r"$|v_y|$")
        ax1.set_title("Velocity field $|v_y|$")
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
        ax2.set_ylabel("Amplitude Velocity $v_y$")
        ax2.set_title("Dynamics at Sensor Positions")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, file_name)
        plt.savefig(save_file)
        plt.savefig(save_file.replace('.png', '.pdf'))
        print(f"Plot saved to {save_file}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

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
    generator=None,
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
            generator=generator,  # Pass generator for reproducibility
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
            generator=generator,  # Pass generator for reproducibility
        )
        return validation_errors



def evaluate_model(model, test_dataset, test_dataset_test, sc, sc_test):
    test_recons = sc.inverse_transform(model(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    test_ground_truth_test = sc_test.inverse_transform(test_dataset_test.Y.detach().cpu().numpy())
    
    error_norm = np.linalg.norm(test_recons - test_ground_truth_test) / np.linalg.norm(test_ground_truth_test)
    
    # Calcular o SSIM global
    ssim_score = ssim(test_ground_truth_test, test_recons, data_range=test_recons.max() - test_recons.min())
    
    # Calcular metricas do ultimo snapshot para visibilidade (matching plot_turb.ipynb)
    last_snapshot_idx = test_recons.shape[0] - 1
    last_ground_truth = test_ground_truth_test[last_snapshot_idx]
    last_reconstruction = test_recons[last_snapshot_idx]
    
    # SSIM do ultimo snapshot
    ssim_last_snapshot = ssim(
        last_ground_truth,
        last_reconstruction,
        data_range=last_ground_truth.max() - last_ground_truth.min()
    )
    
    # PSNR do ultimo snapshot (matching plot_turb.ipynb)
    mse_last = mean_squared_error(last_ground_truth.flatten(), last_reconstruction.flatten())
    max_pixel_last = np.max(last_ground_truth)
    psnr_last_snapshot = 20 * log10(max_pixel_last / np.sqrt(mse_last)) if mse_last > 0 else float('inf')
    
    # Normalized Error do ultimo snapshot (matching plot_turb.ipynb)
    error_norm_last_snapshot = np.linalg.norm(last_reconstruction - last_ground_truth) / np.linalg.norm(last_ground_truth)
    
    print("Normalized Error (global):", error_norm)
    print("SSIM (global):", ssim_score)
    print("SSIM (last snapshot):", ssim_last_snapshot)
    print("PSNR (last snapshot):", psnr_last_snapshot, "dB")
    print("Normalized Error (last snapshot):", error_norm_last_snapshot)
    
    return test_recons, test_ground_truth, test_ground_truth_test, error_norm, ssim_score, ssim_last_snapshot, psnr_last_snapshot, error_norm_last_snapshot


def objective(trial):
    # Caminho do arquivo .npy
    npy_file_path =  r'/home/romulo/Documentos/lpips-env/codigos_para_gerar_imagem_cs-shred/data/turb_vy_combined.npy'
    # Usar o diretório global criado antes do estudo
    global global_results_dir
    results_dir = global_results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Carregamento dos dados
    matrix = load_data(npy_file_path, time_slice=650)
    

    # model_type = 'CSSHRED'
    model_type = 'CS-SHRED'
    seed = 915
    
    # ============================================================================
    # REPRODUCIBILITY: Set all random seeds for deterministic results
    # ============================================================================
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set generator for DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)

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
        locations="c", 
        show_plot=False, 
        seed=seed
    )

    trace_A = snapshot.copy()
    # trace_A_ori deve usar matrix (dados ORIGINAIS), não snapshot (subamostrado)
    # Isso garante que test_ground_truth_test venha de dados originais completos
    # matrix tem formato (time, x, y), precisa fazer transpose para (x, y, time) como snapshot
    trace_A_ori = np.transpose(matrix.copy(), (1, 2, 0))

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

    # Set seed before random choice to ensure reproducibility
    np.random.seed(seed)
    
    # Garantir que o último snapshot do dataset completo (índice n-lags-1) 
    # esteja sempre no conjunto de teste para comparar com o original completo
    last_snapshot_idx = n - lags - 1  # Índice do último snapshot no espaço de índices válidos
    
    # Excluir o último snapshot do conjunto de treinamento
    available_indices = np.arange(0, n - lags)
    available_indices = available_indices[available_indices != last_snapshot_idx]
    
    train_indices = np.random.choice(available_indices, size=train_size, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    mask[last_snapshot_idx] = 0  # Garantir que o último snapshot não seja usado em train
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]

    valid_indices = valid_test_indices[:val_size]
    # Garantir que o último snapshot esteja no conjunto de teste
    test_indices = valid_test_indices[val_size : val_size + test_size]
    if last_snapshot_idx not in test_indices:
        # Se o último snapshot não está no teste, substituir o último índice do teste
        if len(test_indices) > 0:
            test_indices = test_indices[:-1]
        test_indices = np.append(test_indices, last_snapshot_idx)

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    sc_test = MinMaxScaler()
    sc_test = sc_test.fit(load_X_test[train_indices])
    transformed_X_test = sc_test.transform(load_X_test)

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
    # test_data_in_test deve usar all_data_in (SUBAMOSTRADO), não all_data_in_test (ORIGINAL)
    # A entrada do modelo deve ser sempre subamostrada, mesmo no teste
    # Apenas a saída (test_data_out_test) deve usar dados originais para comparação
    test_data_in_test = torch.tensor(
        all_data_in[test_indices], dtype=torch.float32
    ).to(device)

    # Durante o treinamento e validação, o modelo deve comparar com dados ORIGINAIS
    # para aprender a reconstruir as informações ausentes
    # train_data_out e valid_data_out devem vir de transformed_X_test (ORIGINAL), não de transformed_X (SUBAMOSTRADO)
    train_data_out = torch.tensor(
        transformed_X_test[train_indices + lags - 1], dtype=torch.float32
    ).to(device)
    valid_data_out = torch.tensor(
        transformed_X_test[valid_indices + lags - 1], dtype=torch.float32
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

    if model_type == 'CS-SHRED':
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
            generator=generator,  # Pass generator for reproducibility
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
            generator=generator,  # Pass generator for reproducibility
        )
    
    # Convertendo valores para tipos serializáveis antes de salvar
    validation_errors = [float(val) for val in validation_errors]

    # Avaliar o modelo (faz inverse_transform internamente)
    test_recons, test_ground_truth, test_ground_truth_test, error_norm, ssim_score, ssim_last_snapshot, psnr_last_snapshot, error_norm_last_snapshot = evaluate_model(model, test_dataset, test_dataset_test, sc, sc_test)

    # Salvar artefatos apenas do melhor trial até o momento
    # Compara o trial atual com o melhor anterior e substitui se for melhor
    # Isso economiza espaço em disco mantendo apenas ~500MB (do melhor trial)
    best_trial_artifacts_dir = os.path.join(results_dir, "best_trial_artifacts")
    os.makedirs(best_trial_artifacts_dir, exist_ok=True)
    
    # Compatibilizar métrica do Optuna com a loss de treinamento
    # A loss de treinamento usa principalmente MSE (lambL2 * lossMSE), então a métrica do Optuna
    # deve priorizar error_norm (equivalente ao MSE normalizado) para evitar sub-otimização
    # Removido SSIM porque: 1) não está na loss de treinamento, 2) não captura bem artefatos visuais
    # Mantido error_norm_last_snapshot para penalizar especificamente artefatos no último snapshot
    alpha = 1.0  # Peso do error_norm global (alinhado com lambL2 * lossMSE da loss de treinamento)
    beta = 1.5  # Peso do error_norm_last_snapshot (aumentado para priorizar mais o último snapshot com artefatos visuais)
    # Métrica: alpha*error_norm + beta*error_norm_last_snapshot (minimizar)
    # Isso alinha com a loss de treinamento (MSE) e ainda permite penalizar o último snapshot
    composite_metric = alpha * error_norm + beta * error_norm_last_snapshot
    
    # Verificar se este trial é melhor que o melhor anterior
    best_trial_file = os.path.join(results_dir, "current_best_trial.json")
    is_best = False
    
    if os.path.exists(best_trial_file):
        # Carregar melhor trial anterior
        with open(best_trial_file, 'r') as f:
            best_trial_info = json.load(f)
        best_composite = best_trial_info.get("composite_metric", float('inf'))
        # Comparar: menor métrica composta é melhor (menor = alpha*error_norm + beta*error_norm_last)
        if composite_metric < best_composite:
            is_best = True
            print(f"\n✅ Trial {trial.number} é melhor que o anterior")
            print(f"   Composite metric: {composite_metric:.6f} < {best_composite:.6f}")
            print(f"   SSIM global: {ssim_score:.6f} (anterior: {best_trial_info.get('ssim_score', 0):.6f})")
            print(f"   SSIM last snapshot: {ssim_last_snapshot:.6f} (anterior: {best_trial_info.get('ssim_last_snapshot', 0):.6f})")
            print(f"   Error norm: {error_norm:.6f} (anterior: {best_trial_info.get('error_norm', 0):.6f})")
        else:
            print(f"\n⚠️ Trial {trial.number} não é melhor")
            print(f"   Composite metric: {composite_metric:.6f} >= {best_composite:.6f}")
    else:
        # Primeiro trial é automaticamente o melhor
        is_best = True
        print(f"\n✅ Trial {trial.number} é o primeiro (melhor por padrão)")
        print(f"   Composite metric: {composite_metric:.6f}")
    
    # Salvar artefatos apenas se for o melhor trial até o momento
    if is_best:
        # Remover artefatos do melhor trial anterior (se existir)
        if os.path.exists(best_trial_file):
            print(f"   Removendo artefatos do melhor trial anterior...")
            # Os arquivos já serão sobrescritos, mas podemos limpar explicitamente
            for file in ["test_recons.npy", "test_ground_truth.npy", "matrix.npy", "snapshot.npy",
                        "sensor_positions_x.npy", "sensor_positions_y.npy", "validation_errors.npy", "train_error.npy"]:
                file_path = os.path.join(best_trial_artifacts_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Salvar artefatos do melhor trial atual
        print(f"   Salvando artefatos do melhor trial {trial.number}...")
        np.save(os.path.join(best_trial_artifacts_dir, "test_recons.npy"), test_recons)
        np.save(os.path.join(best_trial_artifacts_dir, "test_ground_truth.npy"), test_ground_truth_test)
        np.save(os.path.join(best_trial_artifacts_dir, "matrix.npy"), matrix)
        np.save(os.path.join(best_trial_artifacts_dir, "snapshot.npy"), snapshot)
        np.save(os.path.join(best_trial_artifacts_dir, "sensor_positions_x.npy"), sensor_positions_x)
        np.save(os.path.join(best_trial_artifacts_dir, "sensor_positions_y.npy"), sensor_positions_y)
        np.save(os.path.join(best_trial_artifacts_dir, "validation_errors.npy"), validation_errors)
        if train_error is not None:
            np.save(os.path.join(best_trial_artifacts_dir, "train_error.npy"), train_error)
        
        # Atualizar arquivo com informações do melhor trial
        # Incluir métrica composta para comparação (alinhada com loss de treinamento)
        alpha = 1.0  # Mesmo alpha usado na comparação
        beta = 1.5  # Mesmo beta usado na comparação (aumentado para priorizar mais o último snapshot)
        composite_metric = alpha * error_norm + beta * error_norm_last_snapshot
        best_trial_info = {
            "trial_number": trial.number,
            "composite_metric": float(composite_metric),  # Métrica usada para seleção
            "error_norm": float(error_norm),
            "ssim_score": float(ssim_score),
            "ssim_last_snapshot": float(ssim_last_snapshot),
            "psnr_last_snapshot": float(psnr_last_snapshot),
            "error_norm_last_snapshot": float(error_norm_last_snapshot),
            "alpha": float(alpha),  # Peso do error_norm global na métrica composta (alinhado com loss de treinamento)
            "beta": float(beta)  # Peso do error_norm_last_snapshot na métrica composta
        }
        with open(best_trial_file, 'w') as f:
            json.dump(best_trial_info, f, indent=4)
    else:
        print(f"   Pulando salvamento de artefatos (não é o melhor trial)")

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
            "ssim_score": float(ssim_score),
            "ssim_last_snapshot": float(ssim_last_snapshot),  # Adicionado para visibilidade
            "psnr_last_snapshot": float(psnr_last_snapshot),  # Adicionado para visibilidade (matching plot_turb.ipynb)
            "error_norm_last_snapshot": float(error_norm_last_snapshot),  # Adicionado para visibilidade (matching plot_turb.ipynb)
            
        }
        json.dump(results, f, indent=4)

    # Compatibilizar métrica do Optuna com a loss de treinamento
    # A loss de treinamento usa principalmente MSE (lambL2 * lossMSE), então a métrica do Optuna
    # deve priorizar error_norm (equivalente ao MSE normalizado) para evitar sub-otimização
    # Removido SSIM porque: 1) não está na loss de treinamento, 2) não captura bem artefatos visuais
    # Mantido error_norm_last_snapshot para penalizar especificamente artefatos no último snapshot
    # alpha: peso do error_norm global (alinhado com lambL2 * lossMSE da loss de treinamento)
    # beta: peso do error_norm_last_snapshot (penaliza especificamente artefatos no último snapshot)
    # Métrica composta: alpha*error_norm + beta*error_norm_last_snapshot (minimizar)
    # Isso alinha com a loss de treinamento (MSE) e ainda permite penalizar o último snapshot
    alpha = 1.0  # Peso do error_norm global (alinhado com loss de treinamento)
    beta = 1.5  # Peso do error_norm_last_snapshot (aumentado para priorizar mais o último snapshot com artefatos visuais)
    composite_metric = alpha * error_norm + beta * error_norm_last_snapshot
    
    print(f"Composite metric ({alpha}*error_norm + {beta}*error_norm_last_snapshot): {composite_metric:.6f}")
    print(f"  Error norm global: {error_norm:.6f}, Error norm last snapshot: {error_norm_last_snapshot:.6f}")
    print(f"  (SSIM global: {ssim_score:.6f}, SSIM last snapshot: {ssim_last_snapshot:.6f} - apenas informativo)")
    
    return float(composite_metric)


def save_best_trial_artifacts(best_trial, results_dir, npy_file_path, model_type='CS-SHRED', seed=915):
    """
    Re-executa o melhor trial e salva todos os artefatos necessários para visualização
    no plot_turb.ipynb (arrays numpy + imagens).
    
    Parameters
    ----------
    best_trial : optuna.Trial
        O melhor trial identificado pelo Optuna.
    results_dir : str
        Diretório onde salvar os artefatos.
    npy_file_path : str
        Caminho para o arquivo .npy com os dados.
    model_type : str
        Tipo de modelo ('CS-SHRED' ou 'SHRED').
    seed : int
        Seed para reprodutibilidade.
    """
    print("\n" + "="*80)
    print("RE-EXECUTANDO MELHOR TRIAL PARA SALVAR ARTEFATOS")
    print("="*80)
    
    # Criar diretório para artefatos do melhor trial
    artifacts_dir = os.path.join(results_dir, "best_trial_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"Salvando artefatos em: {artifacts_dir}")
    
    # ============================================================================
    # REPRODUCIBILITY: Set all random seeds
    # ============================================================================
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Extrair parâmetros do melhor trial
    hidden_size = best_trial.params.get("hidden_size")
    hidden_layers = best_trial.params.get("hidden_layers")
    batch_size = best_trial.params.get("batch_size")
    lr = best_trial.params.get("lr")
    lambL2 = best_trial.params.get("lambL2")
    lambL1 = best_trial.params.get("lambL1")
    lambdaSNR = best_trial.params.get("lambdaSNR")
    l1 = best_trial.params.get("l1")
    l2 = best_trial.params.get("l2")
    lags = best_trial.params.get("lags")
    num_sensors = best_trial.params.get("num_sensors")
    num_epochs = best_trial.params.get("num_epochs")
    step_epoch = best_trial.params.get("step_epoch")
    dropout = best_trial.params.get("dropout")
    l1_tol = best_trial.params.get("l1_tol")
    opt_tol = best_trial.params.get("opt_tol")
    ls_tol = best_trial.params.get("ls_tol")
    
    print(f"\nParâmetros do melhor trial (Trial {best_trial.number}):")
    print(f"  hidden_size: {hidden_size}, hidden_layers: {hidden_layers}")
    print(f"  batch_size: {batch_size}, lr: {lr:.6f}")
    print(f"  lags: {lags}, num_sensors: {num_sensors}, num_epochs: {num_epochs}")
    
    # 1. Carregar dados
    print("\n[1/7] Carregando dados...")
    matrix = load_data(npy_file_path, time_slice=650)
    
    # 2. Subsampling
    print("\n[2/7] Realizando subsampling...")
    snapshot = matrix.copy()
    num_cols_subsample = int(snapshot.shape[2] * 0.3)
    num_snapshots_subsample = int(snapshot.shape[0] * 0.3)
    snapshot = subsample(snapshot, num_cols_subsample, num_snapshots_subsample)
    
    # 3. Visualizar dados (salvar imagens)
    print("\n[3/7] Gerando visualizações dos dados...")
    visualize_data(matrix, snapshot, artifacts_dir, save_plots=True)
    
    # 4. Configurar sensores (salvar imagem)
    print("\n[4/7] Configurando sensores...")
    sensor_locations, sensor_positions_x, sensor_positions_y = plot_dynamics_at_sensors(
        snapshot,
        num_sensors,
        locations="c",
        show_plot=False,
        seed=seed,
        save_plot=True,
        save_path=artifacts_dir,
        file_name=f"plot_din_{model_type}.png",
        model_type=model_type
    )
    
    # 5. Preparar datasets
    print("\n[5/7] Preparando datasets...")
    trace_A = snapshot.copy()  # Subsampled data (x, y, time)
    # Original data needs to be transposed to (x, y, time) format
    trace_A_ori = np.transpose(matrix, (1, 2, 0))  # Original data (x, y, time)
    
    dim_x, dim_y, dim_t = trace_A.shape
    train_size = int(0.7 * trace_A.shape[2])
    val_size = int(0.2 * trace_A.shape[2])
    test_size = int(0.1 * trace_A.shape[2])
    
    load_X = trace_A.reshape(dim_x * dim_y, dim_t).T
    load_X_test = trace_A_ori.reshape(dim_x * dim_y, dim_t).T
    
    n = load_X.shape[0]
    m = load_X.shape[1]  # Number of spatial points (flattened)
    
    total_size = train_size + val_size + test_size
    if total_size > n - lags:
        train_size = (train_size * (n - lags)) // total_size
        val_size = (val_size * (n - lags)) // total_size
        test_size = (test_size * (n - lags)) // total_size
    
    np.random.seed(seed)
    
    # Garantir que o último snapshot do dataset completo (índice n-lags-1) 
    # esteja sempre no conjunto de teste para comparar com o original completo
    last_snapshot_idx = n - lags - 1  # Índice do último snapshot no espaço de índices válidos
    
    # Excluir o último snapshot do conjunto de treinamento
    available_indices = np.arange(0, n - lags)
    available_indices = available_indices[available_indices != last_snapshot_idx]
    
    train_indices = np.random.choice(available_indices, size=train_size, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    mask[last_snapshot_idx] = 0  # Garantir que o último snapshot não seja usado em train
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
    
    valid_indices = valid_test_indices[:val_size]
    # Garantir que o último snapshot esteja no conjunto de teste
    test_indices = valid_test_indices[val_size : val_size + test_size]
    if last_snapshot_idx not in test_indices:
        # Se o último snapshot não está no teste, substituir o último índice do teste
        if len(test_indices) > 0:
            test_indices = test_indices[:-1]
        test_indices = np.append(test_indices, last_snapshot_idx)
    
    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)
    
    sc_test = MinMaxScaler()
    sc_test = sc_test.fit(load_X_test[train_indices])
    transformed_X_test = sc_test.transform(load_X_test)
    
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    all_data_in_test = np.zeros((n - lags, lags, num_sensors))
    for i in range(n - lags):
        for j, loc in enumerate(sensor_locations):
            all_data_in[i, :, j] = transformed_X[i : i + lags, loc]
            all_data_in_test[i, :, j] = transformed_X_test[i : i + lags, loc]
    
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
    # test_data_in_test deve usar all_data_in (SUBAMOSTRADO), não all_data_in_test (ORIGINAL)
    # A entrada do modelo deve ser sempre subamostrada, mesmo no teste
    # Apenas a saída (test_data_out_test) deve usar dados originais para comparação
    test_data_in_test = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
    
    # Durante o treinamento e validação, o modelo deve comparar com dados ORIGINAIS
    # para aprender a reconstruir as informações ausentes
    train_data_out = torch.tensor(transformed_X_test[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X_test[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out_test = torch.tensor(transformed_X_test[test_indices + lags - 1], dtype=torch.float32).to(device)
    
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    test_dataset_test = TimeSeriesDataset(test_data_in_test, test_data_out_test)
    
    # 6. Treinar modelo
    print("\n[6/7] Treinando modelo...")
    if model_type == 'CS-SHRED':
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
            num_epochs=num_epochs,
            lr=lr,
            lambL2=lambL2,
            step_epoch=step_epoch,
            lambL1=lambL1,
            lambdaSNR=lambdaSNR,
            verbose=True,  # Mostrar progresso
            patience=15,
            generator=generator,  # Pass generator for reproducibility
        )
    else:
        model = models.SHRED(
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
            verbose=True,
            patience=15,
            generator=generator,  # Pass generator for reproducibility
        )
        train_error = None  # SHRED não retorna train_error
    
    validation_errors = [float(val) for val in validation_errors]
    
    # 7. Avaliar modelo
    print("\n[7/7] Avaliando modelo e salvando artefatos...")
    test_recons, test_ground_truth, test_ground_truth_test, error_norm, ssim_score, ssim_last_snapshot, psnr_last_snapshot, error_norm_last_snapshot = evaluate_model(
        model, test_dataset, test_dataset_test, sc, sc_test
    )
    
    # Salvar todos os arrays numpy
    print("\nSalvando arrays numpy...")
    np.save(os.path.join(artifacts_dir, "test_recons.npy"), test_recons)
    np.save(os.path.join(artifacts_dir, "test_ground_truth.npy"), test_ground_truth_test)  # Original data
    np.save(os.path.join(artifacts_dir, "matrix.npy"), matrix)
    np.save(os.path.join(artifacts_dir, "snapshot.npy"), snapshot)
    np.save(os.path.join(artifacts_dir, "sensor_positions_x.npy"), sensor_positions_x)
    np.save(os.path.join(artifacts_dir, "sensor_positions_y.npy"), sensor_positions_y)
    np.save(os.path.join(artifacts_dir, "validation_errors.npy"), validation_errors)
    if train_error is not None:
        np.save(os.path.join(artifacts_dir, "train_error.npy"), train_error)
    
    print("\n" + "="*80)
    print(f"ARTEFATOS SALVOS COM SUCESSO EM: {artifacts_dir}")
    print("="*80)
    print("\nArquivos salvos:")
    print("  - test_recons.npy")
    print("  - test_ground_truth.npy")
    print("  - matrix.npy")
    print("  - snapshot.npy")
    print("  - sensor_positions_x.npy")
    print("  - sensor_positions_y.npy")
    print("  - validation_errors.npy")
    if train_error is not None:
        print("  - train_error.npy")
    print("  - last_temporal_slice.png / .pdf")
    print("  - last_temporal_subsampled_slice.png / .pdf")
    print(f"  - plot_din_{model_type}.png / .pdf")
    print(f"\nMétricas finais:")
    print(f"  Normalized Error: {error_norm:.6f}")
    print(f"  SSIM (global): {ssim_score:.6f}")
    print(f"  SSIM (last snapshot): {ssim_last_snapshot:.6f}")
    print(f"  PSNR (last snapshot): {psnr_last_snapshot:.2f} dB")
    print("="*80 + "\n")


# Configuração do estudo do Optuna
# Criar diretório base para resultados com timestamp
base_results_dir = r"/home/romulo/Documentos/lpips-env/results/turb_optuna"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(base_results_dir, f"test_{timestamp}")
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Variável global para o diretório de resultados (usada dentro de objective)
global_results_dir = results_dir

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

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

# Incluir composite_metric no best_trial_params.json
# Carregar composite_metric do current_best_trial.json se existir
current_best_file = os.path.join(results_dir, "current_best_trial.json")
composite_metric = None
alpha = 1.0
beta = 1.5
if os.path.exists(current_best_file):
    with open(current_best_file, 'r') as f:
        current_best_info = json.load(f)
    composite_metric = current_best_info.get("composite_metric")
    alpha = current_best_info.get("alpha", 1.0)
    beta = current_best_info.get("beta", 0.5)

best_params.update({
    "error_norm": results.get("error_norm"),
    "ssim_score": results.get("ssim_score"),
    "ssim_last_snapshot": results.get("ssim_last_snapshot"),  # Adicionado para visibilidade
    "psnr_last_snapshot": results.get("psnr_last_snapshot"),  # Adicionado para visibilidade (matching plot_turb.ipynb)
    "error_norm_last_snapshot": results.get("error_norm_last_snapshot"),  # Adicionado para visibilidade (matching plot_turb.ipynb)
    "composite_metric": float(composite_metric) if composite_metric is not None else None,  # Métrica usada para seleção (alinhada com loss de treinamento)
    "alpha": float(alpha),  # Peso do error_norm global na métrica composta (alinhado com loss de treinamento)
    "beta": float(beta)  # Peso do error_norm_last_snapshot na métrica composta
})

# Salvar os parâmetros do melhor ensaio em um arquivo JSON
with open(os.path.join(results_dir, "best_trial_params.json"), "w") as f:
    json.dump(best_params, f, indent=4)

print("Best trial parameters and metrics saved to 'best_trial_params.json'")

# Copiar arrays do melhor trial (salvos durante a otimização) para best_trial_artifacts
# Os artefatos do melhor trial já foram salvos durante a otimização
# (substituindo incrementalmente a cada trial melhor)
# Agora apenas geramos as imagens e salvamos as métricas finais
print("\n" + "="*80)
print("GERANDO IMAGENS E MÉTRICAS DO MELHOR TRIAL")
print("="*80)
seed = 915  # Mesmo seed usado durante a otimização
best_trial_artifacts_dir = os.path.join(results_dir, "best_trial_artifacts")
os.makedirs(best_trial_artifacts_dir, exist_ok=True)

# Carregar informações do melhor trial
best_trial_file = os.path.join(results_dir, "current_best_trial.json")
if os.path.exists(best_trial_file):
    with open(best_trial_file, 'r') as f:
        best_trial_info = json.load(f)
    best_trial_number = best_trial_info.get("trial_number")
    print(f"Melhor trial identificado: {best_trial_number}")
    print(f"Error norm: {best_trial_info.get('error_norm', 'N/A')}")
    print(f"SSIM: {best_trial_info.get('ssim_score', 'N/A')}")
else:
    print("⚠️ Arquivo current_best_trial.json não encontrado!")
    print("   Usando o melhor trial identificado pelo Optuna")
    best_trial_number = trial.number

# Verificar se os artefatos do melhor trial existem
# (já foram salvos durante a otimização com substituição incremental)
if os.path.exists(os.path.join(best_trial_artifacts_dir, "test_recons.npy")):
    print(f"\n✅ Artefatos do melhor trial (Trial {best_trial_number}) encontrados em: {best_trial_artifacts_dir}")
    print("Esses arrays correspondem EXATAMENTE às métricas do melhor trial")
    if os.path.exists(best_trial_file):
        print("(SSIM Global: {:.6f}, SSIM Last Snapshot: {:.6f})".format(
            best_trial_info.get("ssim_score", 0.0), 
            best_trial_info.get("ssim_last_snapshot", 0.0)
        ))
    
    # Salvar as métricas do Optuna no results.json dentro de best_trial_artifacts
    # Isso garante que o plot_turb.ipynb use as métricas exatas do Optuna
    if os.path.exists(best_trial_file):
        # Usar métricas do melhor trial salvo
        optuna_metrics = {
            "SSIM_global": float(best_trial_info.get("ssim_score", 0.0)),
            "SSIM_last_snapshot": float(best_trial_info.get("ssim_last_snapshot", 0.0)),
            "PSNR_last_snapshot": float(best_trial_info.get("psnr_last_snapshot", 0.0)),
            "Normalized_Error_global": float(best_trial_info.get("error_norm", 0.0)),
            "Normalized_Error_last_snapshot": float(best_trial_info.get("error_norm_last_snapshot", 0.0)),
            "source": "Optuna (during optimization)",
            "trial_number": int(best_trial_number)
        }
    else:
        # Fallback: usar métricas do trial atual
        optuna_metrics = {
            "SSIM_global": float(results.get("ssim_score", 0.0)),
            "SSIM_last_snapshot": float(results.get("ssim_last_snapshot", 0.0)),
            "PSNR_last_snapshot": float(results.get("psnr_last_snapshot", 0.0)),
            "Normalized_Error_global": float(results.get("error_norm", 0.0)),
            "Normalized_Error_last_snapshot": float(results.get("error_norm_last_snapshot", 0.0)),
            "source": "Optuna (during optimization)",
            "trial_number": int(trial.number)
        }
    
    results_json_path = os.path.join(best_trial_artifacts_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(optuna_metrics, f, indent=4)
    print(f"\nMétricas do Optuna salvas em: {results_json_path}")
    
    # Carregar os arrays já salvos (não recarregar dados) para gerar imagens
    print("\nGerando imagens a partir dos arrays salvos do melhor trial...")
    matrix = np.load(os.path.join(best_trial_artifacts_dir, "matrix.npy"))
    snapshot = np.load(os.path.join(best_trial_artifacts_dir, "snapshot.npy"))
    
    # Gerar visualizações usando os mesmos dados do trial
    visualize_data(matrix, snapshot, best_trial_artifacts_dir, save_plots=True)
    
    # Gerar plot de dinâmica nos sensores usando o mesmo seed do trial
    # (as posições dos sensores já foram salvas nos arrays copiados)
    model_type = 'CS-SHRED'
    plot_dynamics_at_sensors(
        snapshot,
        trial.params.get("num_sensors"),
        locations="c",
        show_plot=False,
        seed=seed,  # Mesmo seed garante as mesmas posições
        save_plot=True,
        save_path=best_trial_artifacts_dir,
        file_name=f"plot_din_{model_type}.png",
        model_type=model_type
    )
    
    print(f"\nImagens geradas em: {best_trial_artifacts_dir}")
    print("="*80 + "\n")
else:
    print(f"\n⚠️ ATENÇÃO: Artefatos do melhor trial não encontrados em {best_trial_artifacts_dir}!")
    print("Isso não deveria acontecer se a otimização foi executada corretamente.")
    print("\nOPÇÕES:")
    print("1. Re-executar o melhor trial (pode produzir resultados ligeiramente diferentes)")
    print("2. Verificar se houve algum erro durante a otimização")
    print("\n⚠️ IMPORTANTE: Re-execução pode não reproduzir exatamente as métricas do Optuna")
    print("   devido a early stopping, não-determinismo numérico, etc.")
    
    # Re-executar o melhor trial como fallback
    print(f"\n❌ Re-executando o melhor trial como fallback...")
    print("   ⚠️ AVISO: Os resultados podem não corresponder exatamente às métricas do Optuna")
    npy_file_path = r'/home/romulo/Documentos/lpips-env/codigos_para_gerar_imagem_cs-shred/data/turb_vy_combined.npy'
    model_type = 'CS-SHRED'
    save_best_trial_artifacts(trial, results_dir, npy_file_path, model_type=model_type, seed=915)

