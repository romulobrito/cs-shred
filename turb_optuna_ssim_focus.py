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

def subsample(snapshot, num_cols_subsample, num_snapshots_subsample):
    np.random.seed(1001)
    print('snapshot', snapshot.shape)
    
    snapshot = np.transpose(snapshot, (1, 2, 0))
    dim_x, dim_y, dim_t = snapshot.shape
    snapshot_subsampled = snapshot.copy()

    num_snapshots_subsample = min(num_snapshots_subsample, dim_t - 1)
    num_cols_subsample = min(num_cols_subsample, dim_y - 1)
    cols_to_subsample = np.random.choice(dim_y, size=num_cols_subsample, replace=False)
    cols_to_subsample = np.sort(cols_to_subsample)

    available_snapshots = np.arange(dim_t - 1)
    snapshots_to_subsample = np.random.choice(
        available_snapshots, 
        size=num_snapshots_subsample - 1, 
        replace=False
    )
    snapshots_to_subsample = np.append(snapshots_to_subsample, dim_t - 1)
    snapshots_to_subsample = np.sort(snapshots_to_subsample)

    mask = np.zeros((dim_x, dim_y, dim_t), dtype=bool)
    for t in snapshots_to_subsample:
        mask[:, cols_to_subsample, t] = True

    total_points = dim_x * dim_y * dim_t
    masked_points = np.sum(mask)
    if masked_points / total_points > 0.95:
        print("Aviso: Muitos pontos sendo mascarados. Ajustando...")
        return snapshot_subsampled

    snapshot_subsampled[mask] = 0

    if np.all(snapshot_subsampled == 0):
        print("Aviso: Todos os valores foram zerados. Retornando dados originais...")
        return snapshot

    print("Forma do snapshot após subamostragem:", snapshot_subsampled.shape)
    print(f"Porcentagem de dados mantidos: {100 * (1 - np.sum(mask)/mask.size):.2f}%")
    return snapshot_subsampled

def plot_dynamics_at_sensors(trace_A, num_sensors, trial_number, results_dir, locations="c", show_plot=False, seed=101):
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

    sensor_temperature_history = []
    for t in range(dim_t):
        sensor_temperatures = [
            trace_A[int(dim_x * x), int(dim_y * y), t]
            for x, y in zip(sensor_positions_x, sensor_positions_y)
        ]
        sensor_temperature_history.append(sensor_temperatures)

    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.linspace(0, 1, int(dim_x))
        y = np.linspace(0, 1, int(dim_y))
        X, Y = np.meshgrid(x, y)

        cmap = ax1.pcolormesh(X, Y, trace_A[:, :, -1].real, shading="auto", cmap="hot")
        fig.colorbar(cmap, ax=ax1, label=r"$tr(C)$")
        ax1.set_title("Espatial Distribution oF The $tr(C)$ Oldroyd-B Tensor")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.scatter(sensor_positions_x, sensor_positions_y, color="blue", label="Sensor Positions")
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
        plt.close(fig)

    return sensor_locations, sensor_positions_x, sensor_positions_y

def prepare_datasets(trace_A, trace_A_ori, num_sensors, sensor_locations, lags):
    trace_A_ori = np.transpose(trace_A_ori, (1, 2, 0))
    
    dim_x, dim_y, dim_t = trace_A.shape
    
    train_size = int(0.7 * trace_A.shape[2])
    val_size = int(0.2 * trace_A.shape[2])
    test_size = int(0.1 * trace_A.shape[2])
    
    load_X = trace_A.reshape(dim_x * dim_y, dim_t).T
    load_X_test = trace_A_ori.reshape(dim_x * dim_y, dim_t).T
    
    load_X_shape_0, load_X_shape_1 = load_X.shape
    
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
    transformed_X_test = sc.transform(load_X_test)
    
    all_data_in = np.zeros((load_X_shape_0 - lags, lags, num_sensors))
    all_data_in_test = np.zeros((load_X_shape_0 - lags, lags, num_sensors))
    for i in range(load_X_shape_0 - lags):
        for j, loc in enumerate(sensor_locations):
            all_data_in[i, :, j] = transformed_X[i : i + lags, loc]
            all_data_in_test[i, :, j] = transformed_X_test[i : i + lags, loc]
    
    print("device:", device)
    print("Number of sensors:{}".format(num_sensors))
    
    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
    test_data_in_test = torch.tensor(all_data_in_test[test_indices], dtype=torch.float32).to(device)
    
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out_test = torch.tensor(transformed_X_test[test_indices + lags - 1], dtype=torch.float32).to(device)
    
    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    test_dataset_test = TimeSeriesDataset(test_data_in_test, test_data_out_test)
    
    return train_dataset, valid_dataset, test_dataset, test_dataset_test, sc, load_X_shape_1

def train_and_validate_model(type_model, model, train_dataset, valid_dataset, num_epochs, batch_size, lr, lambL2, lambL1, lambdaSNR, step_epoch, verbose, patience):
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
            verbose=verbose,
            patience=patience,
        )
        return validation_errors

def evaluate_model(model, test_dataset, test_dataset_test, sc):
    test_recons = sc.inverse_transform(model(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    test_ground_truth_test = sc.inverse_transform(test_dataset_test.Y.detach().cpu().numpy())
    
    if test_recons.ndim != test_ground_truth_test.ndim:
        raise ValueError("The dimensions of the reconstructed and ground truth data do not correspond.")
    
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
    
    mean_ssim = np.mean(ssim_scores)
    last_snapshot_ssim = ssim_scores[-1]
    
    print("Normalized Error:", error_norm)
    print("Mean SSIM (all samples):", mean_ssim)
    print("Last Snapshot SSIM:", last_snapshot_ssim)
    
    return test_recons, test_ground_truth, test_ground_truth_test, error_norm, mean_ssim, last_snapshot_ssim

def objective(trial):
    # Configuração dos caminhos
    npy_file_path = r'./data/turb_vy_combined.npy'
    results_dir = r"./results/csshred/turb_v3_ssim_fast/"
    os.makedirs(results_dir, exist_ok=True)

    # Carregamento dos dados
    matrix = load_data(npy_file_path, time_slice=650)
    
    model_type = 'CS-SHRED'
    seed = 915

    # Hiperparâmetros OTIMIZADOS para VELOCIDADE + SSIM
    # Baseado em trials anteriores bem-sucedidos (Trial 11, 13)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])      # Focado em valores testados
    hidden_layers = trial.suggest_categorical("hidden_layers", [3, 4])      # Máximo 4 camadas
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])          # Batches eficientes
    lr = trial.suggest_float("lr", 1e-4, 4e-4, log=True)                    # Range mais focado
    lambL2 = trial.suggest_float("lambL2", 0.3, 0.6, log=False)            # Range do Trial 13
    lambL1 = trial.suggest_float("lambL1", 5e-6, 1e-4, log=True)           # Range dos melhores
    lambdaSNR = trial.suggest_float("lambdaSNR", 0.85, 1.0, log=False)     # Focado no Trial 13
    l1 = trial.suggest_categorical("l1", [500])                             # Fixo no valor testado
    l2 = trial.suggest_categorical("l2", [500])                             # Fixo no valor testado
    lags = trial.suggest_int("lags", 15, 50)                        # Range corrigido (era [20,25])
    num_sensors = trial.suggest_categorical("num_sensors", [5, 6])          # Focado em valores bons
    num_epochs = trial.suggest_int("num_epochs", 800, 1800)                # REDUZIDO significativamente
    step_epoch = trial.suggest_int("step_epoch", 10, 60)                    # Range menor
    # Parâmetros CS - ranges mais conservadores baseados em Trial 13
    l1_tol = trial.suggest_float("l1_tol", 3e-4, 1e-3, log=True)           # Mais conservador  
    opt_tol = trial.suggest_float("opt_tol", 1e-4, 3e-4, log=True)         # Range do Trial 13
    ls_tol = trial.suggest_float("ls_tol", 0.1, 0.25, log=False)           # Range mais focado
    dropout = trial.suggest_float("dropout", 0.005, 0.02)                   # Range menor
    patience = trial.suggest_int("patience", 10, 20)                        # Menos paciência = mais rápido

    print("=== TRIAL", trial.number, "HIPERPARÂMETROS ===")
    for param_name, param_value in trial.params.items():
        print(f"{param_name}: {param_value}")

    # Preparação dos dados
    snapshot = matrix.copy()
    num_cols_subsample = int(snapshot.shape[2] * 0.3)
    num_snapshots_subsample = int(snapshot.shape[0] * 0.3)
    snapshot = subsample(snapshot, num_cols_subsample, num_snapshots_subsample)
    
    sensor_locations, sensor_positions_x, sensor_positions_y = plot_dynamics_at_sensors(
        snapshot, num_sensors, trial_number=trial.number, results_dir=results_dir,
        locations="c", show_plot=False, seed=seed
    )

    # Dados originais
    trace_A = snapshot.copy()
    trace_A_ori = matrix.copy()
    
    train_dataset, valid_dataset, test_dataset, test_dataset_test, sc, load_X_shape_1 = prepare_datasets(
        trace_A, trace_A_ori, num_sensors, sensor_locations, lags
    )

    # Modelo CS-SHRED
    model = models.CSSHRED(
        num_sensors,
        load_X_shape_1,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        l1=l1,
        l2=l2,
        dropout=dropout,
        l1_tol=l1_tol,
        opt_tol=opt_tol,
        ls_tol=ls_tol,
        n_sparsity_threshold=num_snapshots_subsample,
        verbosity=0,
        show_plot=False,
    ).to(device)

    # Treinamento
    train_error, validation_errors = train_and_validate_model(
        model_type, model, train_dataset, valid_dataset, num_epochs, batch_size,
        lr, lambL2, lambL1, lambdaSNR, step_epoch, verbose=False, patience=patience
    )

    # Avaliação
    validation_errors = [float(val) for val in validation_errors]
    test_recons, test_ground_truth, test_ground_truth_test, error_norm, mean_ssim, last_snapshot_ssim = evaluate_model(
        model, test_dataset, test_dataset_test, sc
    )

    # Salvar resultados detalhados
    results = {
        "study_version": "v3_ssim_fast",
        "trial_number": trial.number,
        **trial.params,
        "error_norm": float(error_norm),
        "validation_errors": float(np.mean(validation_errors)),
        "ssim_score_mean": float(mean_ssim),
        "ssim_score_last": float(last_snapshot_ssim),
    }
    
    with open(os.path.join(results_dir, f"{trial.number}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Função objetivo FOCADA EM SSIM
    # Estratégia: priorizar SSIM com peso 0.7
    alpha = 0.25  # Menos peso para erro (queremos priorizar SSIM)
    objective_value = alpha * error_norm + (1 - alpha) * (1 - mean_ssim)
    
    print(f"Objective components - Error: {error_norm:.4f}, SSIM: {mean_ssim:.4f}")
    print(f"SSIM-Focused Objective: {objective_value:.4f}")
    
    # Adicionar valor objetivo ao JSON
    with open(os.path.join(results_dir, f"{trial.number}_results.json"), "r") as f:
        results = json.load(f)
    
    results["objective_value"] = float(objective_value)
    results["objective_alpha"] = alpha
    results["objective_strategy"] = "ssim_priority"
    
    with open(os.path.join(results_dir, f"{trial.number}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return float(objective_value)

# Early stopping callback
class EarlyStoppingCallback:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.wait = 0

    def __call__(self, study, trial):
        current_value = trial.value
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                study.stop()

if __name__ == "__main__":
    print("🎯 INICIANDO ESTUDO OPTUNA FOCADO EM SSIM")
    print("=" * 60)
    
    # Criar estudo
    study = optuna.create_study(
        direction="minimize",
        study_name="turb_csshred_ssim_fast_v3",
        storage="sqlite:///optuna_turb_ssim_fast.db",
        load_if_exists=True
    )

    # Adicionar callback de early stopping
    early_stopping = EarlyStoppingCallback(patience=25, min_delta=0.001)

    # Trials iniciais RÁPIDOS baseados em Trial 13 (melhor resultado anterior)
    promising_params = [
        # Trial 1: Baseado no Trial 13 com épocas reduzidas
        {
            "hidden_size": 512,
            "hidden_layers": 3,        # Menos camadas = mais rápido
            "batch_size": 16,
            "lr": 0.000373535922169003,  # Exato do Trial 13
            "lambL2": 0.535095070820673, # Exato do Trial 13
            "lambL1": 8.107628877627401e-06,  # Exato do Trial 13
            "lambdaSNR": 0.8572315905468348,  # Exato do Trial 13
            "l1": 500,
            "l2": 500,
            "lags": 25,
            "num_sensors": 5,          # Original do Trial 13
            "num_epochs": 1200,        # MUITO REDUZIDO (vs 2407 original)
            "step_epoch": 53,
            "l1_tol": 0.0008230382462557888,  # Trial 13
            "opt_tol": 0.00027129845065834403, # Trial 13
            "ls_tol": 0.2090686856607643,      # Trial 13
            "dropout": 0.0189525346844655,     # Trial 13
            "patience": 10,            # Reduzido para ser mais rápido
        },
        # Trial 2: Variação rápida com menos sensores
        {
            "hidden_size": 256,        # Menor = mais rápido
            "hidden_layers": 3,
            "batch_size": 32,          # Maior batch = mais rápido
            "lr": 0.0002,
            "lambL2": 0.4,
            "lambL1": 5e-5,
            "lambdaSNR": 0.9,
            "l1": 500,
            "l2": 500,
            "lags": 20,                # Menos lags = mais rápido
            "num_sensors": 5,
            "num_epochs": 1000,        # Bem reduzido
            "step_epoch": 50,
            "l1_tol": 5e-4,
            "opt_tol": 2e-4,
            "ls_tol": 0.15,
            "dropout": 0.01,
            "patience": 15,
        }
    ]

    for params in promising_params:
        study.enqueue_trial(params)

    print(f"📋 {len(promising_params)} trials promissores adicionados à fila")
    print("🚀 Iniciando otimização...")

    # Executar otimização - REDUZIDO para ser mais gerenciável
    study.optimize(objective, n_trials=25, callbacks=[early_stopping])

    # Relatório final
    print("\n🏆 MELHORES RESULTADOS:")
    print("=" * 60)
    print(f"Melhor valor objetivo: {study.best_value:.4f}")
    print("Melhores parâmetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Salvar estudo
    results_dir = r"./results/csshred/turb_v3_ssim_fast/"
    with open(os.path.join(results_dir, "study_summary.json"), "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "total_trials": len(study.trials),
            "study_name": study.study_name
        }, f, indent=4)

    print(f"💾 Resumo do estudo salvo em {results_dir}study_summary.json") 