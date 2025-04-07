import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import spgl1
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


# torch.cuda.empty_cache()
import pylops
from pylops.optimization.sparsity import spgl1
import torch

import warnings
warnings.filterwarnings("ignore", message="Linesearch failed with error 1")

###################################### CS-SHRED
def recover_signal(x, l1_precision, opt_tol, ls_tol, n_sparsity_threshold, verbosity):
    """_summary_

    Args:
        x (_type_): _description_
        l1_precision (_type_): _description_
        opt_tol (_type_): _description_
        ls_tol (_type_): _description_
        n_sparsity_threshold (_type_): _description_
        verbosity (_type_): _description_

    Returns:
        _type_: _description_
    """

    # x_np = x.cpu().numpy() 
    # n_sparsity_threshold = 1. - n_sparsity_threshold

    # n_sparse = np.where(x_np == 0)[0]
    # size = x_np.size
    # percentage = len(n_sparse) / size
        
    # if percentage > n_sparsity_threshold:
    #     return torch.tensor(x_np)

    x_np = x.cpu().numpy() 
    # n_sparsity_threshold=0.8

    n_sparse = np.where(x_np == 0)[0]
    size = x_np.size
    percentage = len(n_sparse) / size
        
    if percentage <= n_sparsity_threshold:
        return torch.tensor(x_np)
    
    else:
        try:
            print("Starting signal recovery")
        
            iava = np.nonzero(x_np > 0)[0]
            Rop = pylops.Restriction(x.numel(), iava=iava, dtype="float64")
            y = Rop * x_np
            RopH = Rop.H

            Fop = pylops.signalprocessing.FFT(x.numel(), dtype="complex128")
            Op = Rop * Fop.H
            Op_adj = Fop * RopH

            x_recovered, _, _ = spgl1(
                Op,
                y,
                verbosity=verbosity,
                iter_lim=4000,
                opt_tol=opt_tol,
                bp_tol=l1_precision,
                ls_tol=ls_tol,
                show=False,
            )

            recovered_signal_time = Fop.H * x_recovered
            recovered_signal_time = recovered_signal_time.reshape(x.shape)
            recovered_signal_tensor = torch.tensor(recovered_signal_time)  
            print("Signal recovery completed")

            return recovered_signal_tensor
        
        except Exception as e:
            print(f"Error in signal recovery: {e}")
            return torch.tensor(x_np)  # Return original signal if recovery fails
        

# def recover_signal(x, l1_precision, opt_tol, ls_tol, verbosity):
#     x_np = x.cpu().numpy()


#     iava = np.nonzero(x_np > 0)[0]
#     Rop = pylops.Restriction(x.numel(), iava=iava, dtype="float64")
#     y = Rop * x_np
#     RopH = Rop.H

#     Fop = pylops.signalprocessing.FFT(x.numel(), dtype="complex128")
#     Op = Rop * Fop.H
#     Op_adj = Fop * RopH

#     x_recovered, _, _ = spgl1(
#         Op,
#         y,
#         verbosity=verbosity,
#         iter_lim=4000,
#         opt_tol=opt_tol,
#         bp_tol=l1_precision,
#         ls_tol=ls_tol,
#         show=False,
#     )

#     recovered_signal_time = Fop.H * x_recovered
#     recovered_signal_time = recovered_signal_time.reshape(x.shape)
#     recovered_signal_tensor = torch.tensor(recovered_signal_time)

#     return recovered_signal_tensor


def recover_signal_per_column(x, l1_precision, opt_tol, ls_tol, n_sparsity_threshold,verbosity):
    recovered_signals = []
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            column_data = x[:, i, j] # if len(x.shape) > 2 else x[:, i]
            recovered_signal = recover_signal(column_data, l1_precision, opt_tol, ls_tol, n_sparsity_threshold,verbosity)
            recovered_signals.append(recovered_signal)
    # print('************************recovered_signals***********************************')        
    # print(recovered_signals[0], recovered_signals[1], recovered_signals[2])
    # plt.plot(
    #     x[:, 0, 0].detach().cpu().numpy(),
    #     label=f"Subsampled Signal (Column {0}, Channel {0})",
    #     color="red",
    #     linestyle='--'
    
    # )       
    # plt.plot(
    #     recovered_signals[0],
    #     label=f"Recovered Signal (Column {1})",
    #     color="green",
    #     linestyle='dashed'
    # )
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.plot(
    #     recovered_signals[1],
    #     label=f"Recovered Signal (Column {2})",
    #     color="green",
    # )
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.plot(
    #     recovered_signals[2],
    #     label=f"Recovered Signal (Column {3})",
    #     color="green",
    # )
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return recovered_signals


class CSSHRED(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.0,
        l1_tol=1e-3,
        opt_tol=1e-4, 
        ls_tol=1e-4,
        n_sparsity_threshold=0.75,   
        verbosity=-1,
        show_plot=False,
    ):
        super(CSSHRED, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_size, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, output_size)
        self.dropout = nn.Dropout(dropout)

        # Aplicando inicialização Xavier para as camadas lineares
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.l1_tol = l1_tol
        self.opt_tol = opt_tol 
        self.ls_tol = ls_tol
        self.n_sparsity_threshold = n_sparsity_threshold
        self.verbosity_spgl1 = verbosity
        self.show_plot = show_plot

    def forward(self, x):

        recovered_signals_per_column = recover_signal_per_column(
            x, self.l1_tol, self.opt_tol, self.ls_tol, self.n_sparsity_threshold, self.verbosity_spgl1
        )
        combined_recovered_signal = torch.stack(recovered_signals_per_column, dim=1)

        # print("****************combined_recovered_signal********************")
        # print(combined_recovered_signal)

        combined_recovered_signal_expanded = combined_recovered_signal.unsqueeze(-1)
        combined_recovered_signal_expanded = combined_recovered_signal_expanded.squeeze(
            -1
        )
        combined_recovered_signal_expanded = (
            combined_recovered_signal_expanded.permute(0, 2, 1)
            if len(combined_recovered_signal_expanded.shape) > 2
            else combined_recovered_signal_expanded.permute(0, 1)
        )
        # print("****************combined_recovered_signal_expanded****************")
        # print(combined_recovered_signal_expanded)
        combined_recovered_signal_expanded = combined_recovered_signal_expanded.float()
        if self.show_plot:
            num_columns = x.size(1)
            num_channels = x.size(2)
            # plt.figure(figsize=(15*num_channels, 5*num_columns))
            # for i in range(num_columns):
            #     for j in range(num_channels):
            #         plt.subplot(num_columns, num_channels, i*num_channels + j + 1)
            #         plt.plot(
            #             x[:, i, j].detach().cpu().numpy(),
            #             label=f"Subsampled Signal (Column {i+1}, Channel {j+1})",
            #             color="red",
            #         )
            #         plt.plot(
            #             recovered_signals_per_column[i],
            #             label=f"Recovered Signal (Column {i+1})",
            #             color="green",
            #         )
            #     plt.xlabel("Time")
            #     plt.ylabel("Amplitude")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.show()
            # i=1
            # if i==1 or i%50==0:
            plt.plot(
                x[:, 0, 0].detach().cpu().numpy(),
                label=f"Subsampled Signal (Column {0}, Channel {0})",
                color="red", linewidth=5
            
            )
            # plt.xlabel("Time")
            # plt.ylabel("Amplitude")
            # plt.legend()
            # plt.grid(False)
            # plt.show()
            plt.plot(
                x[:, 0, 1].detach().cpu().numpy()+1,
                label=f"Subsampled Signal (Column {0}, Channel {1})",
                color="green", linewidth=5
            )
            # plt.xlabel("Time")
            # plt.ylabel("Amplitude")
            # plt.legend()
            # plt.grid(False)
            # plt.show()
            plt.plot(
                x[:, 0, 2].detach().cpu().numpy()+2,
                label=f"Subsampled Signal (Column {0}, Channel {2})",
                color="blue", linewidth=5
            )
            
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            # plt.legend()
            plt.grid(False)
            plt.show()


        h_0 = torch.zeros(
            self.hidden_layers,
            combined_recovered_signal_expanded.size(0),
            self.hidden_size,
            dtype=torch.float,
        )
        c_0 = torch.zeros(
            self.hidden_layers,
            combined_recovered_signal_expanded.size(0),
            self.hidden_size,
            dtype=torch.float,
        )

        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            combined_recovered_signal_expanded = (
                combined_recovered_signal_expanded.cuda()
            )
        if len(x.shape) > 2:
            combined_recovered_signal_expanded = (
                combined_recovered_signal_expanded.unsqueeze(-1).repeat(
                    1, 1, x.shape[2]
                )
            )
        elif len(x.shape) == 2:
            combined_recovered_signal_expanded = (
                combined_recovered_signal_expanded.unsqueeze(-1)
            )

        if combined_recovered_signal_expanded.size(0) != x.size(0):
            combined_recovered_signal_expanded = combined_recovered_signal_expanded[
                : x.size(0)
            ]

        combined_recovered_signal_expanded = combined_recovered_signal_expanded.float()

        _, (h_out, _) = self.lstm(combined_recovered_signal_expanded, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = F.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = F.relu(output)

        output = self.linear3(output)

        return output

######################################


# Original
class SHRED(torch.nn.Module):
    """SHRED model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state, hidden_size, number of LSTM layers,
    size of fully-connected layers, and dropout parameter"""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        hidden_layers=2,
        l1=350,
        l2=400,
        dropout=0.0,
    ):
        super(SHRED, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            batch_first=True,
        )

        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):

        # plt.plot(
        #         x[:, 0, 0].detach().cpu().numpy(),
        #         label=f"Subsampled Signal (Column {0}, Channel {0})",
        #         color="red",
            
        #     )
        #     # plt.plot(
        #     #     recovered_signals_per_column[0],
        #     #     label=f"Recovered Signal (Column {1})",
        #     #     color="green",
        #     #     linestyle='--'
        #     # )
    
        # plt.xlabel("Time")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        h_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float
        )
        c_0 = torch.zeros(
            (self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float
        )

        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1].view(-1, self.hidden_size)

        output = self.linear1(h_out)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)

        return output


class SDN(torch.nn.Module):
    """SDN model accepts input size (number of sensors), output size (dimension of high-dimensional spatio-temporal state,
    size of fully-connected layers, and dropout parameter"""

    def __init__(self, input_size, output_size, l1=350, l2=400, dropout=0.0):
        super(SDN, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, output_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        output = self.linear1(x)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear2(output)
        output = self.dropout(output)
        output = torch.nn.functional.relu(output)

        output = self.linear3(output)

        return output


def fit(
    model,
    train_dataset,
    valid_dataset,
    batch_size=64,
    num_epochs=4000,
    lr=1e-3,
    step_epoch=50,
    verbose=False,
    patience=5,
):
    """Function for training SHRED and SDN models"""
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):

        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()

        if epoch % step_epoch == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(
                    val_outputs - valid_dataset.Y
                ) / torch.linalg.norm(valid_dataset.Y)
                # val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)

            if verbose == True:
                print("Training epoch " + str(epoch))
                print("Error " + str(val_error_list[-1]))
                # Plotagem a cada x épocas
                # if epoch % 50 == 0 or epoch == 1:
                
                    # plt.figure(figsize=(10, 5))
                    # plt.plot(
                    #     valid_dataset.X[:, 0].detach().cpu().numpy(),
                    #     label="Subsampled Signal",
                    #     color="red",
                    # )
                    # plt.plot(
                    #     val_outputs[:, 0].detach().cpu().numpy(),
                    #     label="Recovered Signal",
                    #     color="green",
                    #     linestyle='--'
                    # )
                    # plt.xlabel("Time")
                    # plt.ylabel("Amplitude")
                    # plt.legend()
                    # plt.grid(True)
                    # plt.title(f"Epoch {epoch}")
                    # plt.show()

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


def total_variation_regularization(outputs):
    h_diff = torch.abs(outputs[:, :-1] - outputs[:, 1:])
    v_diff = torch.abs(outputs[:-1, :] - outputs[1:, :])
    total_var = torch.sum(h_diff) + torch.sum(v_diff)
    return total_var


def abrupt_transition_regularization(outputs):
    h_diff = torch.abs(outputs[:, :-1] - outputs[:, 1:])
    total_diff = torch.sum(h_diff)
    return total_diff


def calculate_snr(original_signal, noisy_signal):
    signal_power = torch.mean(original_signal**2)
    noise_power = torch.mean((noisy_signal - original_signal) ** 2)
    eps = 1e-10
    snr_db = 10 * torch.log10(signal_power / (noise_power + eps))

    return snr_db


def fit_csshred_model(
    model,
    train_dataset,
    valid_dataset,
    batch_size=64,
    num_epochs=4000,
    lr=1e-3,
    lambL2=1,
    lambL1=0.01,
    lambdaSNR=0.03,
    step_epoch=15,
    verbose=False,
    patience=5,
):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    weight_decay = 1e-4
    lambd = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    val_error_list = []
    train_error_list = []
    patience_counter = 0  
    lambL2 = lambL2  
    lambL1 = lambL1  
    lambdaSNR = lambdaSNR  
    best_params = model.state_dict()

    for epoch in range(1, num_epochs + 1):
        train_losses = []

        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()

            lossMSE = criterion(outputs, data[1])
            lossL1 = criterion2(outputs, torch.zeros_like(outputs))
            snr = calculate_snr(data[1], outputs)

            # Regularização L2
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)

            # Ajustando a perda para incentivar a maximização do SNR
            if snr > 0:
                loss = (
                    torch.clamp(1 / (snr + 1e-8), max=100.0) * lambdaSNR
                    + lambL2 * lossMSE
                    + lambL1 * lossL1
                    + weight_decay * l2_reg
                )  # Quanto maior o SNR, menor será a perda
            else:
                loss = (
                    - snr * lambdaSNR
                    + lambL2 * lossMSE
                    + lambL1 * lossL1
                    + weight_decay * l2_reg
                )  # Inversa do SNR: quanto menor o SNR, maior será a perda

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        train_error_list.append(avg_train_loss)

        # Calculando o erro de validação após o término da época
        if epoch % step_epoch == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)

                # Calculando o SNR da validação
                val_snr = calculate_snr(valid_dataset.Y, val_outputs)

                # Ajustando a perda de validação para incentivar a maximização do SNR
                if val_snr > 0:
                    val_loss = (
                        torch.clamp(1 / (val_snr + 1e-8), max=100.0)  * lambdaSNR + lambL2 * lossMSE + lambL1 * lossL1
                    )
                else:
                    val_loss = val_snr * lambdaSNR + lambL2 * lossMSE + lambL1 * lossL1

                val_error_list.append(val_loss)
            scheduler.step(val_loss)

            if verbose:
                print(f"LambdaL2: {lambL2}, LambdaL1: {lambL1}, LambdaSNR: {lambdaSNR}")
                print("Training epoch " + str(epoch))
                print("Training Error: " + str(avg_train_loss))
                print("Validation Error:" + str(val_loss.item()))
                print("SNR:" + str(snr.item()))

                # plt.figure(figsize=(10, 5))
                # plt.plot(
                #     valid_dataset.X[:, 0].detach().cpu().numpy(),
                #     label="Subsampled Signal",
                #     color="red",
                # )
                # plt.plot(
                #     val_outputs[:, 0].detach().cpu().numpy(),
                #     label="Recovered Signal",
                #     color="green",
                #     linestyle='--'
                # )
                # plt.xlabel("Time")
                # plt.ylabel("Amplitude")
                # plt.legend()
                # plt.grid(True)
                # plt.title(f"Epoch {epoch}")
                # plt.show()

            if len(val_error_list) > 0:
                if val_loss == torch.min(torch.tensor(val_error_list)):
                    patience_counter = 0
                    best_params = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter == patience:
                model.load_state_dict(best_params)
                return (
                    torch.tensor(train_error_list).cpu(),
                    torch.tensor(val_error_list).cpu(),
                )

    model.load_state_dict(best_params)
    return torch.tensor(train_error_list).detach().cpu().numpy(),torch.tensor(val_error_list).detach().cpu().numpy()
    



def forecast(forecaster, reconstructor, test_dataset):
    """Takes model and corresponding test dataset, returns tensor containing the
    inputs to generate the first forecast and then all subsequent forecasts
    throughout the test dataset."""
    initial_in = test_dataset.X[0:1].clone()
    vals = []
    for i in range(0, test_dataset.X.shape[1]):
        vals.append(initial_in[0, i, :].detach().cpu().clone().numpy())

    for i in range(len(test_dataset.X)):
        scaled_output = forecaster(initial_in).detach().cpu().numpy()

        vals.append(scaled_output.reshape(test_dataset.X.shape[2]))
        temp = initial_in.clone()
        initial_in[0, :-1] = temp[0, 1:]
        initial_in[0, -1] = torch.tensor(scaled_output)

    device = "cuda" if next(reconstructor.parameters()).is_cuda else "cpu"
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = (
            reconstructor(
                forecasted_vals[i : i + test_dataset.X.shape[1]].reshape(
                    1, test_dataset.X.shape[1], test_dataset.X.shape[2]
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions
