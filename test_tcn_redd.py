import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))

from models import TCNModel
from utils import calculate_nilm_metrics, save_model


class REDDDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_redd_splits():
    print("Loading REDD data...")

    with open('data/redd/train_small.pkl', 'rb') as f:
        train_data = pickle.load(f)[0]
    with open('data/redd/val_small.pkl', 'rb') as f:
        val_data = pickle.load(f)[0]
    with open('data/redd/test_small.pkl', 'rb') as f:
        test_list = pickle.load(f)
    test_df = test_list[3]
    test_data = test_df[
        pd.DatetimeIndex(test_df.index).date == pd.Timestamp('2011-04-30').date()
    ].reset_index(drop=True)

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape:  {test_data.shape}  (House 3, 2011-04-30)")
    print(f"Available columns: {list(train_data.columns)}")

    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=100, target_size=1):
    mains = data['main'].values
    X, y = [], []
    stride = 5
    for i in range(0, len(mains) - window_size - target_size + 1, stride):
        X.append(mains[i:i+window_size])
        midpoint = i + window_size // 2
        y.append(mains[midpoint:midpoint+1])
    return np.array(X).reshape(-1, window_size, 1), np.array(y)


def get_threshold_for_appliance(appliance_name):
    return 0.5 if appliance_name == 'washer dryer' else 10.0


def train_tcn_on_appliance(data_dict, appliance_name, window_size=100,
                            num_channels=None, kernel_size=3, dropout=0.2,
                            epochs=80, lr=0.001, patience=20,
                            save_dir='models/tcn_redd'):
    if num_channels is None:
        num_channels = [32, 64, 128]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"Creating sequences for {appliance_name}...")
    X_train, _ = create_sequences(train_data, window_size=window_size, target_size=1)
    X_val,   _ = create_sequences(val_data,   window_size=window_size, target_size=1)
    X_test,  _ = create_sequences(test_data,  window_size=window_size, target_size=1)

    y_train = train_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_train)]
    y_val   = val_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_val)]
    y_test  = test_data[appliance_name].iloc[::5].values.reshape(-1, 1)[:len(X_test)]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    print(f"Training sequences:   {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences:       {X_test.shape} -> {y_test.shape}")

    train_loader = torch.utils.data.DataLoader(
        REDDDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        REDDDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        REDDDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = TCNModel(
        input_size=1, output_size=1,
        num_channels=num_channels, kernel_size=kernel_size, dropout=dropout
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter = 0

    print(f"Starting TCN training for {appliance_name}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        all_targets = y_scaler.inverse_transform(
            np.concatenate(all_targets).reshape(-1, 1)).flatten()
        all_outputs = y_scaler.inverse_transform(
            np.concatenate(all_outputs).reshape(-1, 1)).flatten()
        threshold = get_threshold_for_appliance(appliance_name)
        metrics = calculate_nilm_metrics(all_targets, all_outputs, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, Val MAE: {metrics['mae']:.2f}, "
              f"Val SAE: {metrics['sae']:.2f}, Val F1: {metrics['f1']:.4f}, "
              f"Val Precision: {metrics['precision']:.4f}, Val Recall: {metrics['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
            best_model_path = os.path.join(
                save_dir, f"tcn_redd_{appliance_name.replace(' ', '_')}_best.pth")
            save_model(model,
                       {'input_size': 1, 'output_size': 1,
                        'num_channels': num_channels, 'kernel_size': kernel_size,
                        'dropout': dropout},
                       {'lr': lr, 'epochs': epochs, 'patience': patience,
                        'window_size': window_size, 'appliance': appliance_name},
                       metrics, best_model_path)
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print("Training completed!")

    model.eval()
    test_loss = 0.0
    all_test_targets, all_test_outputs = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_test_targets = y_scaler.inverse_transform(
        np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    all_test_outputs = y_scaler.inverse_transform(
        np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()
    threshold = get_threshold_for_appliance(appliance_name)
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)

    val_mae_series       = [m['mae']       for m in history['val_metrics']]
    val_sae_series       = [m['sae']       for m in history['val_metrics']]
    val_f1_series        = [m['f1']        for m in history['val_metrics']]
    val_precision_series = [m['precision'] for m in history['val_metrics']]
    val_recall_series    = [m['recall']    for m in history['val_metrics']]

    aggregates = {
        'train_loss_mean':    float(np.mean(history['train_loss'])),
        'train_loss_var':     float(np.var(history['train_loss'])),
        'val_loss_mean':      float(np.mean(history['val_loss'])),
        'val_loss_var':       float(np.var(history['val_loss'])),
        'val_mae_mean':       float(np.mean(val_mae_series)),
        'val_mae_var':        float(np.var(val_mae_series)),
        'val_sae_mean':       float(np.mean(val_sae_series)),
        'val_sae_var':        float(np.var(val_sae_series)),
        'val_f1_mean':        float(np.mean(val_f1_series)),
        'val_f1_var':         float(np.var(val_f1_series)),
        'val_precision_mean': float(np.mean(val_precision_series)),
        'val_precision_var':  float(np.var(val_precision_series)),
        'val_recall_mean':    float(np.mean(val_recall_series)),
        'val_recall_var':     float(np.var(val_recall_series)),
        'test_mae':           float(test_metrics['mae']),
        'test_sae':           float(test_metrics['sae']),
        'test_f1':            float(test_metrics['f1']),
        'test_precision':     float(test_metrics['precision']),
        'test_recall':        float(test_metrics['recall']),
        'test_loss':          float(avg_test_loss),
    }

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series, label='Val F1', color='red')
    plt.axhline(test_metrics['f1'], label='Test F1', color='green', linestyle='--')
    plt.title(f'F1 Score - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
        f"tcn_redd_{appliance_name.replace(' ', '_')}_metrics.png"),
        dpi=300, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'REDD',
        'window_size': window_size,
        'model_params': {
            'input_size': 1, 'output_size': 1,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss':   history['val_loss'][-1]   if history['val_loss']   else None,
            'test_loss':  avg_test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': aggregates
        }
    }
    with open(os.path.join(save_dir,
            f'tcn_redd_{appliance_name.replace(" ", "_")}_history.json'),
            'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


def test_tcn_on_all_redd_appliances(window_size=100, num_channels=None, kernel_size=3,
                                    dropout=0.2, epochs=80, lr=0.001, patience=20):
    if num_channels is None:
        num_channels = [32, 64, 128]

    data_dict = load_redd_splits()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"models/tcn_redd_{timestamp}"

    all_results = {}
    appliances = ['dish washer', 'fridge', 'microwave', 'washer dryer']

    for appliance_name in appliances:
        print(f"\n{'='*60}")
        print(f"Testing TCN on {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name.replace(' ', '_'))
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_tcn_on_appliance(
                data_dict, appliance_name=appliance_name,
                window_size=window_size, num_channels=num_channels,
                kernel_size=kernel_size, dropout=dropout,
                epochs=epochs, lr=lr, patience=patience, save_dir=appliance_dir)
            if model is not None:
                all_results[appliance_name] = {
                    'model_path': os.path.join(
                        appliance_dir,
                        f"tcn_redd_{appliance_name.replace(' ', '_')}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback; traceback.print_exc()

    os.makedirs(base_save_dir, exist_ok=True)
    summary = {
        'timestamp': timestamp, 'dataset': 'REDD', 'model': 'TCN',
        'window_size': window_size,
        'model_params': {'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout},
        'train_params': {'epochs': epochs, 'lr': lr, 'patience': patience},
        'results': all_results
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nTCN REDD testing completed. Results saved to {base_save_dir}")
    for appliance, result in all_results.items():
        m = result['final_metrics']
        print(f"  {appliance:<15}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")
    return all_results


if __name__ == "__main__":
    print("Testing TCN on REDD dataset...")
    for f in ['data/redd/train_small.pkl', 'data/redd/val_small.pkl', 'data/redd/test_small.pkl']:
        if not os.path.exists(f):
            print(f"Error: {f} not found!"); import sys; sys.exit(1)

    test_tcn_on_all_redd_appliances(
        window_size=100, num_channels=[32, 64, 128], kernel_size=3,
        dropout=0.2, epochs=80, lr=0.001, patience=20)
