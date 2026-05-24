"""
LNN (Liquid Neural Network) — REDD Dataset with Fine-tuning

3-phase pipeline:
  Phase 1 — Pretrain:  data/redd/train_small.pkl
  Validation:          data/redd/val_small.pkl
  Phase 2 — Fine-tune: data/redd/test_small.pkl[0]  (first test house)
  Phase 3 — Test:      data/redd/test_small.pkl[3]  (House 3, 2011-04-30)
"""

import sys
import os
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LiquidNetworkModel
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'redd')

REDD_APPLIANCES = ['dish washer', 'fridge', 'microwave', 'washer dryer']
THRESHOLD = 10.0

EPOCHS    = 80;  PATIENCE    = 20;  LR    = 1e-3
EPOCHS_FT = 30;  PATIENCE_FT = 10;  LR_FT = 1e-4
BATCH     = 32;  WIN         = 100;  STRIDE = 5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class REDDDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_redd_splits(data_dir=DEFAULT_DATA_DIR):
    print("Loading REDD data...")
    with open(os.path.join(data_dir, 'train_small.pkl'), 'rb') as f:
        train_data = pickle.load(f)[0]
    with open(os.path.join(data_dir, 'val_small.pkl'), 'rb') as f:
        val_data = pickle.load(f)[0]
    with open(os.path.join(data_dir, 'test_small.pkl'), 'rb') as f:
        test_list = pickle.load(f)

    ft_data   = test_list[0]
    test_df   = test_list[3]
    test_data = test_df[
        pd.DatetimeIndex(test_df.index).date == pd.Timestamp('2011-04-30').date()
    ].reset_index(drop=True)

    print(f"  pretrain : {train_data.shape}  cols={list(train_data.columns)}")
    print(f"  val      : {val_data.shape}")
    print(f"  finetune : {ft_data.shape}   (test_list[0])")
    print(f"  test     : {test_data.shape}  (test_list[3], 2011-04-30)")

    return {'pretrain': train_data, 'val': val_data,
            'finetune': ft_data,    'test': test_data}


def create_sequences(data, appliance_col):
    mains = data['main'].values
    tgts  = data[appliance_col].values
    X, y  = [], []
    for i in range(0, len(mains) - WIN, STRIDE):
        X.append(mains[i:i + WIN])
        y.append(tgts[i + WIN // 2])
    return (np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
            np.array(y, dtype=np.float32).reshape(-1, 1))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total = 0.0
    outs, tgts = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            outs.append(out.detach().cpu().numpy())
            tgts.append(yb.cpu().numpy())
    return total / len(loader), np.concatenate(outs), np.concatenate(tgts)


def _metrics(raw_true, raw_pred):
    return calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLD)


def _aggregates(history, test_metrics):
    vm = history['val_metrics']
    return {
        'train_loss_mean': float(np.mean(history['train_loss'])),
        'train_loss_var':  float(np.var(history['train_loss'])),
        'val_loss_mean':   float(np.mean(history['val_loss'])),
        'val_loss_var':    float(np.var(history['val_loss'])),
        'val_f1_mean':     float(np.mean([m['f1']  for m in vm])),
        'val_f1_var':      float(np.var( [m['f1']  for m in vm])),
        'val_mae_mean':    float(np.mean([m['mae'] for m in vm])),
        'val_mae_var':     float(np.var( [m['mae'] for m in vm])),
        'val_sae_mean':    float(np.mean([m['sae'] for m in vm])),
        'val_sae_var':     float(np.var( [m['sae'] for m in vm])),
        'test_f1':         float(test_metrics['f1']),
        'test_mae':        float(test_metrics['mae']),
        'test_sae':        float(test_metrics['sae']),
        'test_precision':  float(test_metrics['precision']),
        'test_recall':     float(test_metrics['recall']),
    }


# ---------------------------------------------------------------------------
# Per-appliance pipeline
# ---------------------------------------------------------------------------

def train_on_appliance(splits, redd_appliance, data_dir=DEFAULT_DATA_DIR,
                       hidden_size=64, dt=0.1,
                       save_dir='models/lnn_redd_finetune'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nAppliance: {redd_appliance}  |  device: {device}  "
          f"hidden={hidden_size}  dt={dt}")

    X_pre, y_pre = create_sequences(splits['pretrain'],  redd_appliance)
    X_val, y_val = create_sequences(splits['val'],       redd_appliance)
    X_ft,  y_ft  = create_sequences(splits['finetune'],  redd_appliance)
    X_te,  y_te  = create_sequences(splits['test'],      redd_appliance)

    xs = MinMaxScaler(); ys = MinMaxScaler()
    X_pre = xs.fit_transform(X_pre.reshape(-1, 1)).reshape(X_pre.shape)
    X_val = xs.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_ft  = xs.transform(X_ft.reshape(-1, 1)).reshape(X_ft.shape)
    X_te  = xs.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)
    y_pre = ys.fit_transform(y_pre); y_val = ys.transform(y_val)
    y_ft  = ys.transform(y_ft);     y_te  = ys.transform(y_te)

    mk_loader = lambda X, y, shuf: torch.utils.data.DataLoader(
        REDDDataset(X, y), batch_size=BATCH, shuffle=shuf)
    pre_loader = mk_loader(X_pre, y_pre, True)
    val_loader = mk_loader(X_val, y_val, False)
    ft_loader  = mk_loader(X_ft,  y_ft,  True)
    te_loader  = mk_loader(X_te,  y_te,  False)

    model = LiquidNetworkModel(
        input_size=1, hidden_size=hidden_size, output_size=1, dt=dt).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    criterion = torch.nn.MSELoss()

    # -- Phase 1: Pretrain ----------------------------------------------------
    print("  Phase 1: Pretrain")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val  = float('inf'); best_state = None; counter = 0

    pretrain_start = time.time()
    for epoch in range(EPOCHS):
        ep_start = time.time()
        tr_loss, _, _   = _run_epoch(model, pre_loader, criterion, optimizer, device, True)
        va_loss, vo, vt = _run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(va_loss)
        raw_t = ys.inverse_transform(vt).flatten()
        raw_o = ys.inverse_transform(vo).flatten()
        m = _metrics(raw_t, raw_o)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_metrics'].append(m)
        ep_time = time.time() - ep_start
        print(f"    Ep {epoch+1:3d}  train={tr_loss:.5f}  val={va_loss:.5f}  "
              f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
              f"TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}  "
              f"time={ep_time:.1f}s")
        if va_loss < best_val:
            best_val = va_loss; counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model,
                       {'input_size': 1, 'hidden_size': hidden_size,
                        'output_size': 1, 'dt': dt},
                       {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                        'appliance': redd_appliance},
                       m, os.path.join(save_dir,
                           f'pretrain_{redd_appliance.replace(" ", "_")}_best.pth'))
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}"); break

    model.load_state_dict(best_state)
    print(f"  Phase 1 total: {(time.time()-pretrain_start)/60:.1f} min")

    _, to, tt = _run_epoch(model, te_loader, criterion, optimizer, device, False)
    pre_ft_metrics = _metrics(ys.inverse_transform(tt).flatten(),
                               ys.inverse_transform(to).flatten())
    print(f"  Test BEFORE fine-tune: "
          f"F1={pre_ft_metrics['f1']:.4f}  P={pre_ft_metrics['precision']:.4f}  "
          f"R={pre_ft_metrics['recall']:.4f}  MAE={pre_ft_metrics['mae']:.2f}  "
          f"SAE={pre_ft_metrics['sae']:.4f}  "
          f"TP={pre_ft_metrics['TP']}  FP={pre_ft_metrics['FP']}  "
          f"TN={pre_ft_metrics['TN']}  FN={pre_ft_metrics['FN']}")

    # -- Phase 2: Fine-tune ---------------------------------------------------
    print("  Phase 2: Fine-tune")
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=LR_FT)
    best_ft = float('inf'); best_ft_state = None; ft_counter = 0
    ft_history = {'train_loss': []}

    ft_start = time.time()
    for epoch in range(EPOCHS_FT):
        ep_start = time.time()
        tr_loss, _, _ = _run_epoch(model, ft_loader, criterion, ft_optimizer, device, True)
        ft_history['train_loss'].append(tr_loss)
        ep_time = time.time() - ep_start
        print(f"    FT Ep {epoch+1:2d}  loss={tr_loss:.5f}  time={ep_time:.1f}s")
        if tr_loss < best_ft:
            best_ft = tr_loss; ft_counter = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            ft_counter += 1
            if ft_counter >= PATIENCE_FT:
                print(f"    FT early stopping at epoch {epoch+1}"); break

    model.load_state_dict(best_ft_state)
    print(f"  Phase 2 total: {(time.time()-ft_start)/60:.1f} min")

    # -- Phase 3: Test --------------------------------------------------------
    _, to, tt = _run_epoch(model, te_loader, criterion, ft_optimizer, device, False)
    test_metrics = _metrics(ys.inverse_transform(tt).flatten(),
                             ys.inverse_transform(to).flatten())
    print(f"  Test AFTER  fine-tune: "
          f"F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  "
          f"SAE={test_metrics['sae']:.4f}  "
          f"TP={test_metrics['TP']}  FP={test_metrics['FP']}  "
          f"TN={test_metrics['TN']}  FN={test_metrics['FN']}")

    # -- Plot -----------------------------------------------------------------
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)
    plt.plot(ep, history['train_loss'], label='Train', color='blue')
    plt.plot(ep, history['val_loss'],   label='Val',   color='red')
    plt.title(f'Pretrain Loss — {redd_appliance}'); plt.xlabel('Epoch')
    plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 2)
    plt.plot(ep, [m['mae'] for m in history['val_metrics']], color='red', label='Val MAE')
    plt.axhline(pre_ft_metrics['mae'], color='steelblue',  linestyle='--', label='Test pre-FT')
    plt.axhline(test_metrics['mae'],   color='darkorange', linestyle='--', label='Test post-FT')
    plt.title(f'MAE — {redd_appliance}'); plt.xlabel('Epoch'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 3)
    plt.plot(ep, [m['sae'] for m in history['val_metrics']], color='purple', label='Val SAE')
    plt.axhline(pre_ft_metrics['sae'], color='steelblue',  linestyle='--', label='Test pre-FT')
    plt.axhline(test_metrics['sae'],   color='darkorange', linestyle='--', label='Test post-FT')
    plt.title(f'SAE — {redd_appliance}'); plt.xlabel('Epoch'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 4)
    plt.plot(ep, [m['f1'] for m in history['val_metrics']], color='red', label='Val F1')
    plt.axhline(pre_ft_metrics['f1'],  color='steelblue',  linestyle='--', label='Test pre-FT')
    plt.axhline(test_metrics['f1'],    color='darkorange', linestyle='--', label='Test post-FT')
    plt.title(f'F1 — {redd_appliance}'); plt.xlabel('Epoch'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 5)
    ft_ep = range(1, len(ft_history['train_loss']) + 1)
    plt.plot(ft_ep, ft_history['train_loss'], color='green', label='FT train loss')
    plt.title(f'Fine-tune Loss — {redd_appliance}'); plt.xlabel('FT Epoch')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
                    f'lnn_redd_{redd_appliance.replace(" ", "_")}_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- JSON -----------------------------------------------------------------
    config = {
        'appliance': redd_appliance,
        'dataset':   'REDD',
        'model':     'LiquidNetworkModel',
        'data_splits': {
            'pretrain':  'train_small.pkl[0]',
            'val':       'val_small.pkl[0]',
            'finetune':  'test_small.pkl[0]',
            'test':      'test_small.pkl[3] (House 3, 2011-04-30)',
        },
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'pretrain_params': {'lr': LR,    'epochs': EPOCHS,    'patience': PATIENCE},
        'finetune_params': {'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT},
        'test_metrics_before_finetune': {k: float(v) for k, v in pre_ft_metrics.items()},
        'test_metrics_after_finetune':  {k: float(v) for k, v in test_metrics.items()},
        'aggregates': _aggregates(history, test_metrics),
    }
    with open(os.path.join(save_dir,
                  f'lnn_redd_{redd_appliance.replace(" ", "_")}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics, pre_ft_metrics


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(data_dir=DEFAULT_DATA_DIR, hidden_size=64, dt=0.1):
    for req in ['train_small.pkl', 'val_small.pkl', 'test_small.pkl']:
        path = os.path.join(data_dir, req)
        if not os.path.exists(path):
            print(f"Error: {path} not found"); raise SystemExit(1)

    splits    = load_redd_splits(data_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir  = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..',
        f'models/lnn_redd_finetune_{timestamp}')
    all_results = {}
    wall_start  = time.time()

    for redd_app in REDD_APPLIANCES:
        print(f"\n{'='*60}\nLNN REDD — {redd_app}\n{'='*60}")
        app_dir = os.path.join(base_dir, redd_app.replace(' ', '_'))
        try:
            _, _, after, before = train_on_appliance(
                splits, redd_app, data_dir=data_dir,
                hidden_size=hidden_size, dt=dt, save_dir=app_dir)
            all_results[redd_app] = {
                'before_finetune': {k: float(v) for k, v in before.items()},
                'after_finetune':  {k: float(v) for k, v in after.items()},
            }
        except Exception as e:
            print(f"Error on {redd_app}: {e}")
            import traceback; traceback.print_exc()

    os.makedirs(base_dir, exist_ok=True)
    summary = {
        'timestamp':    timestamp,
        'model':        'LiquidNetworkModel',
        'dataset':      'REDD',
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'results': all_results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nLNN REDD complete.  Results → {base_dir}")
    for app, r in all_results.items():
        print(f"  {app:<15}  F1  "
              f"{r['before_finetune']['f1']:.4f} → {r['after_finetune']['f1']:.4f}  "
              f"MAE  {r['before_finetune']['mae']:.1f} → {r['after_finetune']['mae']:.1f}")
    print(f"Total wall-clock time: {(time.time()-wall_start)/60:.1f} min")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='LNN REDD fine-tuning')
    p.add_argument('--data-dir',    default=DEFAULT_DATA_DIR)
    p.add_argument('--hidden-size', type=int,   default=64)
    p.add_argument('--dt',          type=float, default=0.1)
    args = p.parse_args()
    run_all(args.data_dir, args.hidden_size, args.dt)
