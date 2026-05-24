"""
Advanced LNN v5 — REDD Dataset (no fine-tuning)

Same multi-timescale, event-aware, attentive-pooling architecture as v5,
trained directly on REDD pkl splits in a single train → val → test pipeline.

REDD data format  (data/redd/*.pkl):
  train_small.pkl  → list of 1  DataFrame, cols: main / dish washer / fridge / microwave / washer dryer
  val_small.pkl    → list of 1  DataFrame
  test_small.pkl   → list of 6  DataFrames (one per house); all are concatenated for evaluation

Column mapping:
  REDD 'main'         → aggregate input
  REDD 'dish washer'  → dishwasher
  REDD 'fridge'       → fridge
  REDD 'microwave'    → microwave
  REDD 'washer dryer' → washing_machine
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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Source Code'))
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'redd')

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

# Map from our appliance key → REDD column name
REDD_COL = {
    'dishwasher':      'dish washer',
    'fridge':          'fridge',
    'microwave':       'microwave',
    'washing_machine': 'washer dryer',
}

THRESHOLD = 10.0

EPOCHS  = 80;  PATIENCE = 20;  LR = 1e-3
BATCH   = 32;  WIN      = 100; STRIDE = 5

# Appliance-specific tau ranges (same as v5)
APPLIANCE_TAU = {
    'dishwasher':      {'fast': (0.01, 1.0),  'slow': (0.5,  8.0)},
    'fridge':          {'fast': (0.05, 2.0),  'slow': (0.5,  6.0)},
    'microwave':       {'fast': (0.01, 0.5),  'slow': (0.1,  2.0)},
    'washing_machine': {'fast': (0.05, 2.0),  'slow': (1.0, 12.0)},
}


# ---------------------------------------------------------------------------
# Model  (identical to v5)
# ---------------------------------------------------------------------------

class MultiTimescaleLiquidLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1,
                 tau_fast=(0.01, 1.0), tau_slow=(0.5, 10.0)):
        super().__init__()
        assert hidden_size % 2 == 0
        self.half = hidden_size // 2
        self.dt   = dt
        self.tau_fast_min, self.tau_fast_max = tau_fast
        self.tau_slow_min, self.tau_slow_max = tau_slow

        ctx_size = input_size + hidden_size

        self.fast_proj = nn.Linear(input_size, self.half)
        self.fast_rec  = nn.Parameter(torch.empty(self.half, self.half))
        self.fast_tau  = nn.Linear(ctx_size, self.half)
        self.fast_gate = nn.Linear(ctx_size, self.half)
        nn.init.xavier_uniform_(self.fast_rec)

        self.slow_proj = nn.Linear(input_size, self.half)
        self.slow_rec  = nn.Parameter(torch.empty(self.half, self.half))
        self.slow_tau  = nn.Linear(ctx_size, self.half)
        self.slow_gate = nn.Linear(ctx_size, self.half)
        nn.init.xavier_uniform_(self.slow_rec)

    def forward(self, xe, h_fast=None, h_slow=None):
        B = xe.size(0)
        if h_fast is None: h_fast = torch.zeros(B, self.half, device=xe.device)
        if h_slow is None: h_slow = torch.zeros(B, self.half, device=xe.device)

        ctx = torch.cat([xe, h_fast, h_slow], dim=1)

        f_inp  = self.fast_proj(xe)
        f_rec  = torch.matmul(h_fast, self.fast_rec)
        f_gate = torch.sigmoid(self.fast_gate(ctx))
        f_tau  = (self.tau_fast_min
                  + (self.tau_fast_max - self.tau_fast_min)
                  * torch.sigmoid(self.fast_tau(ctx)))
        dh_f   = ((-h_fast / f_tau) + f_gate * torch.tanh(f_inp + f_rec)) * self.dt
        h_fast_new = (h_fast + dh_f).clamp(-10.0, 10.0)

        s_inp  = self.slow_proj(xe)
        s_rec  = torch.matmul(h_slow, self.slow_rec)
        s_gate = torch.sigmoid(self.slow_gate(ctx))
        s_tau  = (self.tau_slow_min
                  + (self.tau_slow_max - self.tau_slow_min)
                  * torch.sigmoid(self.slow_tau(ctx)))
        dh_s   = ((-h_slow / s_tau) + s_gate * torch.tanh(s_inp + s_rec)) * self.dt
        h_slow_new = (h_slow + dh_s).clamp(-10.0, 10.0)

        return h_fast_new, h_slow_new


class MultiTimescaleLNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, dt=0.1,
                 tau_fast=(0.01, 1.0), tau_slow=(0.5, 10.0)):
        super().__init__()
        self.hidden_size = hidden_size
        self.lnn  = MultiTimescaleLiquidLayer(
            input_size + 1, hidden_size, dt, tau_fast, tau_slow)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.size()
        e = torch.zeros_like(x)
        e[:, 1:, :] = (x[:, 1:, :] - x[:, :-1, :]).abs()

        h_fast = h_slow = None
        states = []
        for t in range(T):
            xe = torch.cat([x[:, t, :], e[:, t, :]], dim=1)
            h_fast, h_slow = self.lnn(xe, h_fast, h_slow)
            states.append(torch.cat([h_fast, h_slow], dim=1))

        states  = torch.stack(states, dim=1)
        scores  = self.attn(states)
        weights = F.softmax(scores, dim=1)
        context = (weights * states).sum(dim=1)
        return self.fc(context)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class REDDDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def load_redd_splits(dataset_dir):
    """Load train/val/test pkl files. Returns dict of DataFrames."""
    splits = {}
    for name in ('train_small', 'val_small'):
        path = os.path.join(dataset_dir, f'{name}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        df = pd.concat(data, ignore_index=True)
        splits[name] = df
        print(f"  {name:15s}: {len(df):7,} rows")
    # Test: House 3 (index 3), day 2011-04-30
    test_path = os.path.join(dataset_dir, 'test_small.pkl')
    with open(test_path, 'rb') as f:
        test_list = pickle.load(f)
    test_df = test_list[3]
    test_data = test_df[
        pd.DatetimeIndex(test_df.index).date == pd.Timestamp('2011-04-30').date()
    ].reset_index(drop=True)
    splits['test_small'] = test_data
    print(f"  {'test_small':15s}: {len(test_data):7,} rows  (House 3, 2011-04-30)")
    return splits


def create_sequences(df, appliance):
    """Slide a window over mains; label = appliance value at window midpoint."""
    mains = df['main'].values
    tgts  = df[REDD_COL[appliance]].values
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

def train_on_appliance(splits, appliance, hidden_size=64, dt=0.1,
                       save_dir='models/advanced_lnn_v5_redd'):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tau_cfg  = APPLIANCE_TAU[appliance]
    tau_fast = tau_cfg['fast']
    tau_slow = tau_cfg['slow']
    print(f"\nAppliance: {appliance}  |  device: {device}  "
          f"tau_fast={tau_fast}  tau_slow={tau_slow}")

    X_tr, y_tr = create_sequences(splits['train_small'], appliance)
    X_va, y_va = create_sequences(splits['val_small'],   appliance)
    X_te, y_te = create_sequences(splits['test_small'],  appliance)

    xs = MinMaxScaler(); ys = MinMaxScaler()
    X_tr = xs.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = xs.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = xs.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)
    y_tr = ys.fit_transform(y_tr)
    y_va = ys.transform(y_va)
    y_te = ys.transform(y_te)

    mk_loader = lambda X, y, shuf: torch.utils.data.DataLoader(
        REDDDataset(X, y), batch_size=BATCH, shuffle=shuf)
    tr_loader = mk_loader(X_tr, y_tr, True)
    va_loader = mk_loader(X_va, y_va, False)
    te_loader = mk_loader(X_te, y_te, False)

    model = MultiTimescaleLNNModel(
        input_size=1, hidden_size=hidden_size, output_size=1, dt=dt,
        tau_fast=tau_fast, tau_slow=tau_slow).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history    = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val   = float('inf'); best_state = None; counter = 0

    train_start = time.time()
    for epoch in range(EPOCHS):
        ep_start = time.time()
        tr_loss, _, _   = _run_epoch(model, tr_loader, criterion, optimizer, device, True)
        va_loss, vo, vt = _run_epoch(model, va_loader, criterion, optimizer, device, False)
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
                       {'input_size': 1, 'output_size': 1, 'hidden_size': hidden_size,
                        'dt': dt, 'tau_fast': tau_fast, 'tau_slow': tau_slow},
                       {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                        'appliance': appliance},
                       m, os.path.join(save_dir, f'{appliance}_best.pth'))
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"  Training total: {(time.time()-train_start)/60:.1f} min")

    # -- Test -----------------------------------------------------------------
    _, to, tt = _run_epoch(model, te_loader, criterion, optimizer, device, False)
    test_metrics = _metrics(ys.inverse_transform(tt).flatten(),
                             ys.inverse_transform(to).flatten())
    print(f"  Test: F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  "
          f"SAE={test_metrics['sae']:.4f}  "
          f"TP={test_metrics['TP']}  FP={test_metrics['FP']}  "
          f"TN={test_metrics['TN']}  FN={test_metrics['FN']}")

    # -- Plot -----------------------------------------------------------------
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(ep, history['train_loss'], label='Train', color='blue')
    plt.plot(ep, history['val_loss'],   label='Val',   color='red')
    plt.title(f'Loss — {appliance}'); plt.xlabel('Epoch')
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(ep, [m['f1'] for m in history['val_metrics']], color='red', label='Val F1')
    plt.axhline(test_metrics['f1'], color='darkorange', linestyle='--', label='Test F1')
    plt.title(f'F1 — {appliance}'); plt.xlabel('Epoch')
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(ep, [m['mae'] for m in history['val_metrics']], color='red', label='Val MAE')
    plt.axhline(test_metrics['mae'], color='darkorange', linestyle='--', label='Test MAE')
    plt.title(f'MAE — {appliance}'); plt.xlabel('Epoch')
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(ep, [m['sae'] for m in history['val_metrics']], color='purple', label='Val SAE')
    plt.axhline(test_metrics['sae'], color='darkorange', linestyle='--', label='Test SAE')
    plt.title(f'SAE — {appliance}'); plt.xlabel('Epoch')
    plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'advanced_lnn_v5_redd_{appliance}_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- JSON -----------------------------------------------------------------
    config = {
        'appliance': appliance,
        'dataset':   'REDD',
        'model':     'MultiTimescaleLNNModel (Advanced LNN v5)',
        'architecture': {
            'multi_timescale':    'fast + slow hidden streams, each (hidden/2)',
            'event_aware_tau':    'e_t = |x_t - x_{t-1}| fed into tau and gate',
            'attentive_pooling':  'scalar softmax attention over all T states',
            'appliance_specific_tau': True,
        },
        'tau': {'fast': list(tau_fast), 'slow': list(tau_slow)},
        'model_params':   {'hidden_size': hidden_size, 'dt': dt},
        'train_params':   {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'test_metrics':   {k: float(v) for k, v in test_metrics.items()},
        'aggregates':     _aggregates(history, test_metrics),
    }
    with open(os.path.join(save_dir, f'advanced_lnn_v5_redd_{appliance}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, dt=0.1):
    print("Loading REDD splits...")
    splits    = load_redd_splits(dataset_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir  = f'models/advanced_lnn_v5_redd_{timestamp}'
    all_results = {}
    wall_start  = time.time()

    for app in APPLIANCES:
        print(f"\n{'='*60}\nAdvanced LNN v5 REDD — {app}\n{'='*60}")
        app_dir = os.path.join(base_dir, app)
        try:
            _, _, test_m = train_on_appliance(
                splits, app, hidden_size=hidden_size, dt=dt, save_dir=app_dir)
            all_results[app] = {k: float(v) for k, v in test_m.items()}
        except Exception as e:
            print(f"Error on {app}: {e}")
            import traceback; traceback.print_exc()

    os.makedirs(base_dir, exist_ok=True)
    summary = {
        'timestamp':  timestamp,
        'model':      'MultiTimescaleLNNModel (Advanced LNN v5)',
        'dataset':    'REDD',
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'appliance_tau': {k: {'fast': list(v['fast']), 'slow': list(v['slow'])}
                          for k, v in APPLIANCE_TAU.items()},
        'results': all_results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN v5 REDD complete.  Results → {base_dir}")
    for app, r in all_results.items():
        tau = APPLIANCE_TAU[app]
        print(f"  {app:<20}  F1={r['f1']:.4f}  MAE={r['mae']:.1f}  SAE={r['sae']:.4f}  "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}  "
              f"tau_fast={tau['fast']}  tau_slow={tau['slow']}")
    print(f"Total wall-clock time: {(time.time()-wall_start)/60:.1f} min")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Advanced LNN v5 on REDD: multi-timescale + event-aware + attention (no fine-tune)')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR,
                   help='Path to folder containing train_small.pkl / val_small.pkl / test_small.pkl')
    p.add_argument('--hidden-size',  type=int,   default=64)
    p.add_argument('--dt',           type=float, default=0.1)
    args = p.parse_args()
    run_all(args.dataset_dir, args.hidden_size, args.dt)
