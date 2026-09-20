# -----------------------------------------------------------------------------
# PROJECT: GOLDEN RIVER ORACLE - ROBUST MULTI-TIMEFRAME (APEX FUZZ EDITION)
# Asset: COMEX Gold Futures (GC=F) via Yahoo Finance
#
# INVENTORS & OWNERS: Cale Sutherland & Daniela Admidina
# CODER: Gemini
# -----------------------------------------------------------------------------

import os
import sys
import time
import math
import json
import pickle
import requests
import warnings
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

YAHOO_SYMBOL = "GC=F"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# -----------------------------
# CROSS-PLATFORM FOLDER SETUP
# -----------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "gold_oracle_signatures")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. STRICT TIMEFRAME ALIGNMENT
TIMEFRAMES = {
    "4hr": {"interval": "15m", "range": "30d",  "horizon": 16, "label": "4 HOUR PROJECTION (15m)", "dt_fmt": "%H:%M", "delta": timedelta(minutes=15)},
    "1d":  {"interval": "1h",  "range": "730d", "horizon": 24, "label": "24 HOUR PROJECTION (1H)", "dt_fmt": "%b %d %H", "delta": timedelta(hours=1)},
    "1wk": {"interval": "1d",  "range": "5y",   "horizon": 5,  "label": "1 WEEK PROJECTION (1D)",  "dt_fmt": "%b %d", "delta": timedelta(days=1)},
    "1mo": {"interval": "1d",  "range": "10y",  "horizon": 21, "label": "1 MONTH PROJECTION (1D)", "dt_fmt": "%b '%y", "delta": timedelta(days=1)}
}

MEMORY_MAX = 50000

# -----------------------------
# HELPERS & PHYSICS
# -----------------------------
def save_scaler(mean_vec, std_vec, filepath):
    try:
        with open(filepath, "w") as f: json.dump({"mean": mean_vec.tolist(), "std": std_vec.tolist()}, f)
    except: pass

def load_scaler(filepath):
    try:
        with open(filepath, "r") as f:
            d = json.load(f)
            return np.array(d["mean"], dtype=float), np.array(d["std"], dtype=float)
    except: return None, None

def fit_scaler_online(existing_mean, existing_std, new_vec, count):
    if existing_mean is None: return new_vec.copy(), np.maximum(np.abs(new_vec)*0.01, np.zeros_like(new_vec))
    alpha = max(0.005, 1.0 / max(1, count)) 
    mean = (1-alpha)*existing_mean + alpha*new_vec
    std = (1-alpha)*existing_std + alpha*np.abs(new_vec - mean)
    std = np.where(std == 0, 1e-6, std)
    return mean, std

def scale_vector(vec, mean_vec, std_vec):
    return (vec - mean_vec) / std_vec if mean_vec is not None and std_vec is not None else vec

def extract_pattern_vector(df, idx, length=5):
    if idx < length: return None
    subset = df.iloc[idx-length:idx+1]
    base = subset['open'].iloc[0]
    if base == 0: return None
    vec = []
    for i in range(len(subset)):
        row = subset.iloc[i]
        vec.extend([(row['close'] - row['open'])/base, (row['high'] - row['low'])/base])
    vec.extend(np.diff(subset['close'].values) / base)
    return np.array(vec)

def calculate_vessel_physics(momentum, tension, volatility):
    pitch_angle = momentum * 1000 
    heel_angle = tension * 500
    raw_tilt = pitch_angle - (heel_angle if pitch_angle > 0 else -heel_angle)
    tilt_angle = np.clip(raw_tilt, -45.0, 45.0)
    safe_vol = max(volatility, 0.0005) 
    stress_factor = tension / safe_vol
    return tilt_angle, stress_factor

def calculate_advanced_chop_bias(df, period=20, std_dev=2.0):
    if len(df) < period: return 0.0
    close_series = df['close']
    sma = close_series.rolling(window=period).mean()
    std = close_series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    bbw = (upper - lower) / sma
    return float(bbw.iloc[-1])

def weighted_euclidean(a, b, weights):
    diff = (a - b) * weights
    return np.sqrt(np.sum(diff * diff))

GHOST_MARKERS = 10
def sample_ghost_markers(pred_curve, realized_curve):
    if len(pred_curve) == 0 or len(realized_curve) == 0: return np.array([1.0])
    idx = np.linspace(0, min(len(pred_curve), len(realized_curve))-1, GHOST_MARKERS).astype(int)
    errors = []
    for i in idx: errors.append(abs(pred_curve[i] - realized_curve[i]))
    return np.array(errors)

# -----------------------------
# CORE LEARNING ENGINES
# -----------------------------
class AdaptationGovernor:
    def __init__(self):
        self.short_term_error = 0.5
        self.long_term_error = 0.5
        self.st_error_trace = deque(maxlen=50)   
        self.lt_error_trace = deque(maxlen=500)  
        self.learning_rate = 0.01
        self.blend_shift_rate = 0.1 
        self.update_count = 0

    def update(self, realized_error):
        self.update_count += 1
        self.st_error_trace.append(realized_error)
        self.lt_error_trace.append(realized_error)
        if self.update_count < 3: return
            
        self.short_term_error = np.mean(self.st_error_trace)
        self.long_term_error = np.mean(self.lt_error_trace)
        error_velocity = self.short_term_error - self.long_term_error
        
        if error_velocity > 0:
            self.learning_rate = np.clip(self.learning_rate + np.clip(error_velocity * 2.0, 0.01, 0.1), 0.02, 0.20)
            self.blend_shift_rate = np.clip(0.1 + (error_velocity * 0.5), 0.2, 0.6)
        else:
            self.learning_rate = np.clip(self.learning_rate * 0.9, 0.001, 0.02)
            self.blend_shift_rate = np.clip(0.1 + (error_velocity * 0.2), 0.01, 0.1)

class MetricLearner:
    def __init__(self, dim):
        self.weights = np.ones(dim, dtype=float)
        self.velocity = np.zeros(dim, dtype=float) 
    def update_from_errors(self, feature_vecs, errors, lr=0.01):
        clipped_errors = np.clip(errors, 0.0, 3.0) 
        grad = np.mean(np.abs(feature_vecs) * clipped_errors[:, None], axis=0)
        grad_norm = np.linalg.norm(grad) + 1e-9
        self.velocity = 0.9 * self.velocity + 0.1 * (grad / grad_norm)
        self.weights *= np.exp(-lr * self.velocity)
        self.weights = np.clip(self.weights, 0.05, 5.0) 
    def get_weights(self): return self.weights / (np.linalg.norm(self.weights) + 1e-9)

class MarketSanctuary:
    def __init__(self, memory_file, scaler_file):
        self.memory_file = memory_file
        self.scaler_file = scaler_file
        self.memories = []
        self.count = 0
        self.mean_vec, self.std_vec = load_scaler(self.scaler_file)
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "rb") as f:
                    self.memories = pickle.load(f)
                    self.count = len(self.memories)
            except: pass

    def _score_memory(self, mem):
        age_hours = (datetime.now() - mem.get("created_at", datetime.now())).total_seconds() / 3600.0
        recency_score = math.exp(-age_hours / 168.0) 
        perf = mem.get("error_ewma", 1.0)
        return recency_score * (1.0 / (1.0 + perf))

    def learn(self, pattern, outcome, volatility=0.0):
        stored_vec = scale_vector(pattern, self.mean_vec, self.std_vec) if self.mean_vec is not None else pattern.copy()
        fuzz_scale = max(0.001, np.mean(np.abs(pattern)) * 0.1 * (1.0 / (1.0 + volatility*10.0)))
        fuzzed = stored_vec + np.random.normal(0, fuzz_scale, size=len(stored_vec))

        now = datetime.now()
        base_entry = {"vector": stored_vec, "outcome": np.array(outcome), "created_at": now, "error_ewma": 1.0, "score": 1.0}
        fuzz_entry = {"vector": fuzzed, "outcome": np.array(outcome), "created_at": now, "error_ewma": 1.0, "score": 1.0}

        self.memories.extend([base_entry, fuzz_entry])
        self.count += 2

        self.mean_vec, self.std_vec = fit_scaler_online(self.mean_vec, self.std_vec, pattern, self.count)
        save_scaler(self.mean_vec, self.std_vec, self.scaler_file)

        if len(self.memories) > MEMORY_MAX:
            for m in self.memories: m["score"] = self._score_memory(m)
            self.memories.sort(key=lambda x: x["score"], reverse=True)
            self.memories = self.memories[:MEMORY_MAX]

        if self.count % 50 == 0:
            try:
                with open(self.memory_file, "wb") as f: pickle.dump(self.memories, f)
            except: pass

    def recall(self, current_pattern, top_k=20, metric_learner=None):
        if not self.memories or current_pattern is None: return []
        cur = scale_vector(current_pattern, self.mean_vec, self.std_vec) if self.mean_vec is not None else current_pattern
        w = metric_learner.get_weights() if metric_learner is not None else None

        scores = []
        for m in self.memories:
            try:
                mem_score = max(0.01, m.get("score", 1.0))
                dist = euclidean(cur, m["vector"])/mem_score if w is None else weighted_euclidean(cur, m["vector"], w)/mem_score
                scores.append((dist, m["outcome"], m))
            except: continue

        scores.sort(key=lambda x: x[0])
        return scores[:top_k]

# -----------------------------
# ADVANCED FUZZ ENGINE
# -----------------------------
class FuzzPacket:
    def __init__(self, value, direction, velocity):
        self.value = value       
        self.direction = direction 
        self.velocity = velocity   
        self.active = True

class MultiAdjustableLens:
    def __init__(self): self.focus_strength = 1.0
    def tune_lens(self, governor_drift, current_chop_pct):
        chop_modifier = 1.0 + (current_chop_pct * 5.0) 
        if governor_drift > 0: self.focus_strength = max(0.5, self.focus_strength - 0.1) * chop_modifier
        else: self.focus_strength = min(2.0, self.focus_strength + 0.1) / max(1.0, chop_modifier)
    def process(self, packet):
        packet.value *= self.focus_strength
        packet.velocity *= (self.focus_strength * 0.8)
        return packet

class MirrorNode:
    def __init__(self, position_index, reflection_decay):
        self.position = position_index 
        self.decay = reflection_decay  
    def reflect(self, packet):
        if packet.active:
            packet.value *= self.decay
            if packet.value < 0.0001: packet.active = False 
        return packet

class AdvancedFuzzEngine:
    def __init__(self, horizon):
        self.horizon = horizon
        self.lens = MultiAdjustableLens()
        self.mirrors = [
            MirrorNode(position_index=int(horizon*0.33), reflection_decay=0.7),
            MirrorNode(position_index=int(horizon*0.66), reflection_decay=0.4)
        ]

    def shoot_fuzz(self, base_curve, momentum_vector, volatility, governor_drift, chop_pct):
        curve = np.array(base_curve)
        if momentum_vector == 0: return curve
        self.lens.tune_lens(governor_drift, chop_pct)
        direction = 1 if momentum_vector > 0 else -1
        packet = FuzzPacket(value=abs(momentum_vector) * 5.0, direction=direction, velocity=volatility * 10.0)
        packet = self.lens.process(packet)
        
        fuzz_deformation = np.zeros(self.horizon)
        for t in range(self.horizon):
            if not packet.active: break
            for mirror in self.mirrors:
                if t == mirror.position: packet = mirror.reflect(packet)
            force = packet.value * packet.direction * (1 + packet.velocity)
            fuzz_deformation[t] = force
            packet.value *= 0.85 

        return curve + fuzz_deformation

def negotiate_paths(paths, weights, shift_rate=0.1):
    if len(paths) == 0: return np.zeros(len(paths[0]) if len(paths)>0 else 10)
    final_points = np.array([p[-1] for p in paths])
    median_dest = np.median(final_points)
    camp_a_idx = np.where(final_points >= median_dest)[0]
    camp_b_idx = np.where(final_points < median_dest)[0]
    
    weight_a = np.sum(weights[camp_a_idx]) if len(camp_a_idx) > 0 else 0.0
    weight_b = np.sum(weights[camp_b_idx]) if len(camp_b_idx) > 0 else 0.0
    total_weight = weight_a + weight_b + 1e-9
    
    prob_a = weight_a / total_weight
    prob_b = weight_b / total_weight
    
    shape_a = np.average(paths[camp_a_idx], axis=0, weights=weights[camp_a_idx]) if len(camp_a_idx) > 0 else np.zeros(len(paths[0]))
    shape_b = np.average(paths[camp_b_idx], axis=0, weights=weights[camp_b_idx]) if len(camp_b_idx) > 0 else np.zeros(len(paths[0]))
    
    return (shape_a * prob_a) + (shape_b * prob_b)

def update_memory_errors(sanctuary, pattern, realized_outcome, metric_learner, governor):
    matches = sanctuary.recall(pattern, top_k=10, metric_learner=metric_learner)
    if not matches: return
    all_marker_errors = []
    for dist, outcome, mem_obj in matches:
        try: 
            marker_errors = sample_ghost_markers(np.asarray(outcome), np.asarray(realized_outcome))
            err = np.clip(np.mean(marker_errors), 0.0, 5.0)
            all_marker_errors.append(err)
            
            prev = mem_obj.get("error_ewma", 1.0)
            mem_obj["error_ewma"] = 0.8 * prev + 0.2 * err
            mem_obj["score"] = sanctuary._score_memory(mem_obj)
            if metric_learner is not None:
                metric_learner.update_from_errors(np.array([mem_obj["vector"]]), np.array([err]), lr=governor.learning_rate)
        except: continue
    if all_marker_errors: governor.update(np.mean(all_marker_errors))

# -----------------------------
# DATA INGESTION
# -----------------------------
def fetch_timeframe_data(interval, range_val):
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{YAHOO_SYMBOL}?interval={interval}&range={range_val}"
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200: return None
        data = r.json()['chart']['result'][0]
        df = pd.DataFrame({
            "ts": np.array(data['timestamp']) * 1000,
            "open": data['indicators']['quote'][0]['open'],
            "high": data['indicators']['quote'][0]['high'],
            "low": data['indicators']['quote'][0]['low'],
            "close": data['indicators']['quote'][0]['close'],
        }).dropna().reset_index(drop=True)
        df["date"] = pd.to_datetime(df["ts"], unit="ms") + (datetime.now() - datetime.utcnow())
        return df
    except: return None

def fetch_live_price():
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{YAHOO_SYMBOL}?interval=1m&range=1d"
        r = requests.get(url, headers=HEADERS, timeout=4)
        if r.status_code == 200: return float(r.json()['chart']['result'][0]['meta']['regularMarketPrice'])
    except: pass
    return None

# -----------------------------
# MASTER SYSTEM INITIALIZATION
# -----------------------------
ENGINES = {}
for tf, cfg in TIMEFRAMES.items():
    mem_path = os.path.join(DATA_DIR, f"gold_memory_{tf}.pkl")
    scale_path = os.path.join(DATA_DIR, f"gold_scaler_{tf}.json")
    ENGINES[tf] = {
        "sanctuary": MarketSanctuary(mem_path, scale_path),
        "governor": AdaptationGovernor(),
        "fuzz": AdvancedFuzzEngine(cfg["horizon"]),
        "metric": None,
        "last_learned_ts": 0
    }

def bootstrap_history(tf_key):
    cfg = TIMEFRAMES[tf_key]
    engine = ENGINES[tf_key]
    if engine["sanctuary"].count > 50: return # Already bootstrapped
    
    print(f"[*] Bootstrapping {tf_key} Neural Matrix...")
    df = fetch_timeframe_data(cfg["interval"], cfg["range"])
    if df is None: return
    
    learned_count = 0
    for i in range(10, len(df) - cfg["horizon"] - 1):
        pat = extract_pattern_vector(df, i)
        if pat is not None:
            base_p = df['close'].iloc[i]
            future_p = df['close'].iloc[i+1 : i+1+cfg["horizon"]].values
            if len(future_p) == cfg["horizon"] and base_p > 0:
                actual_outcome = (future_p - base_p) / base_p
                volatility = df['close'].iloc[i-10:i].std() if i >= 10 else 0.0
                engine["sanctuary"].learn(pat, actual_outcome, volatility=volatility)
                learned_count += 1
    print(f"[+] {tf_key} Bootstrap complete. Absorbed {learned_count} signatures.")

def run_multi_grid_oracle():
    print("---------------------------------------------------------")
    print(" GOLDEN RIVER ORACLE - APEX FUZZ EDITION")
    print(" INVENTORS & OWNERS: Cale Sutherland & Daniela Admidina")
    print(" CALIBRATOR: Vanessa Georgou")
    print(" CODER: Gemini")
    print("---------------------------------------------------------")
    
    for tf in TIMEFRAMES.keys(): bootstrap_history(tf)

    plt.ion()
    # Establish the 2x2 layout from the very start to prevent backend crashing
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#050505')
    fig.suptitle("GOLDEN RIVER ORACLE - APEX FUZZ EDITION (COMEX GOLD)", color='#D4AF37', fontsize=12, fontweight='bold')

    # WARMUP COUNTDOWN PHASE
    WARMUP_SECONDS = 600  # 10 minutes (600 seconds)
    
    # Temporarily hide all 4 graphs
    for ax in axs.flatten():
        ax.set_facecolor('#050505')
        ax.axis('off')
        
    # Place a central text object for the timer
    timer_text = fig.text(0.5, 0.5, "", color='#D4AF37', fontfamily='monospace', 
                          fontsize=22, ha='center', va='center', fontweight='bold', alpha=0.9)
                          
    for remaining in range(WARMUP_SECONDS, 0, -1):
        timer_text.set_text(f"CALIBRATING ORACLE ENGINE\n\nALIGNING NEURAL PATHS\n\nSTANDBY... {remaining}s")
        plt.draw()
        plt.pause(1)

    print("[+] Calibration complete. Engaging graph matrix.")
    timer_text.remove() # Delete the text object so the graphs can show

    grid_map = [("4hr", axs[0, 0]), ("1d", axs[0, 1]), ("1wk", axs[1, 0]), ("1mo", axs[1, 1])]

    while True:
        live_price = fetch_live_price()
        if live_price is None:
            print("[-] Waiting on Yahoo Finance API (Live Price)... Retrying in 4s.")
            plt.pause(4)
            continue

        for tf_key, ax in grid_map:
            cfg = TIMEFRAMES[tf_key]
            engine = ENGINES[tf_key]
            df = fetch_timeframe_data(cfg["interval"], cfg["range"])
            
            if df is None or len(df) < 20: 
                print(f"[-] Waiting on Yahoo Finance API ({tf_key} historical data)...")
                continue
            
            # --- CONTINUOUS LEARNING LOOP ---
            mature_idx = len(df) - cfg["horizon"] - 2
            if mature_idx > 10:
                mature_ts = df['date'].iloc[mature_idx].timestamp()
                if mature_ts > engine["last_learned_ts"]:
                    past_pat = extract_pattern_vector(df, mature_idx)
                    if past_pat is not None:
                        base_p = df['close'].iloc[mature_idx]
                        future_p = df['close'].iloc[mature_idx+1 : mature_idx+1+cfg["horizon"]].values
                        if len(future_p) == cfg["horizon"] and base_p > 0:
                            actual_outcome = (future_p - base_p) / base_p
                            volatility = df['close'].iloc[mature_idx-10:mature_idx].std() if mature_idx >= 10 else 0.0
                            engine["sanctuary"].learn(past_pat, actual_outcome, volatility=volatility)
                            
                            if engine["metric"] is None: engine["metric"] = MetricLearner(len(past_pat))
                            update_memory_errors(engine["sanctuary"], past_pat, actual_outcome, engine["metric"], engine["governor"])
                            engine["last_learned_ts"] = mature_ts

            # --- LIVE PROJECTION ENGINE ---
            cur_pat = extract_pattern_vector(df, len(df)-1)
            if cur_pat is None: continue
            
            if engine["metric"] is None: engine["metric"] = MetricLearner(len(cur_pat))
            matches = engine["sanctuary"].recall(cur_pat, top_k=12, metric_learner=engine["metric"])
            
            if not matches: continue
            
            dists = np.array([m[0] for m in matches])
            paths_raw = [m[1] for m in matches]
            mean_dist = np.mean(dists) + 1e-9
            weights = np.exp(-dists / mean_dist)
            weights /= weights.sum()

            aligned = []
            for p in paths_raw:
                p = np.asarray(p)
                if p.size < cfg["horizon"]: p = np.pad(p, (0, cfg["horizon"] - p.size), 'edge')
                elif p.size > cfg["horizon"]: p = p[:cfg["horizon"]]
                aligned.append(p)
            paths = np.vstack(aligned) 
            
            avg_outcome = negotiate_paths(paths, weights, shift_rate=engine["governor"].blend_shift_rate)
            tension = float(np.std(paths))
            
            recent_closes = df['close'].tail(5).values
            momentum = (recent_closes[-1] - recent_closes[0]) / (recent_closes[0] + 1e-9)
            volatility = df['close'].tail(15).std() / df['close'].iloc[-1]
            chop_bbw = calculate_advanced_chop_bias(df)
            
            drift = engine["governor"].short_term_error - engine["governor"].long_term_error
            fuzzed_curve = engine["fuzz"].shoot_fuzz(avg_outcome, momentum, volatility, drift, chop_bbw)
            
            tilt, stress = calculate_vessel_physics(momentum, tension, volatility)
            dist_certainty = np.clip(1.0 / (1.0 + mean_dist), 0, 1)
            certainty = dist_certainty * 0.8 
            
            base_time = df['date'].iloc[-1]
            future_prices = (live_price * (1 + fuzzed_curve)).tolist()
            future_times = [base_time + (cfg["delta"] * (i + 1)) for i in range(cfg["horizon"])]
            target_p = future_prices[-1]
            move_pct = ((target_p - live_price) / live_price) * 100

            pred_p_plot = [live_price] + future_prices
            pred_t_plot = [base_time] + future_times
            recent_df = df.tail(45)

            # --- RENDERING HUD ---
            ax.clear() # This safely wipes the axis and restores its grid visibility automatically
            ax.set_facecolor('#0A0A0A')
            ax.grid(color='#D4AF37', linestyle=':', linewidth=0.5, alpha=0.15)
            
            ax.plot(recent_df['date'], recent_df['close'], color='#EAEAEA', linewidth=1.5, alpha=0.8)
            
            is_bullish = target_p >= live_price
            line_color = '#00FF66' if is_bullish else '#FF3366'
            
            fuzz_vol = live_price * tension
            fuzz_upper = [p + (fuzz_vol * (i/len(pred_p_plot))) for i, p in enumerate(pred_p_plot)]
            fuzz_lower = [p - (fuzz_vol * (i/len(pred_p_plot))) for i, p in enumerate(pred_p_plot)]
            ax.fill_between(pred_t_plot, fuzz_lower, fuzz_upper, color=line_color, alpha=0.15)
            
            ax.plot(pred_t_plot, pred_p_plot, color=line_color, linewidth=2, linestyle='--')
            ax.plot(base_time, live_price, marker='o', markersize=6, markerfacecolor='#D4AF37', markeredgecolor='white')

            stress_warn = "[! CAPSIZE !]" if stress > 10.0 else "NOMINAL"
            hud = (f"LIVE:   ${live_price:.2f}\n"
                   f"TARGET: ${target_p:.2f} ({move_pct:+.2f}%)\n"
                   f"CONF:   {certainty*100:.0f}%\n"
                   f"-------------------\n"
                   f"TILT:   {tilt:+.1f} DEG\n"
                   f"STRESS: {stress:.1f}x {stress_warn}\n"
                   f"CHOP:   {chop_bbw:.4f} (BBW)\n"
                   f"LR/DR:  {engine['governor'].learning_rate:.3f}")
            
            ax.text(0.03, 0.95, hud, transform=ax.transAxes, color='#D4AF37', fontfamily='monospace', 
                    fontsize=8, va='top', bbox=dict(facecolor='#111', edgecolor='#333', alpha=0.9))

            ax.xaxis.set_major_formatter(mdates.DateFormatter(cfg["dt_fmt"]))
            ax.tick_params(axis='x', rotation=20, colors='#666', labelsize=8)
            ax.tick_params(axis='y', colors='#666', labelsize=8)
            ax.set_title(cfg["label"], color='#D4AF37', fontsize=9, fontweight='bold')

        fig.tight_layout()
        plt.draw()
        plt.pause(25)

if __name__ == "__main__":
    run_multi_grid_oracle()
