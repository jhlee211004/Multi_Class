import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============================================================
# 1. Load PKL File
# ============================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

print("Loading monitored dataset...")
mon_raw = load_pickle("mon_standard.pkl")   # path 그대로 사용
print("→ Monitored sites loaded.")

# unmon load (fallback to 3000 file)
print("Loading unmonitored dataset...")
try:
    unmon_raw = load_pickle("unmon_standard10.pkl")
    print("→ Loaded unmonitored 10000 samples")
except:
    unmon_raw = load_pickle("unmon_standard10_3000.pkl")
    print("→ Loaded unmonitored 3000 samples")


# ============================================================
# 2. Convert PKL → (X1, X2, y)
# ============================================================
X1_mon, X2_mon, y_mon = [], [], []
URL_PER_SITE = 10
TOTAL_URLS_MON = 950   # 95 sites × 10 URLs
USE_SUBLABEL = False

print("Converting monitored traces...")
for i in range(TOTAL_URLS_MON):
    label = i if USE_SUBLABEL else i // URL_PER_SITE

    for trace in mon_raw[i]:    # each URL has 20 traces
        time_seq = []
        size_seq = []

        for c in trace:
            dr = 1 if c > 0 else -1
            time_seq.append(abs(c))
            size_seq.append(dr * 512)

        X1_mon.append(time_seq)
        X2_mon.append(size_seq)
        y_mon.append(label)

print(f"Loaded {len(y_mon)} monitored samples")


# Unmonitored conversion
print("Converting unmonitored traces...")
X1_unmon, X2_unmon, y_unmon = [], [], []

for trace in unmon_raw:
    time_seq = []
    size_seq = []

    for c in trace:
        dr = 1 if c > 0 else -1
        time_seq.append(abs(c))
        size_seq.append(dr * 512)

    X1_unmon.append(time_seq)
    X2_unmon.append(size_seq)
    y_unmon.append(-1)

y_mon = np.array(y_mon)
y_unmon = np.array(y_unmon)

print(f"Loaded {len(y_unmon)} unmonitored samples")


# ============================================================
# 3. Feature Extraction Function
# ============================================================
def extract_features(seq_sizes, timestamps):
    sizes = np.array(seq_sizes)
    times = np.array(timestamps)

    if len(times) < 2:
        return np.zeros(19)

    inter = np.diff(times)
    incoming = sizes < 0
    outgoing = sizes > 0
    total = len(sizes)

    # I. Volume & Direction
    num_in = np.sum(incoming)
    num_out = np.sum(outgoing)
    total_pkts = total
    sum_all = np.sum(sizes)
    alt_sum = np.sum(np.abs(sizes))

    # II. Ratio
    frac_in = num_in / total if total > 0 else 0
    frac_out = num_out / total if total > 0 else 0
    ratio_large = np.sum(np.abs(sizes) > 512) / total

    # III. Ordering / Burstiness
    out_idx = np.where(outgoing)[0]
    out_mean = np.mean(out_idx) if len(out_idx) > 0 else 0
    out_std = np.std(out_idx) if len(out_idx) > 0 else 0

    burst = 1
    max_burst = 1
    for i in range(1, total):
        if np.sign(sizes[i]) == np.sign(sizes[i-1]):
            burst += 1
        else:
            max_burst = max(max_burst, burst)
            burst = 1
    max_burst = max(max_burst, burst)

    first30 = sizes[:30]
    first30_in = np.sum(first30 < 0)
    first30_out = np.sum(first30 > 0)

    # IV. Temporal
    duration = times[-1] - times[0] if len(times) > 1 else 0
    mean_iat = np.mean(inter)
    std_iat = np.std(inter)
    pkts_per_sec = total / duration if duration > 0 else 0

    inc_idx = np.where(incoming)[0]
    out_idx = np.where(outgoing)[0]
    if len(inc_idx) > 0 and len(out_idx) > 0:
        in_t = times[inc_idx][0]
        out_t = times[out_idx][0]
        ratio_t = in_t / out_t if out_t != 0 else 0
    else:
        ratio_t = 0

    return np.array([
        num_in, num_out, total_pkts, sum_all, alt_sum,
        frac_in, frac_out, ratio_large,
        out_mean, out_std, max_burst, first30_in, first30_out,
        duration, mean_iat, std_iat, pkts_per_sec, ratio_t
    ])


# ============================================================
# 4. Apply Feature Extraction
# ============================================================
print("Extracting features…")

X_feat_mon = np.array([extract_features(X2_mon[i], X1_mon[i]) for i in tqdm(range(len(X1_mon)))])
X_feat_unmon = np.array([extract_features(X2_unmon[i], X1_unmon[i]) for i in tqdm(range(len(X1_unmon)))])

X_all = np.vstack([X_feat_mon, X_feat_unmon])
y_all = np.concatenate([y_mon, y_unmon])

print("Features extracted:", X_all.shape)


# ============================================================
# 5. Train/Test Split + Scaling
# ============================================================
print("Splitting train/test…")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
)

print("Scaling…")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Done!")
print("Train:", X_train_scaled.shape, "Test:", X_test_scaled.shape)
