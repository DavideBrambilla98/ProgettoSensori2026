import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# CONFIGURAZIONE
# ==========================================
DATASET_FILE = "ml_dataset_features.csv"
MODEL_OUTPUT_FILE = "hr_sensor_fusion_model.joblib"
MAX_BPM_RATE_PER_SEC = 12.0  # Vincolo fisiologico anti-salto [BPM/s]

# Solo feature adimensionali di qualità spettrale/rapporto tra sensori
# (escluse ampiezze grezze scg_rms, imu_std_tot, pcg_rms_* per evitare overfitting sulla pressione di appoggio)
FEATURE_COLS = [
    "scg_kurtosis",
    "bpm_diff_sensors",
    "snr_scg",
    "snr_pcg",
    "spec_ent_scg",
    "spec_ent_pcg",
    "snr_diff",      # snr_scg - snr_pcg (positivo => SCG più pulito)
    "ent_diff",      # spec_ent_pcg - spec_ent_scg (positivo => PCG più rumoroso)
    "bpm_ratio"      # bpm_pcg / bpm_scg (~2.0 => raddoppio armonico sul PCG)
]
TARGET_COL = "bpm_ground_truth"
BASE_COL = "bpm_baseline_fusion"
GROUP_COL = "run_id"


def apply_physiological_rate_limiter(bpm_series, time_s, max_rate_bpm_s=MAX_BPM_RATE_PER_SEC):
    """
    Vincolo fisico di continuità: un cuore umano non può variare di oltre ~12 BPM/s
    da una finestra alla successiva. Filtra artefatti impulsivi isolati.
    """
    filtered = np.array(bpm_series, dtype=float).copy()
    if len(filtered) < 2:
        return filtered

    # Inizializzazione robusta sui primi 3 campioni per evitare spike al t=0
    init_window = min(3, len(filtered))
    filtered[0] = np.median(filtered[:init_window])

    for i in range(1, len(filtered)):
        dt = max(float(time_s[i] - time_s[i - 1]), 0.5)
        max_delta = max_rate_bpm_s * dt
        delta = filtered[i] - filtered[i - 1]
        if abs(delta) > max_delta:
            filtered[i] = filtered[i - 1] + np.sign(delta) * max_delta

    return filtered


# ==========================================
# CARICAMENTO E CREAZIONE TARGET BINARIO PESATO
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"File '{DATASET_FILE}' non trovato. Esegui prima extract_ml_features.py."
        )

    df = pd.read_csv(DATASET_FILE)

    # Calcolo delle 3 feature relative prima del dropna
    df["snr_diff"] = df["snr_scg"] - df["snr_pcg"]
    df["ent_diff"] = df["spec_ent_pcg"] - df["spec_ent_scg"]
    df["bpm_ratio"] = df["bpm_pcg"] / (df["bpm_scg"] + 1e-5)

    req_cols = FEATURE_COLS + [TARGET_COL, BASE_COL, "bpm_scg", "bpm_pcg", "time_s"]
    df_clean = df.dropna(subset=req_cols).copy()

    n_samples = len(df_clean)
    n_runs = df_clean[GROUP_COL].nunique()
    print(f"--> Caricato '{DATASET_FILE}': {n_samples} finestre valide su {n_runs} acquisizioni distinte.")

    if n_runs < 2:
        raise ValueError("Servono almeno 2 acquisizioni (run_id diversi) per la validazione LOGO!")

    # Target binario:
    # 1 = SCG (IMU) più preciso del PCG
    # 0 = PCG (Mic) più preciso dell'SCG
    err_scg = np.abs(df_clean["bpm_scg"].values - df_clean[TARGET_COL].values)
    err_pcg = np.abs(df_clean["bpm_pcg"].values - df_clean[TARGET_COL].values)
    y_best_sensor = (err_scg < err_pcg).astype(int)

    # Peso del campione: più grande è la differenza di errore tra i due sensori,
    # più è critico che il modello indovini il sensore giusto
    sample_weights = np.abs(err_scg - err_pcg) + 0.1

    X = df_clean[FEATURE_COLS].values
    y_true = df_clean[TARGET_COL].values
    y_base = df_clean[BASE_COL].values
    bpm_scg = df_clean["bpm_scg"].values
    bpm_pcg = df_clean["bpm_pcg"].values
    time_s = df_clean["time_s"].values
    groups = df_clean[GROUP_COL].values

    # ==========================================
    # CLASSIFICATORE LINEARE REGOLARIZZATO (ANTI-OVERFITTING)
    # ==========================================
    models = {
        "Logistic Regression (C=0.5)": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, random_state=42)
        ),
        "Logistic Regression (C=1.0)": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, random_state=42)
        ),
    }

    # ==========================================
    # VALIDAZIONE LEAVE-ONE-RUN-OUT (LOGO)
    # ==========================================
    logo = LeaveOneGroupOut()

    mae_baseline = mean_absolute_error(y_true, y_base)
    rmse_baseline = np.sqrt(mean_squared_error(y_true, y_base))

    print("\n==================================================")
    print(" RISULTATI VALIDAZIONE (HARD SWITCH + RATE LIMITER)")
    print("==================================================")
    print(f"Baseline Deterministica (Fallback) -> MAE: {mae_baseline:.2f} BPM | RMSE: {rmse_baseline:.2f} BPM")
    print("-" * 74)

    results_oof_bpm = {}
    results_oof_probs = {}
    best_model_name = None
    best_mae = float("inf")

    # Addestriamo il classificatore SOLO dove i due sensori sono in reale disaccordo (> 4 BPM)
    conflict_mask = df_clean["bpm_diff_sensors"].values > 4.0

    for name, model in models.items():
        oof_prob_scg = np.zeros(len(y_true), dtype=float)
        oof_bpm = np.zeros(len(y_true), dtype=float)

        for train_idx, test_idx in logo.split(X, y_best_sensor, groups):
            valid_train = train_idx[conflict_mask[train_idx]]
            if len(valid_train) < 5:
                valid_train = train_idx

            model.fit(
                X[valid_train],
                y_best_sensor[valid_train],
                logisticregression__sample_weight=sample_weights[valid_train]
            )

            probs_fold = model.predict_proba(X[test_idx])[:, 1]
            oof_prob_scg[test_idx] = probs_fold

            # HARD SWITCH + VINCOLO FISIOLOGICO SULLA SINGOLA RUN DI TEST
            use_scg = probs_fold >= 0.5
            agree_mask = df_clean["bpm_diff_sensors"].values[test_idx] < 3.0
            raw_fold_bpm = np.where(
                agree_mask,
                0.5 * (bpm_scg[test_idx] + bpm_pcg[test_idx]),
                np.where(use_scg, bpm_scg[test_idx], bpm_pcg[test_idx])
            )
            oof_bpm[test_idx] = apply_physiological_rate_limiter(
                raw_fold_bpm, time_s[test_idx], MAX_BPM_RATE_PER_SEC
            )

        mae = mean_absolute_error(y_true, oof_bpm)
        rmse = np.sqrt(mean_squared_error(y_true, oof_bpm))
        results_oof_bpm[name] = oof_bpm
        results_oof_probs[name] = oof_prob_scg

        improvement = mae_baseline - mae
        sign = "+" if improvement >= 0 else ""
        print(f"{name:<30} -> MAE: {mae:.2f} BPM | RMSE: {rmse:.2f} BPM (Diff vs Baseline: {sign}{improvement:.2f} BPM)")

        if mae < best_mae:
            best_mae = mae
            best_model_name = name

    print("==================================================")
    print(f"--> Miglior modello: {best_model_name} (MAE: {best_mae:.2f} BPM)\n")

    # ==========================================
    # ADDESTRAMENTO FINALE SU TUTTI I DATI E SALVATAGGIO
    # ==========================================
    best_model = models[best_model_name]
    valid_all = np.where(conflict_mask)[0]
    if len(valid_all) < 5:
        valid_all = np.arange(len(y_true))

    best_model.fit(
        X[valid_all],
        y_best_sensor[valid_all],
        logisticregression__sample_weight=sample_weights[valid_all]
    )

    joblib.dump({
        "model": best_model,
        "model_name": best_model_name,
        "feature_cols": FEATURE_COLS,
        "fusion_mode": "hard_switch",
        "max_bpm_rate_per_sec": MAX_BPM_RATE_PER_SEC
    }, MODEL_OUTPUT_FILE)
    print(f"--> Modello finale salvato in '{MODEL_OUTPUT_FILE}'.")

    # ==========================================
    # ANALISI IMPORTANZA DELLE FEATURE & GRAFICI
    # ==========================================
    df_clean["ml_pred"] = results_oof_bpm[best_model_name]
    df_clean["prob_scg"] = results_oof_probs[best_model_name]

    run_summary = []
    for run_id, grp in df_clean.groupby(GROUP_COL):
        run_summary.append({
            "run_id": run_id,
            "MAE_Baseline": mean_absolute_error(grp[TARGET_COL], grp[BASE_COL]),
            "MAE_ML": mean_absolute_error(grp[TARGET_COL], grp["ml_pred"]),
            "Prob_Media_SCG": grp["prob_scg"].mean()
        })
    df_runs = pd.DataFrame(run_summary)
    print("\n--- CONFRONTO PER SINGOLA ACQUISIZIONE ---")
    print(df_runs.to_string(index=False))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    target_plot_run = "raw_data_sim_rumore_1" if "raw_data_sim_rumore_1" in df_clean[GROUP_COL].values else df_clean[GROUP_COL].iloc[-1]
    sample_run = df_clean[df_clean[GROUP_COL] == target_plot_run]
    axs[0, 0].plot(sample_run["time_s"], sample_run[TARGET_COL], "k--", linewidth=2, label="Ground Truth")
    axs[0, 0].plot(sample_run["time_s"], sample_run[BASE_COL], color="orange", alpha=0.8, label="Baseline Deterministica")
    axs[0, 0].plot(sample_run["time_s"], sample_run["ml_pred"], color="green", linewidth=2, label=f"ML ({best_model_name})")
    axs[0, 0].set_title(f"1. Predizione Out-of-Fold ({target_plot_run})", fontweight="bold")
    axs[0, 0].set_xlabel("Tempo [s]")
    axs[0, 0].set_ylabel("Frequenza [BPM]")
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    axs[0, 1].scatter(y_true, results_oof_bpm[best_model_name], alpha=0.6, color="teal", edgecolors="k", s=30, label="Predizioni ML")
    min_val, max_val = min(y_true.min(), 45), max(y_true.max(), 180)
    axs[0, 1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideale (y = x)")
    axs[0, 1].set_title("2. Accuratezza Complessiva (Reale vs Predetto)", fontweight="bold")
    axs[0, 1].set_xlabel("Ground Truth [BPM]")
    axs[0, 1].set_ylabel("BPM Predetto dal ML [BPM]")
    axs[0, 1].legend(loc="upper left")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    x_idx = np.arange(len(df_runs))
    width = 0.35
    axs[1, 0].bar(x_idx - width/2, df_runs["MAE_Baseline"], width, label="Baseline Deterministica", color="coral")
    axs[1, 0].bar(x_idx + width/2, df_runs["MAE_ML"], width, label=f"ML ({best_model_name})", color="seagreen")
    axs[1, 0].set_xticks(x_idx)
    axs[1, 0].set_xticklabels(df_runs["run_id"], rotation=25, ha="right", fontsize=8)
    axs[1, 0].set_title("3. Errore Medio (MAE) per Singola Acquisizione", fontweight="bold")
    axs[1, 0].set_ylabel("MAE [BPM]")
    axs[1, 0].legend()
    axs[1, 0].grid(True, axis="y", linestyle="--", alpha=0.5)

    coefs = np.abs(best_model.named_steps["logisticregression"].coef_[0])
    sorted_idx = np.argsort(coefs)
    axs[1, 1].barh(np.array(FEATURE_COLS)[sorted_idx], coefs[sorted_idx], color="slateblue")
    axs[1, 1].set_title("4. Peso Assoluto Coefficienti (Logistic)", fontweight="bold")
    axs[1, 1].set_xlabel("|Coefficiente Standardizzato|")
    axs[1, 1].grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()