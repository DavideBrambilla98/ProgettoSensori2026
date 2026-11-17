import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# ==========================================
# CONFIGURAZIONE
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATTERN = os.path.join(BASE_DIR, "raw_data_*.csv")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth.csv")
FIXED_GROUND_TRUTH_BPM = None
OUTPUT_DATASET_FILE = os.path.join(BASE_DIR, "ml_dataset_features.csv")

FS_UNIFORM = 100.0
WINDOW_SEC = 12.0
STEP_SEC = 1.0
N_FFT = 8192

MIN_BPM, MAX_BPM = 45.0, 180.0
LOW_CUT_HZ = MIN_BPM / 60.0   # 0.75 Hz
HIGH_CUT_HZ = MAX_BPM / 60.0  # 3.00 Hz
PCG_LOW_HZ, PCG_HIGH_HZ = 0.5, 6.0  # Banda larga per preservare la 2a armonica (2f) dell'HPS
SNR_THRESHOLD = 4.0

# ==========================================
# FUNZIONI DI FILTRAGGIO E FEATURE EXTRACTION
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def extract_features_from_raw(raw_csv_path, gt_df=None, fixed_gt_bpm=None):
    """Estrae feature temporali, spettrali e di sensor fusion da un singolo file raw_data_*.csv."""
    df_raw = pd.read_csv(raw_csv_path)
    run_name = os.path.basename(raw_csv_path).replace(".csv", "")

    t_raw = df_raw["timestamp_ms"].values
    ax_raw = df_raw["ax"].values
    ay_raw = df_raw["ay"].values
    az_raw = df_raw["az"].values
    mic_raw = df_raw["mic_rms"].values

    # 1. Ricampionamento uniforme a 100 Hz
    t_sec_raw = (t_raw - t_raw[0]) / 1000.0
    unique_mask = np.concatenate(([True], np.diff(t_sec_raw) > 0))
    t_sec_raw = t_sec_raw[unique_mask]
    ax_raw, ay_raw, az_raw, mic_raw = (
        ax_raw[unique_mask], ay_raw[unique_mask], az_raw[unique_mask], mic_raw[unique_mask]
    )

    t_sec = np.arange(0, t_sec_raw[-1], 1.0 / FS_UNIFORM)
    ax = np.interp(t_sec, t_sec_raw, ax_raw)
    ay = np.interp(t_sec, t_sec_raw, ay_raw)
    az = np.interp(t_sec, t_sec_raw, az_raw)
    pcg_env_raw = np.interp(t_sec, t_sec_raw, mic_raw)

    # 2. Pre-processing SCG (5-25 Hz + Inviluppo Hilbert -> 0.75-3.0 Hz)
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    a_mag -= np.mean(a_mag)
    scg_impulses = apply_bandpass(a_mag, lowcut=5.0, highcut=25.0, fs=FS_UNIFORM, order=4)
    scg_env = np.abs(hilbert(scg_impulses))
    scg_filtered = apply_bandpass(scg_env, lowcut=LOW_CUT_HZ, highcut=HIGH_CUT_HZ, fs=FS_UNIFORM, order=4)
    scg_filtered -= np.mean(scg_filtered)

    # 3. Pre-processing PCG (Inviluppo RMS -> banda larga 0.5-6.0 Hz per HPS)
    pcg_filtered = apply_bandpass(pcg_env_raw, lowcut=PCG_LOW_HZ, highcut=PCG_HIGH_HZ, fs=FS_UNIFORM, order=4)
    pcg_filtered -= np.mean(pcg_filtered)

    # 4. Finestre scorrevoli
    win_samples = int(WINDOW_SEC * FS_UNIFORM)
    step_samples = int(STEP_SEC * FS_UNIFORM)
    window = np.hanning(win_samples)

    fft_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / FS_UNIFORM)
    bpm_axis = fft_freqs * 60.0
    mask = (bpm_axis >= MIN_BPM) & (bpm_axis <= MAX_BPM)
    bpm_valid = bpm_axis[mask]

    rows = []

    for start in range(0, len(scg_filtered) - win_samples + 1, step_samples):
        end = start + win_samples
        t_center = t_sec[start + win_samples // 2]

        # Segmenti grezzi (per rilevare artefatti di movimento o rumore ambientale)
        seg_ax, seg_ay, seg_az = ax[start:end], ay[start:end], az[start:end]
        seg_impulses = scg_impulses[start:end]
        seg_pcg_raw = pcg_env_raw[start:end]

        # Segmenti filtrati e finestrati per FFT
        seg_scg = scg_filtered[start:end] * window
        seg_pcg = pcg_filtered[start:end] * window

        # Feature nel dominio del tempo (utili al ML per riconoscere artefatti/movimenti)
        imu_std_tot = np.sqrt(np.var(seg_ax) + np.var(seg_ay) + np.var(seg_az))
        scg_rms = np.sqrt(np.mean(seg_impulses**2))
        scg_kurtosis = np.mean((seg_impulses - np.mean(seg_impulses))**4) / (np.var(seg_impulses)**2 + 1e-9)
        pcg_rms_mean = np.mean(seg_pcg_raw)
        pcg_rms_std = np.std(seg_pcg_raw)

        # FFT + HPS
        fft_scg = np.abs(np.fft.rfft(seg_scg, n=N_FFT))
        fft_pcg_plain = np.abs(np.fft.rfft(seg_pcg, n=N_FFT))

        fft_pcg_comp = fft_pcg_plain[::2]
        n_hps = len(fft_pcg_comp)
        fft_pcg_hps = fft_pcg_plain.copy()
        fft_pcg_hps[:n_hps] = fft_pcg_plain[:n_hps] * fft_pcg_comp

        # Normalizzazione
        scg_norm = np.zeros_like(fft_scg)
        pcg_norm = np.zeros_like(fft_pcg_hps)
        pcg_plain_norm = np.zeros_like(fft_pcg_plain)

        if np.max(fft_scg[mask]) > 0:
            scg_norm[mask] = fft_scg[mask] / np.max(fft_scg[mask])
        if np.max(fft_pcg_hps[mask]) > 0:
            pcg_norm[mask] = fft_pcg_hps[mask] / np.max(fft_pcg_hps[mask])
        if np.max(fft_pcg_plain[mask]) > 0:
            pcg_plain_norm[mask] = fft_pcg_plain[mask] / np.max(fft_pcg_plain[mask])

        # Feature spettrali: picchi, SNR (calcolato SENZA HPS per entrambi) ed entropia
        bpm_scg = bpm_valid[np.argmax(scg_norm[mask])]
        bpm_pcg = bpm_valid[np.argmax(pcg_norm[mask])]

        snr_scg = np.max(scg_norm[mask]) / (np.mean(scg_norm[mask]) + 1e-6)
        snr_pcg = np.max(pcg_plain_norm[mask]) / (np.mean(pcg_plain_norm[mask]) + 1e-6)

        # Entropia spettrale normalizzata (misura quanto lo spettro è "pulito" vs rumoroso)
        p_scg = scg_norm[mask] / (np.sum(scg_norm[mask]) + 1e-9)
        p_pcg = pcg_norm[mask] / (np.sum(pcg_norm[mask]) + 1e-9)
        spec_ent_scg = -np.sum(p_scg * np.log2(p_scg + 1e-9)) / np.log2(len(p_scg))
        spec_ent_pcg = -np.sum(p_pcg * np.log2(p_pcg + 1e-9)) / np.log2(len(p_pcg))

        # Sensor Fusion classica (baseline deterministica)
        if snr_scg >= SNR_THRESHOLD and snr_pcg >= SNR_THRESHOLD:
            fft_fused = scg_norm * pcg_norm
        elif snr_pcg > snr_scg:
            fft_fused = 0.8 * pcg_norm + 0.2 * scg_norm
        else:
            fft_fused = 0.8 * scg_norm + 0.2 * pcg_norm

        bpm_baseline_fusion = bpm_valid[np.argmax(fft_fused[mask])]
        bpm_diff_sensors = abs(bpm_scg - bpm_pcg)

        # Associazione Ground Truth (interpolata al tempo t_center)
        bpm_gt = np.nan
        if gt_df is not None:
            bpm_gt = np.interp(t_center, gt_df["time_s"].values, gt_df["bpm_real"].values)
        elif fixed_gt_bpm is not None:
            bpm_gt = float(fixed_gt_bpm)

        rows.append({
            "run_id": run_name,
            "time_s": round(t_center, 1),
            # Feature temporali (qualità segnale / artefatti)
            "imu_std_tot": round(imu_std_tot, 5),
            "scg_rms": round(scg_rms, 5),
            "scg_kurtosis": round(scg_kurtosis, 3),
            "pcg_rms_mean": round(pcg_rms_mean, 2),
            "pcg_rms_std": round(pcg_rms_std, 2),
            # Feature spettrali
            "bpm_scg": round(bpm_scg, 2),
            "bpm_pcg": round(bpm_pcg, 2),
            "bpm_diff_sensors": round(bpm_diff_sensors, 2),
            "snr_scg": round(snr_scg, 3),
            "snr_pcg": round(snr_pcg, 3),
            "spec_ent_scg": round(spec_ent_scg, 4),
            "spec_ent_pcg": round(spec_ent_pcg, 4),
            # Stima deterministica e Target ML
            "bpm_baseline_fusion": round(bpm_baseline_fusion, 2),
            "bpm_ground_truth": round(bpm_gt, 2) if not np.isnan(bpm_gt) else np.nan
        })

    return pd.DataFrame(rows)

# ==========================================
# MAIN EXECUTION: COSTRUZIONE DATASET & VALUTAZIONE
# ==========================================
if __name__ == "__main__":
    raw_files = sorted(glob.glob(RAW_DATA_PATTERN))
    if not raw_files:
        raise FileNotFoundError(
            f"Nessun file trovato con pattern '{RAW_DATA_PATTERN}'. "
            "Esegui prima Analysis_sensorFusionFallBack.py per generare i CSV."
        )

    gt_df = None
    if os.path.exists(GROUND_TRUTH_FILE):
        gt_df = pd.read_csv(GROUND_TRUTH_FILE)
        print(f"--> Caricata Ground Truth da '{GROUND_TRUTH_FILE}' ({len(gt_df)} punti).")
    elif FIXED_GROUND_TRUTH_BPM is not None:
        print(f"--> Uso Ground Truth costante: {FIXED_GROUND_TRUTH_BPM} BPM.")
    else:
        print("ATTENZIONE: Nessun file 'ground_truth.csv' trovato e FIXED_GROUND_TRUTH_BPM=None.")
        print("Le feature verranno estratte comunque, ma il target 'bpm_ground_truth' sarà NaN.")

    all_dfs = []
    for path in raw_files:
        print(f"Elaborazione {path}...")
        # Cerca un file ground truth dedicato: es. raw_data_XXX.csv -> gt_XXX.csv
        gt_path = path.replace("raw_data_", "gt_")
        run_gt_df = None

        if os.path.exists(gt_path):
            run_gt_df = pd.read_csv(gt_path)
            print(f"  -> Associata Ground Truth dedicata: '{gt_path}'")
        elif os.path.exists(GROUND_TRUTH_FILE):
            run_gt_df = pd.read_csv(GROUND_TRUTH_FILE)
            print(f"  -> Associata Ground Truth generica: '{GROUND_TRUTH_FILE}'")

        df_run = extract_features_from_raw(path, gt_df=run_gt_df, fixed_gt_bpm=FIXED_GROUND_TRUTH_BPM)
        all_dfs.append(df_run)

    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset.to_csv(OUTPUT_DATASET_FILE, index=False)
    print(f"--> Dataset ML salvato con successo in '{OUTPUT_DATASET_FILE}' ({len(dataset)} finestre totali).")

    # ==========================================
    # VALUTAZIONE METRICHE RISPETTO ALLA GROUND TRUTH
    # ==========================================
    valid_gt = dataset.dropna(subset=["bpm_ground_truth"])
    if len(valid_gt) > 0:
        err_scg = np.abs(valid_gt["bpm_scg"] - valid_gt["bpm_ground_truth"])
        err_pcg = np.abs(valid_gt["bpm_pcg"] - valid_gt["bpm_ground_truth"])
        err_fus = np.abs(valid_gt["bpm_baseline_fusion"] - valid_gt["bpm_ground_truth"])

        rmse_fus = np.sqrt(np.mean((valid_gt["bpm_baseline_fusion"] - valid_gt["bpm_ground_truth"])**2))

        print("==========================================")
        print(" CONFRONTO CON GROUND TRUTH (ERRORE MAE)")
        print("==========================================")
        print(f"MAE Solo Accelerometro (SCG): {err_scg.mean():.2f} BPM")
        print(f"MAE Solo Microfono (PCG):     {err_pcg.mean():.2f} BPM")
        print(f"MAE Sensor Fusion Baseline:   {err_fus.mean():.2f} BPM (RMSE: {rmse_fus:.2f} BPM)")
        print("==========================================")

        # Grafico di confronto sull'ultima acquisizione
        last_run = valid_gt[valid_gt["run_id"] == valid_gt["run_id"].iloc[-1]]
        err_last_run = np.abs(last_run["bpm_baseline_fusion"] - last_run["bpm_ground_truth"])

        fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        axs[0].plot(last_run["time_s"], last_run["bpm_ground_truth"], 'k--', linewidth=2, label="Ground Truth (Riferimento)")
        axs[0].plot(last_run["time_s"], last_run["bpm_baseline_fusion"], 'g-o', markersize=4, label="Sensor Fusion Baseline")
        axs[0].plot(last_run["time_s"], last_run["bpm_scg"], color='crimson', alpha=0.5, label="Solo SCG (IMU)")
        axs[0].plot(last_run["time_s"], last_run["bpm_pcg"], color='royalblue', alpha=0.5, label="Solo PCG (Mic)")
        axs[0].set_ylabel("Frequenza [BPM]")
        axs[0].set_title(f"Confronto Stime vs Ground Truth ({last_run['run_id'].iloc[0]})", fontweight="bold")
        axs[0].legend(loc="upper right")
        axs[0].grid(True, linestyle="--", alpha=0.5)

        axs[1].plot(last_run["time_s"], err_last_run,
                    color="darkorange", linewidth=1.8, label="Errore Assoluto Fusion [BPM]")
        axs[1].axhline(y=err_last_run.mean(), color="red", linestyle="--",
                       label=f"MAE Run: {err_last_run.mean():.2f} BPM (Globale: {err_fus.mean():.2f} BPM)")
        axs[1].set_xlabel("Tempo [s]")
        axs[1].set_ylabel("Errore [BPM]")
        axs[1].set_title("Andamento Errore Assoluto nel Tempo", fontweight="bold")
        axs[1].legend(loc="upper right")
        axs[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()