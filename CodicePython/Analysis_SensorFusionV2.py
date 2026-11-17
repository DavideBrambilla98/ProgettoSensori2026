import os
import time
import joblib
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, hilbert

# Importa la stessa identica pipeline DSP/feature del training!
from extract_ml_features import (
    extract_features_from_raw,
    apply_bandpass,
    FS_UNIFORM,
    LOW_CUT_HZ,
    HIGH_CUT_HZ,
    PCG_LOW_HZ,
    PCG_HIGH_HZ,
    N_FFT,
    SNR_THRESHOLD
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# CONFIGURAZIONE
# ==========================================
# Imposta il nome di un file CSV (es. "raw_data_20261116_210830.csv") per test offline,
# oppure lascia None per acquisire in diretta dalla scheda seriale.
INPUT_CSV_FILE = "raw_data_20261116_210830.csv"

SERIAL_PORT = None         # None = rilevamento automatico porta USB/Arduino
BAUD_RATE = 115200
RECORD_TIME_SEC = 60
MODEL_FILE = "hr_sensor_fusion_model.joblib"

MIN_BPM, MAX_BPM = 45.0, 180.0
DEFAULT_MAX_BPM_RATE_PER_SEC = 12.0


def apply_physiological_rate_limiter(bpm_series, time_s, max_rate_bpm_s=DEFAULT_MAX_BPM_RATE_PER_SEC):
    """
    Vincolo fisiologico anti-salto: limita la variazione massima istantanea
    tra finestre consecutive per eliminare picchi spuri da urti/artefatti.
    """
    filtered = np.array(bpm_series, dtype=float).copy()
    if len(filtered) < 2:
        return filtered

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
# ACQUISIZIONE DATI DA SERIALE
# ==========================================
def find_serial_port(preferred=None):
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("Nessuna porta seriale rilevata. Verifica il cavo USB della scheda.")
    if preferred:
        for p in ports:
            if p.device.upper() == preferred.upper():
                return p.device
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino", "nano", "usb", "ch340", "cp210", "cdc"]):
            print(f"--> Porta rilevata automaticamente: {p.device} ({p.description})")
            return p.device
    print(f"--> Uso prima porta disponibile: {ports[0].device} ({ports[0].description})")
    return ports[0].device


def acquire_serial_data(port, baudrate, duration):
    print(f"--> Connessione alla porta {port} ({baudrate} baud)...")
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    timestamps, ax_list, ay_list, az_list, mic_list = [], [], [], [], []
    print(f"--> Avvio acquisizione ({duration}s). Rimanere immobili sul petto...")
    start_time = time.time()
    last_print = start_time

    while (time.time() - start_time) < duration:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            if len(parts) == 5:
                t, ax, ay, az, mic = map(float, parts)
                timestamps.append(t)
                ax_list.append(ax)
                ay_list.append(ay)
                az_list.append(az)
                mic_list.append(mic)

            if time.time() - last_print >= 10:
                elapsed = int(time.time() - start_time)
                print(f"    Progresso: {elapsed}/{duration} s - Campioni: {len(timestamps)}")
                last_print = time.time()
        except ValueError:
            continue

    ser.close()
    print("--> Acquisizione terminata.\n")
    return (np.array(timestamps), np.array(ax_list),
            np.array(ay_list), np.array(az_list), np.array(mic_list))


def save_raw(t_raw, ax, ay, az, mic, run_id):
    fname = f"raw_data_{run_id}.csv"
    np.savetxt(fname, np.column_stack((t_raw, ax, ay, az, mic)), delimiter=",",
               header="timestamp_ms,ax,ay,az,mic_rms", comments="",
               fmt=["%.0f", "%.5f", "%.5f", "%.5f", "%.3f"])
    print(f"--> Dati grezzi salvati in: {fname}")
    return fname


def check_acquisition(csv_path):
    df = pd.read_csv(csv_path)
    t_raw = df["timestamp_ms"].values
    ax, ay, az = df["ax"].values, df["ay"].values, df["az"].values
    duration_s = (t_raw[-1] - t_raw[0]) / 1000.0
    fs_est = (len(t_raw) - 1) / duration_s
    g_mean = np.mean(np.sqrt(ax**2 + ay**2 + az**2))
    print("--- CONTROLLI ACQUISIZIONE ---")
    print(f"Campioni: {len(t_raw)} | Frequenza effettiva: {fs_est:.1f} Hz (attesa ~119 Hz)")
    print(f"Modulo medio accelerazione: {g_mean:.3f} g (atteso ~1.0 g)")
    if fs_est < 100:
        print("ATTENZIONE: frequenza effettiva bassa.")
    if abs(g_mean - 1.0) > 0.2:
        print("ATTENZIONE: modulo medio lontano da 1 g.")
    print()


def compute_diagnostic_signals(csv_path):
    """Ricostruisce segnali temporali e spettri medi FFT per le 3 finestre diagnostiche."""
    df_raw = pd.read_csv(csv_path)
    t_raw = df_raw["timestamp_ms"].values
    ax_raw, ay_raw, az_raw = df_raw["ax"].values, df_raw["ay"].values, df_raw["az"].values
    mic_raw = df_raw["mic_rms"].values

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
    mic = np.interp(t_sec, t_sec_raw, mic_raw)

    # Elaborazione SCG
    a_mag_raw = np.sqrt(ax**2 + ay**2 + az**2)
    a_mag = a_mag_raw - np.mean(a_mag_raw)
    scg_impulses = apply_bandpass(a_mag, lowcut=5.0, highcut=25.0, fs=FS_UNIFORM, order=4)
    scg_env = np.abs(hilbert(scg_impulses))
    scg_filtered = apply_bandpass(scg_env, lowcut=LOW_CUT_HZ, highcut=HIGH_CUT_HZ, fs=FS_UNIFORM, order=4)
    scg_filtered -= np.mean(scg_filtered)

    # Elaborazione PCG
    pcg_filtered = apply_bandpass(mic, lowcut=PCG_LOW_HZ, highcut=PCG_HIGH_HZ, fs=FS_UNIFORM, order=4)
    pcg_filtered -= np.mean(pcg_filtered)
    pcg_envelope_raw = np.abs(hilbert(pcg_filtered))

    # Spettro medio FFT su tutto il segnale
    window = np.hanning(len(t_sec))
    fft_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / FS_UNIFORM)
    bpm_axis = fft_freqs * 60.0
    mask = (bpm_axis >= MIN_BPM) & (bpm_axis <= MAX_BPM)

    fft_scg = np.abs(np.fft.rfft(scg_filtered * window, n=N_FFT))
    fft_pcg_plain = np.abs(np.fft.rfft(pcg_filtered * window, n=N_FFT))

    fft_pcg_comp = fft_pcg_plain[::2]
    n_hps = len(fft_pcg_comp)
    fft_pcg_hps = fft_pcg_plain.copy()
    fft_pcg_hps[:n_hps] = fft_pcg_plain[:n_hps] * fft_pcg_comp

    scg_norm = np.zeros_like(fft_scg)
    pcg_norm = np.zeros_like(fft_pcg_hps)
    pcg_plain_norm = np.zeros_like(fft_pcg_plain)

    if np.max(fft_scg[mask]) > 0:
        scg_norm[mask] = fft_scg[mask] / np.max(fft_scg[mask])
    if np.max(fft_pcg_hps[mask]) > 0:
        pcg_norm[mask] = fft_pcg_hps[mask] / np.max(fft_pcg_hps[mask])
    if np.max(fft_pcg_plain[mask]) > 0:
        pcg_plain_norm[mask] = fft_pcg_plain[mask] / np.max(fft_pcg_plain[mask])

    snr_scg = np.max(scg_norm[mask]) / (np.mean(scg_norm[mask]) + 1e-6)
    snr_pcg = np.max(pcg_plain_norm[mask]) / (np.mean(pcg_plain_norm[mask]) + 1e-6)

    if snr_scg >= SNR_THRESHOLD and snr_pcg >= SNR_THRESHOLD:
        fft_fused = scg_norm * pcg_norm
    elif snr_pcg > snr_scg:
        fft_fused = 0.8 * pcg_norm + 0.2 * scg_norm
    else:
        fft_fused = 0.8 * scg_norm + 0.2 * pcg_norm

    return (t_sec, a_mag_raw, scg_filtered, mic, pcg_filtered,
            pcg_envelope_raw, bpm_axis, scg_norm, pcg_norm, fft_fused)


# ==========================================
# MAIN EXECUTION & SENSOR FUSION ML
# ==========================================
if __name__ == "__main__":
    # 0. Caricamento modello ML
    ml_bundle = None
    if os.path.exists(MODEL_FILE):
        ml_bundle = joblib.load(MODEL_FILE)
        print(f"--> Modello ML caricato: '{ml_bundle['model_name']}' ({len(ml_bundle['feature_cols'])} feature)")
    else:
        print(f"ATTENZIONE: '{MODEL_FILE}' non trovato. Uso solo Baseline Deterministica.")

    # 1. Acquisizione da seriale oppure file CSV offline
    if INPUT_CSV_FILE:
        csv_path = INPUT_CSV_FILE
        run_id = os.path.splitext(os.path.basename(csv_path))[0].replace("raw_data_", "")
        print(f"--> Modalita' offline: analisi di '{csv_path}'")
    else:
        port = find_serial_port(SERIAL_PORT)
        t_raw, ax, ay, az, mic = acquire_serial_data(port, BAUD_RATE, RECORD_TIME_SEC)
        if len(t_raw) < 100:
            raise RuntimeError("Errore: Dati insufficienti. Verificare la connessione seriale.")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = save_raw(t_raw, ax, ay, az, mic, run_id)

    check_acquisition(csv_path)

    # Cerca eventuale Ground Truth associata per confronto grafico (se presente)
    gt_path = csv_path.replace("raw_data_", "gt_")
    gt_df = pd.read_csv(gt_path) if os.path.exists(gt_path) else None

    # 2. Estrazione identica delle feature tramite extract_ml_features.py
    df_feat = extract_features_from_raw(csv_path, gt_df=gt_df)
    df_feat["snr_diff"] = df_feat["snr_scg"] - df_feat["snr_pcg"]
    df_feat["ent_diff"] = df_feat["spec_ent_pcg"] - df_feat["spec_ent_scg"]
    df_feat["bpm_ratio"] = df_feat["bpm_pcg"] / (df_feat["bpm_scg"] + 1e-5)

    # 3. Predizione ML (Hard Switch + Vincolo Fisiologico Anti-Salto)
    if ml_bundle is not None:
        X = df_feat[ml_bundle["feature_cols"]].values
        prob_scg = ml_bundle["model"].predict_proba(X)[:, 1]
        agree_mask = df_feat["bpm_diff_sensors"].values < 3.0
        use_scg = prob_scg >= 0.5
        bpm_ml_raw = np.where(
            agree_mask,
            0.5 * (df_feat["bpm_scg"].values + df_feat["bpm_pcg"].values),
            np.where(use_scg, df_feat["bpm_scg"].values, df_feat["bpm_pcg"].values)
        )
        max_rate = ml_bundle.get("max_bpm_rate_per_sec", DEFAULT_MAX_BPM_RATE_PER_SEC)
        bpm_ml = apply_physiological_rate_limiter(bpm_ml_raw, df_feat["time_s"].values, max_rate)
    else:
        prob_scg = np.full(len(df_feat), 0.5)
        bpm_ml = df_feat["bpm_baseline_fusion"].values

    df_feat["prob_scg"] = np.round(prob_scg, 3)
    df_feat["bpm_ml_estimated"] = np.round(bpm_ml, 2)

    median_ml = np.median(bpm_ml)
    mean_ml = np.mean(bpm_ml)
    median_base = np.median(df_feat["bpm_baseline_fusion"])

    print(f"--- RISULTATI SENSOR FUSION ML ({len(df_feat)} finestre) ---")
    print("==========================================================")
    print(f" ML HARD SWITCH -> Mediana: {median_ml:.1f} BPM | Media: {mean_ml:.1f} BPM")
    print(f" BASELINE       -> Mediana: {median_base:.1f} BPM")
    if gt_df is not None:
        mae_ml = np.mean(np.abs(df_feat["bpm_ml_estimated"] - df_feat["bpm_ground_truth"]))
        mae_base = np.mean(np.abs(df_feat["bpm_baseline_fusion"] - df_feat["bpm_ground_truth"]))
        print(f" MAE vs GT      -> ML: {mae_ml:.2f} BPM | Baseline: {mae_base:.2f} BPM")
    print("==========================================================\n")

    # 4. Salvataggio risultati in sottocartella dedicata
    os.makedirs("risultati_output", exist_ok=True)
    res_filename = os.path.join("risultati_output", f"fusion_bpm_{run_id}.csv")
    df_feat[["time_s", "bpm_scg", "bpm_pcg", "bpm_baseline_fusion", "bpm_ml_estimated", "prob_scg"]].to_csv(
        res_filename, index=False
    )

    # Calcolo segnali e spettri medi per le finestre diagnostiche 1, 2 e 3
    (t_sec, a_mag_raw, scg_filtered, mic_interp, pcg_filtered,
     pcg_envelope_raw, bpm_axis, scg_norm, pcg_norm, fft_fused) = compute_diagnostic_signals(csv_path)

    # ==========================================
    # FINESTRA 1: ACCELEROMETRO (SCG)
    # ==========================================
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 9))
    fig1.suptitle("Elaborazione Accelerometro (SCG)", fontsize=14, fontweight='bold')

    axs1[0].plot(t_sec, a_mag_raw, color='gray', linewidth=0.8, label='Modulo Grezzo (sqrt(ax² + ay² + az²))')
    axs1[0].set_title("1. Segnale Grezzo (include gravità ~1g, respiro e rumore)", fontweight='bold')
    axs1[0].set_ylabel("Acc. [g]")
    axs1[0].grid(True, linestyle='--', alpha=0.5)
    axs1[0].legend(loc="upper right")

    axs1[1].plot(t_sec, scg_filtered, color='crimson', linewidth=1.0, label=f'SCG Filtrato ({LOW_CUT_HZ:.2f} - {HIGH_CUT_HZ:.1f} Hz)')
    axs1[1].set_title("2. Dopo Filtraggio Passa-Banda (rimossi offset DC, respiro e vibrazioni alte)", fontweight='bold')
    axs1[1].set_ylabel("Acc. [g]")
    axs1[1].set_xlabel("Tempo [s]")
    axs1[1].grid(True, linestyle='--', alpha=0.5)
    axs1[1].legend(loc="upper right")

    axs1[2].plot(bpm_axis, scg_norm, color='firebrick', linewidth=1.2, label='Spettro FFT Normalizzato (SCG)')
    axs1[2].set_xlim(30, 200)
    axs1[2].set_title("3. Trasformata di Fourier (FFT) - Accelerometro", fontweight='bold')
    axs1[2].set_ylabel("Ampiezza Norm.")
    axs1[2].set_xlabel("Frequenza Cardiaca [BPM]")
    axs1[2].grid(True, linestyle='--', alpha=0.5)
    axs1[2].legend(loc="upper right")

    fig1.tight_layout()
    fig1.savefig(os.path.join("risultati_output", f"plot_1_scg_{run_id}.png"), dpi=150)

    # ==========================================
    # FINESTRA 2: MICROFONO PDM (PCG)
    # ==========================================
    fig2, axs2 = plt.subplots(4, 1, figsize=(12, 11))
    fig2.suptitle("Elaborazione Microfono PDM (PCG)", fontsize=14, fontweight='bold')

    axs2[0].plot(t_sec, mic_interp, color='gray', linewidth=0.8, label='Segnale Inviluppo RMS Grezzo (PDM)')
    axs2[0].set_title("1. Segnale Grezzo Acquisito dal Microfono (con offset DC e rumore ambientale)", fontweight='bold')
    axs2[0].set_ylabel("Ampiezza")
    axs2[0].grid(True, linestyle='--', alpha=0.5)
    axs2[0].legend(loc="upper right")

    axs2[1].plot(t_sec, pcg_filtered, color='royalblue', linewidth=0.8, label=f'PCG Filtrato ({PCG_LOW_HZ:.1f} - {PCG_HIGH_HZ:.1f} Hz)')
    axs2[1].set_title("2. Dopo Filtraggio Passa-Banda (isolati i toni cardiaci S1 e S2)", fontweight='bold')
    axs2[1].set_ylabel("Ampiezza")
    axs2[1].grid(True, linestyle='--', alpha=0.5)
    axs2[1].legend(loc="upper right")

    axs2[2].plot(t_sec, pcg_filtered, color='royalblue', alpha=0.35, linewidth=0.8, label='PCG Filtrato')
    axs2[2].plot(t_sec, pcg_envelope_raw, color='darkorange', linewidth=1.4, label='Inviluppo di Hilbert')
    axs2[2].set_title("3. Estrazione Inviluppo di Hilbert (oscillazioni trasformate in picchi lenti)", fontweight='bold')
    axs2[2].set_ylabel("Ampiezza")
    axs2[2].set_xlabel("Tempo [s]")
    axs2[2].grid(True, linestyle='--', alpha=0.5)
    axs2[2].legend(loc="upper right")

    axs2[3].plot(bpm_axis, pcg_norm, color='royalblue', linewidth=1.2, label='Spettro FFT Normalizzato + HPS (PCG)')
    axs2[3].set_xlim(30, 200)
    axs2[3].set_title("4. Trasformata di Fourier (FFT) - Inviluppo Microfono", fontweight='bold')
    axs2[3].set_ylabel("Ampiezza Norm.")
    axs2[3].set_xlabel("Frequenza Cardiaca [BPM]")
    axs2[3].grid(True, linestyle='--', alpha=0.5)
    axs2[3].legend(loc="upper right")

    fig2.tight_layout()
    fig2.savefig(os.path.join("risultati_output", f"plot_2_pcg_{run_id}.png"), dpi=150)

    # ==========================================
    # FINESTRA 3: SENSOR FUSION SPETTRALE
    # ==========================================
    fig3, axs3 = plt.subplots(2, 1, figsize=(12, 7))
    fig3.suptitle("Confronto Spettrale e Sensor Fusion", fontsize=14, fontweight='bold')

    axs3[0].plot(bpm_axis, scg_norm, color='firebrick', alpha=0.75, linewidth=1.2, label='Spettro Accelerometro (SCG)')
    axs3[0].plot(bpm_axis, pcg_norm, color='royalblue', alpha=0.75, linewidth=1.2, label='Spettro Microfono (PCG)')
    axs3[0].set_xlim(30, 200)
    axs3[0].set_title("1. Sovrapposizione degli Spettri Indipendenti", fontweight='bold')
    axs3[0].set_ylabel("Ampiezza Norm.")
    axs3[0].grid(True, linestyle='--', alpha=0.5)
    axs3[0].legend(loc="upper right")

    axs3[1].plot(bpm_axis, fft_fused, color='black', linewidth=1.8, label='Spettro Fuso (SCG × PCG)')
    axs3[1].axvline(x=median_ml, color='green', linestyle='--', linewidth=2,
                    label=f'Battito Mediano ML: {median_ml:.1f} BPM')
    axs3[1].set_xlim(30, 200)
    axs3[1].set_title("2. Risultato Sensor Fusion (abbattimento picchi spuri non correlati)", fontweight='bold')
    axs3[1].set_xlabel("Frequenza Cardiaca [BPM]")
    axs3[1].set_ylabel("Ampiezza Fusa")
    axs3[1].grid(True, linestyle='--', alpha=0.5)
    axs3[1].legend(loc="upper right")

    fig3.tight_layout()
    fig3.savefig(os.path.join("risultati_output", f"plot_3_spectral_fusion_{run_id}.png"), dpi=150)

    # ==========================================
    # FINESTRA 4: EVOLUZIONE TEMPORALE E DECISIONE ML
    # ==========================================
    fig4, axs4 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig4.suptitle(f"Analisi Temporale Sensor Fusion ML ({run_id})", fontsize=14, fontweight='bold')

    if gt_df is not None:
        axs4[0].plot(df_feat["time_s"], df_feat["bpm_ground_truth"], 'k--', linewidth=2, label='Ground Truth')
    axs4[0].plot(df_feat["time_s"], df_feat["bpm_scg"], color='crimson', alpha=0.45, linestyle='--', label='Solo SCG (IMU)')
    axs4[0].plot(df_feat["time_s"], df_feat["bpm_pcg"], color='royalblue', alpha=0.45, linestyle='--', label='Solo PCG (Mic)')
    axs4[0].plot(df_feat["time_s"], df_feat["bpm_baseline_fusion"], color='orange', alpha=0.85, linewidth=1.3, label='Baseline Deterministica')
    axs4[0].plot(df_feat["time_s"], df_feat["bpm_ml_estimated"], marker='o', markersize=4, color='darkgreen', linewidth=2.0, label='ML Hard Switch (Finale)')
    axs4[0].axhline(y=median_ml, color='black', linestyle=':', linewidth=1.4, label=f'Mediana ML: {median_ml:.1f} BPM')
    axs4[0].set_ylim(MIN_BPM, MAX_BPM)
    axs4[0].set_title("1. Frequenza Cardiaca Stimata nel Tempo", fontweight='bold')
    axs4[0].set_ylabel("Frequenza [BPM]")
    axs4[0].legend(loc="upper right")
    axs4[0].grid(True, linestyle='--', alpha=0.5)

    axs4[1].plot(df_feat["time_s"], df_feat["prob_scg"], color='purple', linewidth=1.8, label='Probabilità Affidabilità SCG (IMU)')
    axs4[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.2, label='Soglia Switch (0.5: sopra=SCG, sotto=PCG)')
    axs4[1].set_ylim(-0.05, 1.05)
    axs4[1].set_title("2. Decisione del Classificatore ML nel Tempo", fontweight='bold')
    axs4[1].set_ylabel("Probabilità SCG")
    axs4[1].legend(loc="upper right")
    axs4[1].grid(True, linestyle='--', alpha=0.5)

    axs4[2].plot(df_feat["time_s"], df_feat["bpm_ratio"], color='teal', linewidth=1.6, label='bpm_ratio (PCG / SCG)')
    axs4[2].axhline(y=1.0, color='gray', linestyle='--', linewidth=1.0, label='Accordo Sensori (1.0)')
    axs4[2].axhline(y=2.0, color='red', linestyle=':', linewidth=1.2, label='Raddoppio Armonico PCG (2.0)')
    axs4[2].set_title("3. Rapporto Armonico tra Sensori (Feature Principale)", fontweight='bold')
    axs4[2].set_xlabel("Tempo [s]")
    axs4[2].set_ylabel("Rapporto BPM")
    axs4[2].legend(loc="upper right")
    axs4[2].grid(True, linestyle='--', alpha=0.5)

    fig4.tight_layout()
    plot_filename = os.path.join("risultati_output", f"plot_4_ml_timeline_{run_id}.png")
    fig4.savefig(plot_filename, dpi=150)
    print(f"--> Tutte le 4 finestre grafiche salvate in 'risultati_output/' (es. {plot_filename})")

    plt.show()