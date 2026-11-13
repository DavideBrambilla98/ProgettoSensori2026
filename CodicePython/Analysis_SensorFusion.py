import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# ==========================================
# CONFIGURAZIONE
# ==========================================
SERIAL_PORT = 'COM4'       # Imposta la porta COM corretta
BAUD_RATE = 115200         # Baud rate configurato su Arduino
RECORD_TIME_SEC = 60       # Durata acquisizione (60 secondi)

# Range di frequenza cardiaca fisiologica (in BPM e Hz)
MIN_BPM, MAX_BPM = 45.0, 180.0
LOW_CUT_HZ = MIN_BPM / 60.0   # 0.75 Hz
HIGH_CUT_HZ = MAX_BPM / 60.0  # 3.00 Hz

# ==========================================
# FUNZIONI DI FILTRAGGIO
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# ==========================================
# ACQUISIZIONE DATI DA SERIALE
# ==========================================
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

# ==========================================
# MAIN EXECUTION & ELABORAZIONE
# ==========================================
if __name__ == "__main__":
    # 1. Acquisizione Dati
    t_raw, ax, ay, az, mic = acquire_serial_data(SERIAL_PORT, BAUD_RATE, RECORD_TIME_SEC)

    if len(t_raw) < 100:
        raise RuntimeError("Errore: Dati insufficienti. Verificare la connessione seriale.")

    # 2. Frequenza di Campionamento Effettiva
    t_sec = (t_raw - t_raw[0]) / 1000.0
    fs = 1.0 / np.mean(np.diff(t_sec))
    print(f"Frequenza di campionamento reale (fs): {fs:.2f} Hz")

    # 3. Accelerometro (SCG)
    a_mag_raw = np.sqrt(ax**2 + ay**2 + az**2)          # Step 3a: Modulo grezzo (include 1g + respiro)
    a_mag_centered = a_mag_raw - np.mean(a_mag_raw)     # Step 3b: Rimozione componente continua (DC)
    scg_filtered = apply_bandpass(a_mag_centered, lowcut=LOW_CUT_HZ, highcut=HIGH_CUT_HZ, fs=fs, order=4)

    # 4. Microfono (PCG)
    mic_clean = mic - np.mean(mic)                      # Step 4a: Rimozione offset DC
    high_cutoff_mic = min(40.0, (fs / 2) - 1.0)
    pcg_filtered = apply_bandpass(mic_clean, lowcut=10.0, highcut=high_cutoff_mic, fs=fs, order=3)
    pcg_envelope_raw = np.abs(hilbert(pcg_filtered))    # Step 4b: Inviluppo di Hilbert (positivo)
    pcg_envelope = pcg_envelope_raw - np.mean(pcg_envelope_raw)  # Step 4c: Inviluppo centrato per FFT

    # 5. Analisi Spettrale (FFT) con Finestra di Hann
    N = len(scg_filtered)
    window = np.hanning(N)

    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
    bpm_axis = fft_freqs * 60.0  # Conversione Hz -> BPM

    fft_scg = np.abs(np.fft.rfft(scg_filtered * window))
    fft_pcg = np.abs(np.fft.rfft(pcg_envelope * window))

    # Maschera per la banda cardiaca fisiologica (45 - 180 BPM)
    mask = (bpm_axis >= MIN_BPM) & (bpm_axis <= MAX_BPM)

    scg_norm = np.zeros_like(fft_scg)
    pcg_norm = np.zeros_like(fft_pcg)

    if np.max(fft_scg[mask]) > 0:
        scg_norm[mask] = fft_scg[mask] / np.max(fft_scg[mask])
    if np.max(fft_pcg[mask]) > 0:
        pcg_norm[mask] = fft_pcg[mask] / np.max(fft_pcg[mask])

    # 6. SENSOR FUSION (Moltiplicazione Spettrale)
    fft_fused = scg_norm * pcg_norm

    bpm_valid = bpm_axis[mask]
    fused_valid = fft_fused[mask]
    peak_idx = np.argmax(fused_valid)
    estimated_bpm = bpm_valid[peak_idx]

    print("==========================================")
    print(f" FREQUENZA CARDIACA STIMATA: {estimated_bpm:.1f} BPM")
    print("==========================================\n")

    # ==========================================
    # FINESTRA 1: ACCELEROMETRO (SCG)
    # ==========================================
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 9))
    fig1.suptitle("Elaborazione Accelerometro (SCG)", fontsize=14, fontweight='bold')

    # Step 1: Grezzo
    axs1[0].plot(t_sec, a_mag_raw, color='gray', linewidth=0.8, label='Modulo Grezzo (sqrt(ax² + ay² + az²))')
    axs1[0].set_title("1. Segnale Grezzo (include gravità ~1g, respiro e rumore)", fontweight='bold')
    axs1[0].set_ylabel("Acc. [g]")
    axs1[0].grid(True, linestyle='--', alpha=0.5)
    axs1[0].legend(loc="upper right")

    # Step 2: Filtrato
    axs1[1].plot(t_sec, scg_filtered, color='crimson', linewidth=1.0, label=f'SCG Filtrato ({LOW_CUT_HZ:.2f} - {HIGH_CUT_HZ:.1f} Hz)')
    axs1[1].set_title("2. Dopo Filtraggio Passa-Banda (rimossi offset DC, respiro e vibrazioni alte)", fontweight='bold')
    axs1[1].set_ylabel("Acc. [g]")
    axs1[1].set_xlabel("Tempo [s]")
    axs1[1].grid(True, linestyle='--', alpha=0.5)
    axs1[1].legend(loc="upper right")

    # Step 3: FFT SCG
    axs1[2].plot(bpm_axis, scg_norm, color='firebrick', linewidth=1.2, label='Spettro FFT Normalizzato (SCG)')
    axs1[2].set_xlim(30, 200)
    axs1[2].set_title("3. Trasformata di Fourier (FFT) - Accelerometro", fontweight='bold')
    axs1[2].set_ylabel("Ampiezza Norm.")
    axs1[2].set_xlabel("Frequenza Cardiaca [BPM]")
    axs1[2].grid(True, linestyle='--', alpha=0.5)
    axs1[2].legend(loc="upper right")

    plt.tight_layout()

    # ==========================================
    # FINESTRA 2: MICROFONO PDM (PCG)
    # ==========================================
    fig2, axs2 = plt.subplots(4, 1, figsize=(12, 11))
    fig2.suptitle("Elaborazione Microfono PDM (PCG)", fontsize=14, fontweight='bold')

    # Step 1: Grezzo
    axs2[0].plot(t_sec, mic, color='gray', linewidth=0.8, label='Segnale Audio Grezzo (PDM)')
    axs2[0].set_title("1. Segnale Grezzo Acquisito dal Microfono (con offset DC e rumore ambientale)", fontweight='bold')
    axs2[0].set_ylabel("Ampiezza")
    axs2[0].grid(True, linestyle='--', alpha=0.5)
    axs2[0].legend(loc="upper right")

    # Step 2: Filtrato
    axs2[1].plot(t_sec, pcg_filtered, color='royalblue', linewidth=0.8, label=f'PCG Filtrato (10 - {high_cutoff_mic:.1f} Hz)')
    axs2[1].set_title("2. Dopo Filtraggio Passa-Banda (isolati i toni cardiaci oscillanti S1 e S2)", fontweight='bold')
    axs2[1].set_ylabel("Ampiezza")
    axs2[1].grid(True, linestyle='--', alpha=0.5)
    axs2[1].legend(loc="upper right")

    # Step 3: Inviluppo di Hilbert
    axs2[2].plot(t_sec, pcg_filtered, color='royalblue', alpha=0.35, linewidth=0.8, label='PCG Filtrato')
    axs2[2].plot(t_sec, pcg_envelope_raw, color='darkorange', linewidth=1.4, label='Inviluppo di Hilbert')
    axs2[2].set_title("3. Estrazione Inviluppo di Hilbert (oscillazioni trasformate in picchi lenti)", fontweight='bold')
    axs2[2].set_ylabel("Ampiezza")
    axs2[2].set_xlabel("Tempo [s]")
    axs2[2].grid(True, linestyle='--', alpha=0.5)
    axs2[2].legend(loc="upper right")

    # Step 4: FFT PCG
    axs2[3].plot(bpm_axis, pcg_norm, color='royalblue', linewidth=1.2, label='Spettro FFT Normalizzato (Inviluppo PCG)')
    axs2[3].set_xlim(30, 200)
    axs2[3].set_title("4. Trasformata di Fourier (FFT) - Inviluppo Microfono", fontweight='bold')
    axs2[3].set_ylabel("Ampiezza Norm.")
    axs2[3].set_xlabel("Frequenza Cardiaca [BPM]")
    axs2[3].grid(True, linestyle='--', alpha=0.5)
    axs2[3].legend(loc="upper right")

    plt.tight_layout()

    # ==========================================
    # FINESTRA 3: SENSOR FUSION
    # ==========================================
    fig3, axs3 = plt.subplots(2, 1, figsize=(12, 7))
    fig3.suptitle("Confronto Spettrale e Sensor Fusion", fontsize=14, fontweight='bold')

    # Confronto sovrapposto
    axs3[0].plot(bpm_axis, scg_norm, color='firebrick', alpha=0.75, linewidth=1.2, label='Spettro Accelerometro (SCG)')
    axs3[0].plot(bpm_axis, pcg_norm, color='royalblue', alpha=0.75, linewidth=1.2, label='Spettro Microfono (PCG)')
    axs3[0].set_xlim(30, 200)
    axs3[0].set_title("1. Sovrapposizione degli Spettri Indipendenti", fontweight='bold')
    axs3[0].set_ylabel("Ampiezza Norm.")
    axs3[0].grid(True, linestyle='--', alpha=0.5)
    axs3[0].legend(loc="upper right")

    # Risultato Fuso
    axs3[1].plot(bpm_axis, fft_fused, color='black', linewidth=1.8, label='Spettro Fuso (SCG × PCG)')
    axs3[1].axvline(x=estimated_bpm, color='green', linestyle='--', linewidth=2,
                    label=f'Battito Rilevato: {estimated_bpm:.1f} BPM')
    axs3[1].set_xlim(30, 200)
    axs3[1].set_title("2. Risultato Sensor Fusion (abbattimento picchi spuri non correlati)", fontweight='bold')
    axs3[1].set_xlabel("Frequenza Cardiaca [BPM]")
    axs3[1].set_ylabel("Ampiezza Fusa")
    axs3[1].grid(True, linestyle='--', alpha=0.5)
    axs3[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()