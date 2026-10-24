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
LOW_CUT_HZ = MIN_BPM / 60.0   # Hz
HIGH_CUT_HZ = MAX_BPM / 60.0  # Hz

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

    # 3. Filtraggio Accelerometro (SCG)
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    a_mag -= np.mean(a_mag)
    scg_filtered = apply_bandpass(a_mag, lowcut=LOW_CUT_HZ, highcut=HIGH_CUT_HZ, fs=fs, order=4)

    # 4. Filtraggio Microfono (PCG) e Inviluppo di Hilbert
    mic_clean = mic - np.mean(mic)
    high_cutoff_mic = min(40.0, (fs / 2) - 1.0)
    pcg_filtered = apply_bandpass(mic_clean, lowcut=10.0, highcut=high_cutoff_mic, fs=fs, order=3)
    pcg_envelope = np.abs(hilbert(pcg_filtered))
    pcg_envelope -= np.mean(pcg_envelope)

    # 5. Analisi Spettrale (FFT) con Finestra di Hann
    N = len(scg_filtered)
    window = np.hanning(N)
    
    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
    bpm_axis = fft_freqs * 60.0  # Conversione Hz -> BPM

    # Calcolo magnitudo FFT
    fft_scg = np.abs(np.fft.rfft(scg_filtered * window))
    fft_pcg = np.abs(np.fft.rfft(pcg_envelope * window))

    # Maschera per la banda cardiaca fisiologica (45 - 180 BPM)
    mask = (bpm_axis >= MIN_BPM) & (bpm_axis <= MAX_BPM)

    # Normalizzazione degli spettri (0 a 1) nella sola banda utile
    scg_norm = np.zeros_like(fft_scg)
    pcg_norm = np.zeros_like(fft_pcg)
    
    if np.max(fft_scg[mask]) > 0:
        scg_norm[mask] = fft_scg[mask] / np.max(fft_scg[mask])
    if np.max(fft_pcg[mask]) > 0:
        pcg_norm[mask] = fft_pcg[mask] / np.max(fft_pcg[mask])

    # 6. SENSOR FUSION (Moltiplicazione Spettrale)
    fft_fused = scg_norm * pcg_norm

    # Stima del picco dominante
    bpm_valid = bpm_axis[mask]
    fused_valid = fft_fused[mask]
    
    peak_idx = np.argmax(fused_valid)
    estimated_bpm = bpm_valid[peak_idx]

    print("==========================================")
    print(f" FREQUENZA CARDIACA STIMATA: {estimated_bpm:.1f} BPM")
    print("==========================================\n")

    # ==========================================
    # VISUALIZZAZIONE GRAFICA
    # ==========================================
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

    # 1. SCG nel Tempo
    axs[0].plot(t_sec, scg_filtered, color='crimson', label='SCG Filtrato (Accelerometro)')
    axs[0].set_title("Segnale Temporale: Sismocardiogramma (SCG)", fontweight='bold')
    axs[0].set_ylabel("Acc. [g]")
    axs[0].set_xlabel("Tempo [s]")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc="upper right")

    # 2. FFT IMU (Accelerometro)
    axs[1].plot(bpm_axis, scg_norm, color='firebrick', label='Spettro FFT IMU (SCG)')
    axs[1].set_xlim(30, 200)
    axs[1].set_title("1. Spettro FFT - Accelerometro (SCG)", fontweight='bold')
    axs[1].set_ylabel("Ampiezza Norm.")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc="upper right")

    # 3. FFT Microfono (PCG)
    axs[2].plot(bpm_axis, pcg_norm, color='royalblue', label='Spettro FFT Microfono (Inviluppo PCG)')
    axs[2].set_xlim(30, 200)
    axs[2].set_title("2. Spettro FFT - Microfono PDM (PCG)", fontweight='bold')
    axs[2].set_ylabel("Ampiezza Norm.")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc="upper right")

    # 4. FFT Complessiva (SENSOR FUSION)
    axs[3].plot(bpm_axis, fft_fused, color='black', linewidth=1.8, label='Spettro Fuso (IMU × Microfono)')
    axs[3].axvline(x=estimated_bpm, color='green', linestyle='--', linewidth=2,
                   label=f'Battito Rilevato: {estimated_bpm:.1f} BPM')
    axs[3].set_xlim(30, 200)
    axs[3].set_title("3. Spettro Complessivo (SENSOR FUSION)", fontweight='bold')
    axs[3].set_xlabel("Frequenza Cardiaca [BPM]")
    axs[3].set_ylabel("Ampiezza Fusa")
    axs[3].grid(True, linestyle='--', alpha=0.5)
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    plt.show()