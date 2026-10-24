import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# ==========================================
# CONFIGURAZIONE
# ==========================================
SERIAL_PORT = 'COM4'       # Sostituire con la porta corretta (es. '/dev/ttyACM0' su Linux/macOS)
BAUD_RATE = 115200         # Deve coincidere con Serial.begin(115200) su Arduino
RECORD_TIME_SEC = 60       # Durata dell'acquisizione (60s = risoluzione FFT di 1 BPM)

# ==========================================
# FUNZIONI DI FILTRAGGIO
# ==========================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    """Crea i coefficienti del filtro passa-banda di Butterworth."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, order=4):
    """Applica il filtro passa-banda a fase zero (filtfilt evita distorsioni di fase)."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# ==========================================
# ACQUISIZIONE DATI DA SERIALE
# ==========================================
def acquire_serial_data(port, baudrate, duration):
    print(f"--> Apertura porta seriale {port} a {baudrate} baud...")
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)  # Attesa reset della scheda
    ser.reset_input_buffer()

    timestamps = []
    ax_list, ay_list, az_list = [], [], []
    mic_list = []

    print(f"--> Avvio acquisizione dati per {duration} secondi. Rimanere immobili e respirare normalmente...")
    start_time = time.time()
    last_print = start_time

    while (time.time() - start_time) < duration:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            
            # Verifica che la riga contenga i 5 campi: timestamp, ax, ay, az, mic
            if len(parts) == 5:
                t = float(parts[0])
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
                mic = float(parts[4])

                timestamps.append(t)
                ax_list.append(ax)
                ay_list.append(ay)
                az_list.append(az)
                mic_list.append(mic)

            # Indicatore di progresso ogni 5 secondi
            if time.time() - last_print >= 5:
                elapsed = int(time.time() - start_time)
                print(f"    Tempo trascorso: {elapsed}/{duration} s - Campioni letti: {len(timestamps)}")
                last_print = time.time()

        except ValueError:
            continue

    ser.close()
    print("--> Acquisizione completata con successo.\n")
    
    return (np.array(timestamps), 
            np.array(ax_list), 
            np.array(ay_list), 
            np.array(az_list), 
            np.array(mic_list))

# ==========================================
# ELABORAZIONE E ANALISI FFT
# ==========================================
if __name__ == "__main__":
    # 1. Acquisizione
    t_raw, ax, ay, az, mic = acquire_serial_data(SERIAL_PORT, BAUD_RATE, RECORD_TIME_SEC)

    if len(t_raw) < 100:
        raise RuntimeError("Errore: Numero di campioni insufficienti. Verificare la porta seriale.")

    # 2. Calcolo della Frequenza di Campionamento Reale (fs)
    t_sec = (t_raw - t_raw[0]) / 1000.0  # Converte ms in secondi
    dt = np.diff(t_sec)
    fs = 1.0 / np.mean(dt)
    print(f"--- PARAMETRI ACQUISIZIONE ---")
    print(f"Totale campioni: {len(t_sec)}")
    print(f"Frequenza di campionamento misurata (fs): {fs:.2f} Hz\n")

    # 3. Elaborazione Segnale IMU (Sismocardiogramma - SCG)
    # Calcolo del modulo dell'accelerazione per renderlo indipendente dall'orientamento
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    a_mag -= np.mean(a_mag)  # Rimozione componente continua DC (gravità)

    # Filtro passa-banda SCG: intervallo cardiaci fisiologici (0.8 Hz - 3.0 Hz => 48 - 180 BPM)
    scg_filtered = apply_bandpass(a_mag, lowcut=0.8, highcut=3.0, fs=fs, order=4)

    # 4. Elaborazione Segnale Microfono (Fonocardiogramma - PCG)
    mic_clean = mic - np.mean(mic)
    # Passa-banda per toni acustici cardiaci (limitato dalla frequenza di Nyquist fs/2)
    high_cutoff_mic = min(40.0, (fs / 2) - 1)
    pcg_filtered = apply_bandpass(mic_clean, lowcut=10.0, highcut=high_cutoff_mic, fs=fs, order=3)
    # Calcolo dell'inviluppo d'energia (Trasformata di Hilbert) per evidenziare i toni S1/S2
    pcg_envelope = np.abs(hilbert(pcg_filtered))

    # 5. Analisi Spettrale (FFT su segnale SCG)
    N = len(scg_filtered)
    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_magnitude = np.abs(np.fft.rfft(scg_filtered))

    # Conversione asse x in BPM
    bpm_axis = fft_freqs * 60.0

    # Ricerca picco nella banda d'interesse cardiaca (45 BPM - 180 BPM)
    valid_mask = (bpm_axis >= 45.0) & (bpm_axis <= 180.0)
    bpm_valid = bpm_axis[valid_mask]
    mag_valid = fft_magnitude[valid_mask]

    peak_index = np.argmax(mag_valid)
    estimated_bpm = bpm_valid[peak_index]

    print(f"==========================================")
    print(f" FREQUENZA CARDIACA STIMATA: {estimated_bpm:.1f} BPM")
    print(f"==========================================\n")

    # ==========================================
    # VISUALIZZAZIONE GRAFICA
    # ==========================================
    fig, axs = plt.subplots(3, 1, figsize=(12, 9))

    # Grafico 1: SCG nel tempo
    axs[0].plot(t_sec, a_mag, color='lightgray', label='Modulo Acc. Grezzo')
    axs[0].plot(t_sec, scg_filtered, color='crimson', linewidth=1.5, label='SCG Filtrato (0.8 - 3.0 Hz)')
    axs[0].set_title("Sismocardiogramma (SCG) - Accelerometro", fontsize=12, fontweight='bold')
    axs[0].set_xlabel("Tempo [s]")
    axs[0].set_ylabel("Accelerazione [g]")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Grafico 2: PCG (Microfono) nel tempo
    axs[1].plot(t_sec, pcg_filtered, color='royalblue', alpha=0.5, label='Audio Filtrato')
    axs[1].plot(t_sec, pcg_envelope, color='navy', linewidth=1.5, label='Inviluppo di Energia (Hilbert)')
    axs[1].set_title("Fonocardiogramma (PCG) - Microfono PDM", fontsize=12, fontweight='bold')
    axs[1].set_xlabel("Tempo [s]")
    axs[1].set_ylabel("Ampiezza [u.a.]")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Grafico 3: Spettro FFT e Stima BPM
    axs[2].plot(bpm_axis, fft_magnitude, color='black', label='Spettro di Frequenza (FFT)')
    axs[2].axvline(x=estimated_bpm, color='green', linestyle='--', linewidth=2, 
                   label=f'Battito Cardiaco: {estimated_bpm:.1f} BPM')
    axs[2].set_xlim(30, 200)
    axs[2].set_title("Analisi nel Dominio delle Frequenze (FFT)", fontsize=12, fontweight='bold')
    axs[2].set_xlabel("Frequenza Cardiaca [BPM]")
    axs[2].set_ylabel("Ampiezza FFT")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()