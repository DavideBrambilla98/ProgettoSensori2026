#include "Nano33BLEAccelerometer.h"
#include <Wire.h>
#include <PDM.h>

#define LSM9DS1_ADDRESS_ACC   0x6B
#define LSM9DS1_CTRL_REG6_XL  0x20

// Buffer e variabili per il Microfono PDM
#define MIC_BUFFER_SIZE 256
volatile int16_t sampleBuffer[MIC_BUFFER_SIZE];
volatile int samplesRead = 0;

unsigned long current_time = 0;
unsigned long last_time = 0;
Nano33BLEAccelerometerData acc_data;

// Callback chiamata automaticamente quando il buffer PDM è pieno
void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read((void*)sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2; // Converti byte in int16 (2 byte per campione)
}

void setup() {
  // Inizializza seriale ad alta velocità
  Serial.begin(115200);

  // Attesa di sicurezza per la Seriale (evita blocchi sulla porta USB)
  unsigned long startWait = millis();
  while (!Serial && (millis() - startWait < 3000));

  // 1. Inizializza l'accelerometro
  Accelerometer.begin();

  // Configura a ±2g e 119 Hz (valore 0x60) su bus interno Wire1
  Wire1.beginTransmission(LSM9DS1_ADDRESS_ACC);
  Wire1.write(LSM9DS1_CTRL_REG6_XL);
  Wire1.write(0x60); // 119 Hz, ±2g
  Wire1.endTransmission();

  // 2. Inizializza il Microfono PDM
  PDM.onReceive(onPDMdata);
  PDM.setGain(30); // Aumenta il guadagno per intercettare i toni cardiaci
  if (!PDM.begin(1, 16000)) { // 1 canale (mono), 16 kHz di campionamento
    Serial.println("ERRORE: Inizializzazione Microfono PDM fallita!");
    while (1);
  }
}

void loop() {
  current_time = millis();

  // Esegue il controllo ogni 10 ms (~100 Hz)
  if (current_time - last_time >= 10) {
    last_time = current_time;

    // Estrae l'ultimo dato acustico disponibile dal buffer PDM
    int16_t mic_sample = 0;
    if (samplesRead > 0) {
      mic_sample = sampleBuffer[samplesRead - 1];
      samplesRead = 0; // Reset contatore
    }

    // Legge il dato solo se presente nel buffer dell'accelerometro
    if (Accelerometer.pop(acc_data)) {
      
      // Output CSV: timestamp_ms, ax, ay, az, mic
      Serial.print(current_time);
      Serial.print(",");
      Serial.print(acc_data.x / 2.0, 4);
      Serial.print(",");
      Serial.print(acc_data.y / 2.0, 4);
      Serial.print(",");
      Serial.print(acc_data.z / 2.0, 4);
      Serial.print(",");
      Serial.println(mic_sample);

    }
  }
}