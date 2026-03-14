const int ecgPin = A0;

// Si tu AD8232 usa leads-off:
const int loPlusPin = 10;
const int loMinusPin = 11;

const int CENTER = 512;

// Baseline lento para quitar deriva
float baseline = 512.0;
const float baselineAlpha = 0.005;

// Canal suavizado
float smoothCh2 = 512.0;
const float smoothAlpha = 0.10;

// Historial de actividad
float activityAvg = 0.0;
const float activityAlpha = 0.08;

// Ganancias base
const float idleGain = 0.10;     // reposo
const float activeGain = 0.35;   // actividad detectada

// Límite de excursión final
const int idleExcursion = 10;
const int activeExcursion = 35;

// Umbral para decidir "actividad"
const float activityThreshold = 6.0;

void setup() {
  Serial.begin(115200);
  pinMode(loPlusPin, INPUT);
  pinMode(loMinusPin, INPUT);
}

void loop() {
  // Si se despega un electrodo
  if (digitalRead(loPlusPin) == HIGH || digitalRead(loMinusPin) == HIGH) {
    Serial.println("512,512");
    delay(10);
    return;
  }

  int raw = analogRead(ecgPin);

  // 1) baseline lento
  baseline = baseline + baselineAlpha * (raw - baseline);

  // 2) señal centrada
  float centered = raw - baseline;

  // 3) actividad instantánea
  float instantActivity = abs(centered);

  // 4) actividad promedio
  activityAvg = activityAvg + activityAlpha * (instantActivity - activityAvg);

  // 5) decidir ganancia según actividad
  float gain;
  int maxExcursion;

  if (activityAvg > activityThreshold) {
    gain = activeGain;
    maxExcursion = activeExcursion;
  } else {
    gain = idleGain;
    maxExcursion = idleExcursion;
  }

  // 6) canal 1
  float ch1f = CENTER + centered * gain;

  // limitar excursión
  if (ch1f > CENTER + maxExcursion) ch1f = CENTER + maxExcursion;
  if (ch1f < CENTER - maxExcursion) ch1f = CENTER - maxExcursion;

  // 7) canal 2 suavizado
  smoothCh2 = smoothCh2 + smoothAlpha * (ch1f - smoothCh2);
  float ch2f = smoothCh2;

  int ch1 = constrain((int)ch1f, 0, 1023);
  int ch2 = constrain((int)ch2f, 0, 1023);

  // Formato que espera tu UI
  Serial.print(ch1);
  Serial.print(",");
  Serial.println(ch2);

  delay(10);  // ~100 Hz
}