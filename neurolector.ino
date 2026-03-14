const int ch1Pin = A0;
const int ch2Pin = A1;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int ch1 = analogRead(ch1Pin);
  int ch2 = analogRead(ch2Pin);

  Serial.print(ch1);
  Serial.print(",");
  Serial.println(ch2);

  delay(10);
}