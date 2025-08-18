const int triggerPin = 2;   // Arduino output to camera trigger

void setup() {
  Serial.begin(9600);
  pinMode(triggerPin, OUTPUT);

  digitalWrite(triggerPin, LOW);   
  Serial.println("Press 'q' to trigger camera...");
}

void loop() {
  if (Serial.available() > 0) {
    char input = Serial.read();
    if (input == 'q' || input == 'Q') {
      Serial.println("Triggering camera...");

      // Send trigger pulse
      digitalWrite(triggerPin, HIGH);
      delayMicroseconds(1);
      digitalWrite(triggerPin, LOW);

     
    }}}