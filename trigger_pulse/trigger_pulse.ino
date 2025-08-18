const int triggerPin = 2;   // Arduino output to camera trigger
const int signalPin  = 3;   // Camera output (exposure done / frame ready)

unsigned long lastTime = 0;
int frameCount = 0;

void setup() {
  Serial.begin(9600);
  pinMode(triggerPin, OUTPUT);
  pinMode(signalPin, INPUT);

  digitalWrite(triggerPin, LOW);   
  Serial.println("Press 'q' to trigger camera...");
}

void loop() {
  if (Serial.available() > 0) {
    char input = Serial.read();
    if (input == 'q' || input == 'Q') {}
      Serial.println("Triggering camera...");

      // Send trigger pulse
      digitalWrite(triggerPin, HIGH);
      delayMicroseconds(1);
      digitalWrite(triggerPin, LOW);

      // Wait for 3 frames
      frameCount = 0;
      lastTime = 0;

      while (frameCount < 3) {
        // Wait for rising edge
        while (digitalRead(signalPin) == LOW);  // wait for HIGH
        digitalWrite(triggerPin, HIGH);
      delayMicroseconds(1);
      digitalWrite(triggerPin, LOW);
        unsigned long now = micros();

        frameCount++;

        if (lastTime == 0) {
          Serial.print("Frame ");
          Serial.print(frameCount);
          Serial.println(" received.");
        } else {
          unsigned long delta = now - lastTime;
          Serial.print("Frame ");
          Serial.print(frameCount);
          Serial.print(" received. Δt = ");
          Serial.print(delta);
          Serial.println(" µs");
        }

        lastTime = now;

        // Wait until signal goes LOW again (to avoid multiple counts)
        while (digitalRead(signalPin) == HIGH);
      }

      Serial.println("Captured 3 frames. Done.");
    }}