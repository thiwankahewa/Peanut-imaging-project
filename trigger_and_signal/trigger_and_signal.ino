const int triggerPin = 2;   // Arduino output to camera trigger
const int signalPin  = 3;   // Camera output (exposure done / frame ready)

const int redLed    = 4;
const int greenLed  = 5;
const int blueLed   = 6;


unsigned long lastTime = 0;
int frameCount = 0;

void setup() {
  Serial.begin(9600);
  pinMode(triggerPin, OUTPUT);
  pinMode(signalPin, INPUT);

  digitalWrite(triggerPin, LOW);   
  digitalWrite(redLed, LOW);
  digitalWrite(greenLed, LOW);
  digitalWrite(blueLed, LOW);
  Serial.println("Press 'q' to trigger camera...");
}

void loop() {
  if (Serial.available() > 0) {
    char input = Serial.read();

    if (input == 'q' || input == 'Q') {
      Serial.println("Triggering camera...");
      
      digitalWrite(triggerPin, HIGH);
      delayMicroseconds(1);
      digitalWrite(triggerPin, LOW);     // Send trigger pulse

      frameCount = 0;
      lastTime = 0;

      while (frameCount < 4) {
        while (digitalRead(signalPin) == LOW);  // Wait for rising edge
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

        if (frameCount == 1) {Serial.println("Red LED ON");}
        if (frameCount == 3) {Serial.println("Green LED ON");}
        if (frameCount == 4) {Serial.println("Blue LED ON");}

        lastTime = now;

        while (digitalRead(signalPin) == HIGH);    // Wait until signal goes LOW again

        digitalWrite(triggerPin, HIGH);
        delayMicroseconds(1);
        digitalWrite(triggerPin, LOW);
      }

      Serial.println("Captured 4 frames. Done.");
    }}}