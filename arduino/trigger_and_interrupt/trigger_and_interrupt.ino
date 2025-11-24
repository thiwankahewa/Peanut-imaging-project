const int triggerPin = 2;   // Arduino -> Camera
const int signalPin  = 3;   // Camera -> Arduino

const int redLed    = 4;
const int greenLed  = 5;
const int blueLed   = 6;

volatile unsigned long lastTime = 0;
volatile int frameCount = 0;

void setup() {
  Serial.begin(9600);

  pinMode(triggerPin, OUTPUT);
  pinMode(signalPin, INPUT);

  pinMode(redLed, OUTPUT);
  pinMode(greenLed, OUTPUT);
  pinMode(blueLed, OUTPUT);

  attachInterrupt(digitalPinToInterrupt(signalPin), onFrameSignal, RISING);   }

void loop() {
    if (Serial.available()) {
    char c = Serial.read();
    if (c == 'q') {
      sendTriggerPulse();
      Serial.println("Trigger pulse sent!");
    }
  }
}

void onFrameSignal() {
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

  // LED actions
  if (frameCount == 1) Serial.println("Red LED ON");
  if (frameCount == 3) Serial.println("Green LED ON");
  if (frameCount == 4) Serial.println("Blue LED ON");

  lastTime = now;
  sendTriggerPulse();
}

void sendTriggerPulse() {
      digitalWrite(triggerPin, HIGH);
      delayMicroseconds(1);  
      digitalWrite(triggerPin, LOW);
}
