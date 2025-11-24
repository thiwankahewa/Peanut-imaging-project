// --- Pin setup -------------------------------------------------
const int DRIVER_RELAY = 5;   // Relay to DIM/driver-enable
const int LED1_RELAY   = 2;   // Relay to LED1 channel
const int LED2_RELAY   = 3;   // Relay to LED2 channel
const int LED3_RELAY   = 4;   // Relay to LED3 channel

const unsigned long ON_TIME_MS = 3000;  // LED on-time per channel

void setup() {
  // Set relay pins as outputs
  pinMode(DRIVER_RELAY, OUTPUT);
  pinMode(LED1_RELAY,   OUTPUT);
  pinMode(LED2_RELAY,   OUTPUT);
  pinMode(LED3_RELAY,   OUTPUT);

  // All OFF at start (HIGH for active-LOW relays)
  digitalWrite(DRIVER_RELAY, HIGH);
  digitalWrite(LED1_RELAY,   HIGH);
  digitalWrite(LED2_RELAY,   HIGH);
  digitalWrite(LED3_RELAY,   HIGH);

  Serial.begin(9600);
  Serial.println("LED-relay sequence ready");
}

void loop() {
  // Sequence through 3 LEDs
  for (int i = 1; i <= 3; i++) {
    activateLED(i);
  }

  // Optional pause before repeating
  delay(2000);
}

void activateLED(int ledNumber) {
  // 1. Turn ON driver (DIM relay)
  digitalWrite(DRIVER_RELAY, LOW);
  delay(100);  // small stabilization delay

  // 2. Turn ON selected LED relay
  switch (ledNumber) {
    case 1: digitalWrite(LED1_RELAY, LOW); break;
    case 2: digitalWrite(LED2_RELAY, LOW); break;
    case 3: digitalWrite(LED3_RELAY, LOW); break;
  }

  Serial.print("LED "); Serial.print(ledNumber); Serial.println(" ON");
  delay(ON_TIME_MS);  // keep LED ON for 1 s

  // 3. Turn OFF that LED relay
  switch (ledNumber) {
    case 1: digitalWrite(LED1_RELAY, HIGH); break;
    case 2: digitalWrite(LED2_RELAY, HIGH); break;
    case 3: digitalWrite(LED3_RELAY, HIGH); break;
  }

  // 4. Turn OFF driver relay
  digitalWrite(DRIVER_RELAY, HIGH);

  Serial.print("LED "); Serial.print(ledNumber); Serial.println(" OFF");
  delay(1000);  // short gap before next LED
}
