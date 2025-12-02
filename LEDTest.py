#!/usr/bin/env python3
import time
import platform
from gpiozero import Device

# -------- MOCK GPIO FOR WINDOWS / WSL --------
if platform.system() != "Linux":
    from gpiozero.pins.mock import MockFactory
    Device.pin_factory = MockFactory()
    print("[MOCK GPIO] Using simulated GPIO on Windows")

from gpiozero import OutputDevice

# -------- GPIO PINS (BCM) --------
DRIVER = 17
LED1   = 27
LED2   = 22
LED3   = 23

# -------- RELAYS --------
# active_low → relay turns ON when output is LOW
driver = OutputDevice(DRIVER, active_high=False, initial_value=False)
led1   = OutputDevice(LED1,   active_high=False, initial_value=False)
led2   = OutputDevice(LED2,   active_high=False, initial_value=False)
led3   = OutputDevice(LED3,   active_high=False, initial_value=False)

leds = [led1, led2, led3]

def all_off():
    """Turn OFF all LEDs."""
    for led in leds:
        led.off()

def turn_on_only(led, name, duration=1.0):
    """Turn ON only one LED at a time."""
    print(f"[SET] Turning ON {name} (others OFF)")
    all_off()
    led.on()          # active_low → relay energizes (ON)
    time.sleep(duration)
    led.off()
    print(f"[CLEAR] {name} OFF\n")
    time.sleep(0.5)

def main():
    print("== One-LED-at-a-Time Relay Test ==")

    print("\n[INIT] Turning ON driver relay...")
    driver.on()
    time.sleep(1)

    turn_on_only(led1, "LED1", 1)
    turn_on_only(led2, "LED2", 1)
    turn_on_only(led3, "LED3", 1)

    print("[STOP] Turning OFF driver relay...")
    driver.off()
    print("\nTest complete.")

if __name__ == "__main__":
    main()
