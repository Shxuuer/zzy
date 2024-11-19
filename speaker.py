import RPi.GPIO as GPIO
import time
import math

BUZZER_PIN = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

AMPLITUDE = 500
BASE = 1000
DURATION = 10
STEPS = 1000

def speak():
    pwm = GPIO.PWM(BUZZER_PIN, 1)
    pwm.start(50)

    for i in range(STEPS):
        t = i / STEPS * DURATION
        frequency = BASE + AMPLITUDE * math.sin(60 * math.pi * t / DURATION)
        pwm.ChangeFrequency(max(1, int(frequency)))
        time.sleep(DURATION / STEPS)

    pwm.stop()
    GPIO.cleanup()
