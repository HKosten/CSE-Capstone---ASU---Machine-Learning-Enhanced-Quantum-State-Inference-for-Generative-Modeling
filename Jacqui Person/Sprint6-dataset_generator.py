import csv
import random
import math
from math import pi as pi
from math import sin as sin

seq_length = 128

#1000 bitstrings - each bitstring represents a sin wave shifted based on a random phase
#0s = when sin is negative
#1s = when sin is positive
with open("/Users/jacquiperson/Desktop/school/CSE486/bitstrings.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["bitstring"])

    for _ in range(1000):

        phase = random.uniform(0, 2 * math.pi)
        bitstring = ""

        for i in range(128):

            x = i / 128

            wave = sin(2 * pi * x + phase)

            if wave > 0:
                bitstring += "1"
            else:
                bitstring += "0"

        writer.writerow([bitstring])


