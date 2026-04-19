import stringGenerator as sg
import mmd

p1 = 0.9
p0 = 0.1

with open(f"mmd_test_good1.txt", 'w') as f:
    f.write("")

with open(f"mmd_test_good2.txt", 'w') as f:
    f.write("")

with open(f"mmd_test_bad.txt", 'w') as f:
    f.write("")

for i in range(100):
    with open(f"mmd_test_good1.txt", 'a') as f:
        f.write(str(sg.generate_string(p0, p1, 12)) + "\n")

for i in range(100):
    with open(f"mmd_test_good2.txt", 'a') as f:
        f.write(str(sg.generate_string(p0, p1, 12)) + "\n")

for i in range(100):
    with open(f"mmd_test_bad1.txt", 'a') as f:
        f.write(str(sg.generate_string(p0, p0, 12)) + "\n")

for i in range(100):
    with open(f"mmd_test_bad2.txt", 'a') as f:
        f.write(str(sg.generate_string(p1, p1, 12)) + "\n")

for i in range(100):
    with open(f"mmd_test_bad3.txt", 'a') as f:
        f.write(str(sg.generate_string(p1, p1, 12)) + "\n")

with open(f"mmd_test_good1.txt", 'r') as f:
    good1 = [list(map(int, line.strip()[1:-1].split(','))) for line in f.readlines()]

with open(f"mmd_test_good2.txt", 'r') as f:
    good2 = [list(map(int, line.strip()[1:-1].split(','))) for line in f.readlines()]

with open(f"mmd_test_bad1.txt", 'r') as f:
    bad1 = [list(map(int, line.strip()[1:-1].split(','))) for line in f.readlines()]

with open(f"mmd_test_bad2.txt", 'r') as f:
    bad2 = [list(map(int, line.strip()[1:-1].split(','))) for line in f.readlines()]

with open(f"mmd_test_bad3.txt", 'r') as f:
    bad3 = [list(map(int, line.strip()[1:-1].split(','))) for line in f.readlines()]

print(mmd.mmd(good1, good2))
print(mmd.mmd(good1, bad1))
print(mmd.mmd(good1, bad2))
print(mmd.mmd(good1, bad3))