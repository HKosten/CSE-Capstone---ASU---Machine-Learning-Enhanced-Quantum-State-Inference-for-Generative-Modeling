import numpy as np

def generate_string(probability0, probability1, length):
    str = [np.random.choice(range(0, 6)).item()]
    for i in range(1, length):
        if str[i - 1] <= 2:
            if(np.random.rand() < probability0):
                str.append(np.random.choice(range(3, 6)).item())
            else:
                str.append(np.random.choice(range(0, 3)).item())
        else:
            if(np.random.rand() < probability1):
                str.append(np.random.choice(range(3, 6)).item())
            else:
                str.append(np.random.choice(range(0, 3)).item())

    return str

with open("60-40,60-40.txt", "w") as f:
    f.write(str(generate_string(0.6, 0.6, 1000)))

with open("60-40,40-60.txt", "w") as f:
    f.write(str(generate_string(0.6, 0.4, 1000)))

with open("90-10,50-50.txt", "w") as f:
    f.write(str(generate_string(0.9, 0.5, 1000)))