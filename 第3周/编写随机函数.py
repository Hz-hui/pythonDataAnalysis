import random

def my_random (keys, weights):
    total = 0
    for weight in weights:
        total += weight
    a = random.randint(1, total)

    temp = 0
    for i in range(len(weights)):
        temp += weights[i]

        if temp >= a:
            return keys[i]
            break

        else:
            i += 1

for i in range(10):
    print(my_random(keys = ["a", "b", "c"], weights = [1, 2, 2]))