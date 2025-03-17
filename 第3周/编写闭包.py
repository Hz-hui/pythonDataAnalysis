def get_my_counter():
    x = -1

    def inner():
        nonlocal x
        x += 1
        return x

    return inner


my_counter = get_my_counter()

print(my_counter())
print(my_counter())