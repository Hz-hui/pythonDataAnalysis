def my_sum(*arg, value = 1):
    '''
    Add Value To Numbers
    '''
    lis = []
    for item in arg:
        lis.append(item)
    for i in range(len(lis)):
        lis[i] += value
    return lis

print(my_sum(1, 2, 3, 4, value = 10))
print(my_sum(1, 2, 3, 4))