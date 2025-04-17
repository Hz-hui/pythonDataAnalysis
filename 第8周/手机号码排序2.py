import functools

def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('开始执行函数:' + func.__name__)
        ret = func(*args, **kwargs)
        print('函数' + func.__name__ + '执行结束')
        return ret
    return wrapper


def validate_and_standardize(func):
    @functools.wraps(func)
    def wrapper(numbers):
        results = []
        for number in numbers:
            if len(number) == 14 and number[0:3] == '+91' and number[3:].isdigit():
                results.append('+91' + ' ' + number[3:6] + ' ' + number[6:10] + ' ' + number[10:14])
            elif len(number) == 13 and number[0:2] == '91' and number[2:].isdigit():
                results.append('+91' + ' ' + number[2:5] + ' ' + number[5:9] + ' ' + number[9:13])
            elif len(number) == 12 and number[0:1] == '0' and number[1:].isdigit():
                results.append('+91' + ' ' + number[1:4] + ' ' + number[4:8] + ' ' + number[8:12])
            else:
                print(f"无效号码:{number}")
        return func(results)
    return wrapper

@log_decorator
@validate_and_standardize
def sort_phone_number(numbers):
    for number in sorted(numbers):
        print(number)

n = int(input())
phone_numbers = []
for _ in range(n):
    phone_numbers.append(input())

sort_phone_number(phone_numbers)