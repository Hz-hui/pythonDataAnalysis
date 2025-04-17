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
    if len(func == 14) and func[0:3] == '+91' and func[3:].isdigit():
        return func[3:6] + ' ' + func[6:10] + ' ' + func[10:14]
    elif len(func == 13) and func[0:2] == '91' and func[2:].isdigit():
        return func[2:5] + ' ' + func[5:9] + ' ' + func[9:13]
    elif len(func == 12) and func[0:1] == '0' and func[1:].isdigit():
        return func[1:4] + ' ' + func[4:8] + ' ' + func[8:12]
    else:
        print(f"无效号码:{func}")
        return None




@log_decorator
def sort_phone_number(numbers):
    results = []
    for number in numbers:
        results.append(validate_and_standardize(number))
    results.sort()
    return results
    


n = int(input())
phone_numbers = []
for _ in range(n):
    phone_numbers.append(input())

sort_phone_number(phone_numbers)