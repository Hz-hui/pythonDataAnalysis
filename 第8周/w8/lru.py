import functools
import urllib.request
import urllib.error

@functools.lru_cache(maxsize=32)
def get_pep(num):
    'Retrieve text of a Python Enhancement Proposal'
    resource = f'https://peps.python.org/pep-{num:04d}'
    try:
        with urllib.request.urlopen(resource) as s:
            return s.read()
    except urllib.error.HTTPError:
        return 'Not Found'

#如果 maxsize 设为 None，LRU 特性将被禁用且缓存可无限增长。
@functools.lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

#test get_pep
for n in 8, 290, 308, 320, 8, 218, 320, 279, 289, 320, 9991:
    pep = get_pep(n)
    print(n, len(pep))
print(get_pep.cache_info())
get_pep.cache_clear()#清空

#test fib
print([fib(n) for n in range(16)])
print(fib.cache_info())
