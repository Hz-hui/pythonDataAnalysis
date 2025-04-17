def d(f):
    def n(*args):
        return '$'+str(f(*args))
    return n
@d
def p(a, t):
    return a + a*t

print(p(100,0))


max = lambda a, b: a if (a>b) else b
print(max(1,2))

def foo(*args, **kwargs):
    print(f'args={args}, kwargs={kwargs}')

foo()
foo(1,2,3)
foo([1,2,3])
foo(one=1, two=2)