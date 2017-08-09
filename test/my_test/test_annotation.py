def fun(x: int) -> list:
    y = str(x*2)  # type: str
    print(y)
    return '4'.split(y)

z = fun(323)
z.append('aaa')
print(z)
