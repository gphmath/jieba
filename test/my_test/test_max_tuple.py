t=tuple((-x*2,x) for x in range(4))
for i in t:
    print(i)
x_max = max(t)
print('--------')
print(x_max)
print('-------------------------------')
print('--------------------------------')
s=tuple((x*2,-x) for x in range(4))
for i in s:
    print(i)
x_max = max(s)
print('--------')
print(x_max)