# print(None or 233)
# tuple = (1,4,2,7)
# print(max(tuple))
# t=((x*2,x+3) for x in range(4))
# for i in t:
#     print(i)
# x_max = max(t)
# print(x_max)
f=open('test_dict.txt','r',encoding='utf-8')
for a,b in enumerate(f, 1):
    print(a,b)
f.close()