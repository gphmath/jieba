#encoding:UTF-8
def yield_fun(n):
    # 这个n只是一个参数，并不表示生成数组的长度或者任何特定意义
    # 生成器，只能用于for i in gen(arg)的场合
    # 正常执行，遇到yield，返回出去一个值，然后继续执行。完全可以不用for循环，不过一般用for循环才有意义嘛
    for i in range(n+10,n+20):
        print("yield前i=", i)
        yield call(i)
        print("后i=",i)
    # yield n+20
    # yield n+1000
    #做一些其它的事情
    #
    print("do something.")
    print("end.")

def normal_test(n):
    l = []
    for i in range(n):
        l.append(call(i))
        print("i=",i)
    print("do something.")
    print("end.")
    return l


def call(i):
    return i*2

#使用for循环
for y in yield_fun(5):
    print('main y', y, ",")
# z=yield_fun(5)
# print(z(0))
# print('--------------------------------------------')
# for z in normal_test(5):
#     print('main z',z,",")
