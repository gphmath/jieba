from math import log
N=6
DAG = {0: [0, 2], 1: [1, 2], 2: [2], 3: [3], 4: [4, 5], 5: [5]}
logtotal=60101967
sentence = '老百姓大药房'
FREQ = {
    '老': 33423,
    '老百姓': 2994,
    '百': 3336,
    '百姓': 4176,
    '姓': 7240,
    '大': 144099,
    '药': 8404,
    '药房': 260,
    '房': 6407,
}
route = {}
route[N] = (0, 0)
for idx in range(N - 1, -1, -1):
    # print('------')
    print(idx)
    # print(DAG[idx])
    # for x in DAG[idx]:
    #     print(x)


    l=list((log(FREQ.get(sentence[idx:x + 1]) or 1)   # F(i:x+1)，从i到x组成词语的频数，如果不成词，就取1
                    -
                    logtotal                                       # 除以F总，字典中所有词的频数总和
                    +route[x + 1][0], x) for x in DAG[idx])
    print(l)
    route[idx] = max(l)
    # print(route)
print(route)