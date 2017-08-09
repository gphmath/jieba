import jieba
import os
import jieba.posseg as pseg
import datetime
import re


def how_to_use():
    """待分词的字符串可以是 unicode 或 UTF-8 字符串、GBK 字符串。
    注意：不建议直接输入 GBK 字符串，可能无法预料地错误解码成 UTF-8

jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，
或者用jieba.lcut 以及 jieba.lcut_for_search 直接返回 list"""
    dict_path = 'user_dict/user_dict.txt'

    seg_list = jieba.cut("我换不行北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

    jieba.load_userdict(dict_path)

    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    print("Full Mode: " + "/ ".join(seg_list))  # 全模式

    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

    seg_list = jieba.lcut("他来到了网易杭研大厦")  # 默认是精确模式
    print(", ".join(seg_list))
    print(type(seg_list))
    print(seg_list)

    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    print(", ".join(seg_list))

    seg_list = jieba.cut("我换不行北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

    words = pseg.cut("我爱北京天安门")
    print(words)
    for word, flag in words:
        print('%s %s' % (word, flag))

    print('分词：默认模式')
    result = jieba.tokenize(u'永和服装饰品有限公司')
    for tk in result:
        print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))
    print('分词：搜索模式')
    result = jieba.tokenize(u'永和服装饰品有限公司', mode='search')
    for tk in result:
        print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]))

if __name__ == '__main__':
    dict_path = '../../user_dict/user_dict.txt'
    print(os.getcwd())
    jieba.load_userdict(dict_path)
    # how_to_use()
    text_list = [
        # '据记者尹同飞报道，是融创董事长李石与王石关于宝能系和深圳地铁集团正式签约的时间，但戏剧性一幕出现了，',
        # '第三方富力地产加入交易，并且一起去参观了港珠澳大桥和粤港澳大湾区。',
        # '今天我开车去了北一环路',
        '老百姓大药房的药很便宜，很适合老百姓去买。',
        # '万科董事长王石和经理郁亮表示，公司目前和碧桂园还有恒大一起合作建造深圳地铁，目前共有10公里，整个项目包括了两个海底隧道'
    ]
    for text in text_list:
        words = pseg.cut(text)
        seg_list = []
        for word, flag in words:
            # print('%s/%s' % (word, flag))
            seg_list.append('%s/%s' % (word, flag))
        print(', '.join(seg_list))
    print('----------------------------------')
    # for text in text_list:
    #     words = jieba.lcut(text)
    #     print(', '.join(words))
