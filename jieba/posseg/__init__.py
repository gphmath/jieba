from __future__ import absolute_import, unicode_literals
import os
import re
import sys
import jieba
import pickle
from .._compat import *
from .viterbi import viterbi

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"

re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
# 所有中文字

re_skip_detail = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
# 数值(可以有小数点) 或者 字母加数字

re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
# 所有中文字、字母、数字、小数点还有+#&_

re_skip_internal = re.compile("(\r\n|\s)")
# 换行符或者空格

re_eng = re.compile("[a-zA-Z0-9]+")
# 字母和数字

re_num = re.compile("[\.0-9]+")
# 数值，可以有小数点

re_eng1 = re.compile('^[a-zA-Z0-9]$', re.U)


def load_model():
    # For Jython
    start_p = pickle.load(get_module_res("posseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("posseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("posseg", PROB_EMIT_P))
    state = pickle.load(get_module_res("posseg", CHAR_STATE_TAB_P))
    return state, start_p, trans_p, emit_p


if sys.platform.startswith("java"):
    char_state_tab_P, start_P, trans_P, emit_P = load_model()
else:  # 当前Python是这个
    from .char_state_tab import P as char_state_tab_P
    # 字典，键是大部分中文单字，值是各种可能的(角色,词性)的组合——这就是jieba的HMM中的状态，注意状态不是{B,E,M,S}。
    # 比如“呛”这个字的值：
    # (('S', 'v'), ('E', 'v'), ('B', 'a'), ('B', 'v'))
    from .prob_start import P as start_P
    # 文本的初始状态概率（应该取了对数）：一个item形如：('B', 'a'): -4.76230，B的初始概率肯定更大些
    from .prob_trans import P as trans_P
    # 状态转移矩阵：键是每个状态，值是这个状态可能转移到的状态和转移概率.
    # ('B', 'a'): {('E', 'a'): -0.0050648453069648755, ('M', 'a'): -5.287963037107507},
    from .prob_emit import P as emit_P
    # 发射矩阵：键是每个状态，值是这个状态可能产生的所有字及其概率
    # ('E', 'p'): {'\u4e4b': -6.565515869430067, '\u4e86': -2.026203081578427}


class pair(object):
    """
    pair类，王石/nr
    词+词性，两个属性
    """
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __unicode__(self):
        return '%s/%s' % (self.word, self.flag)

    def __repr__(self):
        return 'pair(%r, %r)' % (self.word, self.flag)

    def __str__(self):
        if PY2:
            return self.__unicode__().encode(default_encoding)
        else:
            return self.__unicode__()

    def __iter__(self):
        return iter((self.word, self.flag))

    def __lt__(self, other):
        return self.word < other.word

    def __eq__(self, other):
        return isinstance(other, pair) and self.word == other.word and self.flag == other.flag

    def __hash__(self):
        return hash(self.word)

    def encode(self, arg):
        return self.__unicode__().encode(arg)


class POSTokenizer(object):

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or jieba.Tokenizer()
        self.load_word_tag(self.tokenizer.get_dict_file())
        # 传入参数是f=open(dict_file_path, 'rb')的操作句柄f


    def __repr__(self):
        return '<POSTokenizer tokenizer=%r>' % self.tokenizer

    def __getattr__(self, name):
        if name in ('cut_for_search', 'lcut_for_search', 'tokenize'):
            # may be possible?
            raise NotImplementedError
        return getattr(self.tokenizer, name)

    def initialize(self, dictionary=None):
        self.tokenizer.initialize(dictionary)
        self.load_word_tag(self.tokenizer.get_dict_file())
        # 传入参数是f=open(dict_file_path, 'rb')的操作句柄f

    def load_word_tag(self, f):
        """
        加载词典，提取其中词-词性，不要频数信息，返回一个字典，键是词，值是词性——这个内置词典，每个词只有一种词性。。
        :param f: 
        :return: 词-词性字典，就是dict()
        """
        # f = open(dict_file_path, 'rb')
        self.word_tag_tab = {}
        f_name = resolve_filename(f)
        # f_name=C:\Users\guoph@go-goal.com\PycharmProjects\jieba\jieba\dict.txt
        # 这就是内置词典的完整路径
        # print('f_name=%s'% f_name)
        # print('f的类型：',type(f))
        for line_no, line in enumerate(f, 1):
            # 设字典是这样：
            # 就虚避实 3 i
            # 就行了 3 l
            # 就要 2328 d
            # 则打印print(line_no, line)是这样：
            # 1 就虚避实 3 i
            # 2 就行了 3 l
            # 3 就要 2328 d
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                # 用空格分开：词 频数 词性
                # 把word_tag_tab字典，赋值，键为词，值为词性
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    'invalid POS dictionary entry in %s at Line %s: %s' % (f_name, line_no, line))
        f.close()

    def makesure_userdict_loaded(self):
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence):
        """
        HMM模型，调用Viterbi算法
        1. 调用Viterbi算法，返回最大概率的状态路径（记录了每个字的最佳角色和词性）
        2. 根据上面返回的每个字的角色B,M,E,S，进行规则匹配，把发现的词语yield出去
        :param sentence: 纯中文单字串
        :return: 一系列pair词
        """
        max_prob, best_route = viterbi(
            sentence, char_state_tab_P, start_P, trans_P, emit_P)
        # 返回：
        # max_prob: 最大概率路径的概率（数值）
        # best_route: 最大概率路径(元素是每个字的角色词性状态)，章云飞大，best_route = [('B', 'nr'), ('M', 'nr'), ('E', 'nr'), ('S', 'a')]
        begin, next_i = 0, 0
        # begin：记录词的开始位置的变量

        # 已知每个字的角色和词性，进行匹配和返回词性就好了。
        for i, char in enumerate(sentence):
            pos = best_route[i][0]  # 第i个字，存储了角色-词性二元组：('B', 'nr')，这里取其角色
            if pos == 'B':
                begin = i  # 第i字是一个词的开头，记下begin位置，暂不处理
            elif pos == 'E':
                # 第i字是一个词的结尾，把从begin位置到i的词yield出去，词性选择该词结尾字的词性
                yield pair(sentence[begin:i + 1], best_route[i][1])
                next_i = i + 1
            elif pos == 'S':
                # 第i字是单字词，就直接yield出去
                yield pair(char, best_route[i][1])
                next_i = i + 1
        if next_i < len(sentence):
            # 如果这个短文的最后一个字不是E也不是S，上面的循环就分不完，就把剩余的字作为一个词全yield出去，词性选择现在这个字的词性
            # 这是可能的，因为Viterbi是基于统计的，不是规则的，角色B放最后也是可能的?
            yield pair(sentence[next_i:], best_route[next_i][1])

    def __cut_detail(self, sentence):
        """
        把route返回的最长连续单字串buf，且整个buf不在字典中的情况，进行分词。
        这才是最神奇的部分：文本=章云飞大药房，单字串buf=章云飞大——分词结果：章云飞/nr，大/a，
        1. 正则分割，纯中文单字串，和其他字符单字串
        2. 纯中文单字串，调用__cut是HMM分词函数，用Viterbi算法
        3. 其他字符单字串，判断是数值、英文还是其他特殊字符，分别yield
        :param sentence: 章云飞大，
        :return: 
        """
        blocks = re_han_detail.split(sentence)
        print('buf串=', sentence)
        # 用中文字分割成更小块
        for blk in blocks:
            if re_han_detail.match(blk):
                # 关键部分用了HMM模型的Viterbi算法，切分纯中文单字串：章云飞大
                for word in self.__cut(blk):
                    yield word
            else:
                # 如果不是中文字，细分是数值、英文还是其他特殊字符，分别yield
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):
                            yield pair(x, 'm')
                        elif re_eng.match(x):
                            yield pair(x, 'eng')
                        else:
                            yield pair(x, 'x')

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng1.match(l_word):
                buf += l_word
                x = y
            else:
                if buf:
                    yield pair(buf, 'eng')
                    buf = ''
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
                x = y
        if buf:
            yield pair(buf, 'eng')
            buf = ''


    def __cut_DAG(self, sentence):
        """
        这个函数细节上看得不是很清楚。主要就是单字词串在字典里有的时候，直接一个字一个字输出？那你干嘛把他们攒到一起。。
        连续文本片段分词总理函数。
        1. 调用函数创建词网DAG
        2. 调用函数计算各位置到最后的最大概率路径和组词位置route
        3. 根据route判断每个词语的词性并返回
        :param sentence: 连续文本片段
        :return: Generator[pair]
        逻辑好简单啊，就是根据最大概率路径来分，但是尽量把单字词也试着组词：
        1. 如果路径产生多字词，直接yield
        2. 如果路径产生单字词，就攒到buf里（不断拼接在一起），这时候不判断能不能组词，
            直到route给出多字词，然后才把到目前为止攒在buf里的单字词串，一起处理，处理完了再直接yield当前多字词。处理单字词串逻辑如下：
            i. buf里只攒了一个单字词，直接yield（字典没有的话，就词性为x）
            ii. buf里有多个单字词，而且整个buf字串在字典里没有，就调用__cut_detail函数，用返回生成器yield
            iii. buf里有多个单字词，而且整个buf子串在字典里有（较少见？比如：想去药房），就直接把每个字输出——这个操作逻辑不太懂
        """
        print('传入文本片段 = ', sentence)
        DAG = self.tokenizer.get_DAG(sentence)
        # 获得了词网（记录了所有成词可能，但不计频数）
        print('DAG = ', DAG)
        route = {}
        # 路径初始化！

        self.tokenizer.calc(sentence, DAG, route)
        print('计算后路径：', route)
        # 从而字典route=
        # {
        # 6: (0, 0),
        # 5: (P(房), 5),
        # 4: (P(药房)*1, 5),
        # 3: (P(大)*P(药房)*1, 3),
        # 2: (P(姓)*P(大)*P(药房)*1, 2),
        # 1: (P(百姓)*P(大)*P(药房)*1, 2),
        # 0: (P(老百姓)*P(大)*P(药房)*1, 2)
        # }
        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            # print('x=',x)
            # y=route[0][1]+1 = 2+1 = 3,这是第二个词的开头位置
            y = route[x][1] + 1
            # 下面就是第一个词
            l_word = sentence[x:y]
            # print('l_word = ', l_word)
            if y - x == 1:
                # 说明切出的是单字词,不过这里先不处理，后面一起判断处理
                # print('y-x=1,l_word=%s buf=%s' % (l_word,buf))
                buf += l_word

            else:  # y - x > 1
                # print('y-x > 1,l_word=%s buf=%s' % (l_word,buf))
                if buf:
                    # print('buf.length=',len(buf),'buf=',buf)
                    if len(buf) == 1:
                        # print('buf.len = 1', buf)
                        # 这说明缓冲区存的那个词长度为1，说明上一步是单字词，否则buf一定会重置为空字符
                        # 这里是当前词长度>1，前一个词长度=1的情况，先把前一个单字词返回，能取到词性就返回词性，取不到就设词性为x
                        # print('yield: ', buf)
                        yield pair(buf, self.word_tag_tab.get(buf, 'x'))
                    elif not self.tokenizer.FREQ.get(buf):
                        # print('buf.len > 1, 且没有词性', buf)
                        # 缓冲区buf长度>1，且取不到buf的词性（字典里没有）
                        # 这才是最神奇的部分：文本=章云飞大药房，单字串buf=章云飞大——分词结果：章云飞/nr，大/a，
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            # print('yield: ', t.word)
                            yield t
                    else:
                        # print('buf.len > 1, 且有词性', buf)
                        # buf长度>1，而且字典里有词性，但是最大概率路径主张把他们切开？?
                        # 同样buf长度>1，如果词典里没有这个词，就cut_detail,有这个词，就直接一个个分开？？？？
                        for elem in buf:
                            # print('element = ', elem)
                            # print('yield: ', elem)
                            yield pair(elem, self.word_tag_tab.get(elem, 'x'))
                    buf = ''
                # print('yield: ', l_word)
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
            # x设为下一个词的开头位置，进入下一个循环
            x = y
            # print('更新后x=',x)

        # 终于x=N，最后一步处理和前面类似。
        # 就是说如果最后一个字是单字词，那么buf就不会在上面的循环中处理清空。所以要单独处理
        if buf:
            # print('x= ',N)
            if len(buf) == 1:
                # print('yield: ', buf)
                yield pair(buf, self.word_tag_tab.get(buf, 'x'))
            elif not self.tokenizer.FREQ.get(buf):
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    # print('yield: ', t)
                    yield t
            else:
                for elem in buf:
                    # print('yield: ', elem)
                    yield pair(elem, self.word_tag_tab.get(elem, 'x'))

    def __cut_internal(self, sentence, HMM=True):
        """
        内部切词函数总理，把句子分成标点空格等，把连续文本片段交给__cut_DAG切分
        :param sentence: 
        :param HMM: 
        :return: 
        """
        # 确保用户词典已加载
        self.makesure_userdict_loaded()
        # 确保句子是str类型，如果是就不变，不是就解码成str
        sentence = strdecode(sentence)

        blocks = re_han_internal.split(sentence)
        # 分割，把中文字母数字和.+#&_，作为分割的依据，分出的数组.
        # 比如句子='今天，下雨了。'
        # blocks = ['', '今天', '，', '下雨了', '。'] 5个元素
        # 因为第一个字是中文字，属于分隔符，所以前面有个空字符。最后一个字不是，所以后面没有空字符
        # print(blocks)
        if HMM:
            cut_blk = self.__cut_DAG
        #     默认值是用了HMM了
        else:
            cut_blk = self.__cut_DAG_NO_HMM

        for blk in blocks:
            # 遍历['', '今天', '，', '下雨了', '。'] ，有无标点正文，有标点符号，有空字符换行符
            if re_han_internal.match(blk):
                # print('无标点正文：',blk)
                # 是无标点正文：中文字母数字小数点+#&_
                # 分词的关键部分↓↓
                for word in cut_blk(blk):
                    yield word
            else:
                # print('标点空格换行符：',blk)
                # 标点符号或空字符、换行符
                tmp = re_skip_internal.split(blk)
                # 用空格换行符分割（因为可能标点符号和换行符空格连在一起）
                for x in tmp:
                    if re_skip_internal.match(x):
                        # 如果是空格和换行符，直接切出，标记词性为x
                        yield pair(x, 'x')
                    else:
                        # 如果不是换行符和空格，
                        for xx in x:
                            # 如果是数值型，返回词性为m
                            if re_num.match(xx):
                                yield pair(xx, 'm')
                            # 如果是英文，返回词性eng——这不可能，这属于正文，在上面处理的.
                            elif re_eng.match(x):
                                yield pair(xx, 'eng')
                            else:
                                # 未知词性，设为x
                                yield pair(xx, 'x')

    def _lcut_internal(self, sentence):
        return list(self.__cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence):
        return list(self.__cut_internal(sentence, False))

    def cut(self, sentence, HMM=True):
        # print('POSTokenizer.cut')
        for w in self.__cut_internal(sentence, HMM=HMM):
            # print('POS分词切出一个词了：',w) #  的/uj
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

# default Tokenizer instance

dt = POSTokenizer(jieba.dt)

# global functions

initialize = dt.initialize


def _lcut_internal(s):
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s):
    return dt._lcut_internal_no_hmm(s)


def cut(sentence, HMM=True):
    """
    Global `cut` function that supports parallel processing.

    Note that this only works using dt, custom POSTokenizer
    instances are not supported.
    """
    # print('sentence = ', sentence)
    global dt
    if jieba.pool is None:
        # 目前是这个
        for w in dt.cut(sentence, HMM=HMM):
            # print('切出一个词了：',w) # 示例：的/uj，即：词/词性
            yield w
    else:
        parts = strdecode(sentence).splitlines(True)
        if HMM:
            result = jieba.pool.map(_lcut_internal, parts)
        else:
            result = jieba.pool.map(_lcut_internal_no_hmm, parts)
        for r in result:
            for w in r:
                yield w


def lcut(sentence, HMM=True):
    return list(cut(sentence, HMM))
#  .....原来就是直接转成list，怪不得，里面和外面转list一样。。
