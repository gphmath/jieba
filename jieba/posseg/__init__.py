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
    from .char_state_tab import P as char_state_tab_P  # BMES和对应
    from .prob_start import P as start_P  # 初始状态概率
    from .prob_trans import P as trans_P  # 状态转移矩阵
    from .prob_emit import P as emit_P  # 发射矩阵


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

    def load_word_tag(self, f):
        self.word_tag_tab = {}
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    'invalid POS dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()

    def makesure_userdict_loaded(self):
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence):
        prob, pos_list = viterbi(
            sentence, char_state_tab_P, start_P, trans_P, emit_P)
        begin, nexti = 0, 0

        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield pair(sentence[begin:i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == 'S':
                yield pair(char, pos_list[i][1])
                nexti = i + 1
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    def __cut_detail(self, sentence):
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):
                for word in self.__cut(blk):
                    yield word
            else:
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
        疑似构成词网
        :param sentence: 
        :return: 
        """
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}

        self.tokenizer.calc(sentence, DAG, route)

        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield pair(buf, self.word_tag_tab.get(buf, 'x'))
                    elif not self.tokenizer.FREQ.get(buf):
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:
                            yield pair(elem, self.word_tag_tab.get(elem, 'x'))
                    buf = ''
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
            x = y

        if buf:
            if len(buf) == 1:
                yield pair(buf, self.word_tag_tab.get(buf, 'x'))
            elif not self.tokenizer.FREQ.get(buf):
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield pair(elem, self.word_tag_tab.get(elem, 'x'))

    def __cut_internal(self, sentence, HMM=True):
        """
        内部切词函数
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
                # 分词的关键部分
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
