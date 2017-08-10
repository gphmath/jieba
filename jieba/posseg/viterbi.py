import sys
import operator
MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")

if sys.version_info[0] > 2:
    xrange = range


def get_top_states(t_state_v, K=4):
    # Top-N最佳路径？
    return sorted(t_state_v, key=t_state_v.__getitem__, reverse=True)[:K]


def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    HMM模型的Viterbi算法（前面的route是从后往前的，这里的路径还是从前往后的）
    状态是角色-词性二元组：{(B,nr), (M,v), (E,n), (S,n), (M,a), (S,nr),....}，其中角色属于{B,M,E,S},词性就是所有可能的词性
    观测值就是一个个纯中文字
    :param obs: 观测值：纯中文字串
    :param states: 状态字典，记录了每个观测值（中文单字），可能属于的状态（不含频数概率等信息）。
                    每个字的可能状态是所有状态的子集，相当于在这个子集之外的状态q的条件概率为0：P(i=q|o)=0
    :param start_p: 文本的初始状态概率，文本第一个字属于这个状态的概率（应该取了对数）：
                    一个item形如：('B', 'a'): -4.76230，B的初始概率肯定更大些
    :param trans_p: 状态转移矩阵：键是每个状态，值是这个状态可能转移到的状态和转移概率
                    ('B', 'a'): {('E', 'a'): -0.0050648453069648755, ('M', 'a'): -5.287963037107507},
    :param emit_p:  发射矩阵：键是每个状态，值是这个状态可能产生的所有字及其概率
                    ('E', 'p'): {'\u4e4b': -6.565515869430067, '\u4e86': -2.026203081578427}
    :return: (prob, route)
    """
    Delta_Max_Prob_t_st = [{}]
    # Delta[t][state]: 从开头到位置t的所有路径中，位置t的状态是state的，最大概率路径的概率
    Phi_list_Former_Char_of_best_route_t_st = [{}]
    # Phi[t][state]: 上面Delta[t][state]对应的路径中，第t-1个位置的词

    # tabular,表格，就是list里是字典，键就是字段，这里初始状态有一个空字典
    # 这两个表格，的list的长度=文本的长度，即每个字一个字典
    all_states = trans_p.keys()  # 获取所有可能的状态
    for st in states.get(obs[0], all_states):  # 初始化方法：对现在第一个字，取它所有可能状态子集，如果没取到，就取状态全集
        Delta_Max_Prob_t_st[0][st] = start_p[st] + emit_p[st].get(obs[0], MIN_FLOAT)
        # Delta[0][state] = start_p[state]*emit_P[state->0]
        Phi_list_Former_Char_of_best_route_t_st[0][st] = ''
        # Phi[0][state] = ''.这时候没有词，因为0-1=-1位置不存在
    for t in xrange(1, len(obs)):
        # 位置t从1到M
        Delta_Max_Prob_t_st.append({})
        Phi_list_Former_Char_of_best_route_t_st.append({})
        # prev_states = get_top_states(V[t-1])
        prev_states = [
            x for x in Phi_list_Former_Char_of_best_route_t_st[t - 1].keys() if len(trans_p[x]) > 0]
        # 这是记录Phi[t-1][state],所有可能state。之所以要判断值的长度是因为有：('B', 'ag'): {}这种不会转移出去的状态
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))
        # 集合，t-1时刻的所有可能概率中可能产生的t时刻的状态
        obs_states = set(
            states.get(obs[t], all_states)) & prev_states_expect_next
        # set & set: 表示两个集合的交集
        # 前一个集合是t位置的这个词语可能的状态集合
        # 后一个集合是前一个位置的状态们可能产生的t位置的状态集合
        # 综上，结果是当前t位置真正可能的所有状态集合
        if not obs_states:
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states
        # 如果交集为空，那就取前面t-1可能产生的t位置状态集合
        # 如果前面t-1可能产生的t位置状态集合也为空，那就选所有状态全集
        # 这么多步才把需要计算哪些状态的子集定义好。

        # 下面是递推公式：
        # obs_states: t位置实际可能的状态子集；st：t位置的某个实际状态
        # prev_states：t-1位置的所有状态子集，st0：t-1位置的某个状态
        for st in obs_states:
            prob, state = max((Delta_Max_Prob_t_st[t - 1][st0] + trans_p[st0].get(st, MIN_INF) +
                               emit_p[st].get(obs[t], MIN_FLOAT), st0) for st0 in prev_states)
            # 计算所有Delta[t][st] = Max { Delta[t-1][st0]*trans_p[st0->st] } * emit_p[st->o_t]; 对所有st0
            # Phi[t][st]，就记录上面这个最大值的argmax_st0
            Delta_Max_Prob_t_st[t][st] = prob
            Phi_list_Former_Char_of_best_route_t_st[t][st] = state
        # 以上递推式完成

    # 下面是终止环节：
    last = [(Delta_Max_Prob_t_st[-1][st], st) for st in
            Phi_list_Former_Char_of_best_route_t_st[-1].keys()]
    # if len(last)==0:
    #     print obs
    max_prob, state = max(last)
    # 最大概率，最大概率路径的最后一个节点状态（这回不是前一节点了，否则直接用Phi查即可）
    # 以上迭代终止

    # 下面根据Delta和Phi，最后的概率prob、最佳路径的最后节点state。进行最优路径回溯
    best_route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        best_route[i] = state
        state = Phi_list_Former_Char_of_best_route_t_st[i][state]  # 根据最佳路径在t位置的状态，去选t-1时刻的状态
        i -= 1
    print('max_prob = ',max_prob)
    print('best_route = ',best_route)
    # 章云飞大，best_route = [('B', 'nr'), ('M', 'nr'), ('E', 'nr'), ('S', 'a')]
    return max_prob, best_route
