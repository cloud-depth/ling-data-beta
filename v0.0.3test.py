from src.processing_core import LingData


if __name__ == '__main__':
    # 读取数据构建参数
    core = LingData('databuilder_args/v0.0.3example.json')
    result_r1 = core.run('reader1')
    # # print(type(result_r1['results'][0]))
    result_s1 = core.run('spliter1')
    # # print(result_s1['results'][0][1])
    result_t1 = core.run('transformer1')
    result_s2 = core.run('spliter2')
    # # print(result_s2['results'][0][0])
    result_l1 = core.run('llm1')
    # print(result_l1['results'][0][0])
    result_t2 = core.run('transformer2')
    # print(result_t2['results'][0][0])
    result_d1 = core.run('dataset1')
    # print(result_d1['results'][0][0][0])
