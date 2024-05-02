import json

from src.tools import show_log_base


def clean_error(results):
    cleaned_results = []
    bad_results = []
    for result in results:
        if isinstance(result, Exception):
            bad_results.append(result)
        else:
            cleaned_results.append(result)
    return cleaned_results, bad_results


def write_dataset_sharegpt(
        system,
        human_datas,
        gpt_datas,
        human_source_tag=None,
        human_output_tag=None,
        gpt_tag=None,
        instruction=None,
):
    converted_data = []
    if len(human_datas) != len(gpt_datas):
        print("WARNING: The number of human data and gpt data is not equal")
        print("ling-data: Try to use the minimum number of data")

    for human_data, gpt_data in zip(human_datas, gpt_datas):
        human_parts = [instruction, human_source_tag, human_data, human_output_tag]
        human_conversation_part = "\n\n".join([part for part in human_parts if part is not None])

        gpt_conversation_part = f"{gpt_tag}\n{gpt_data}" if gpt_tag else gpt_data

        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": human_conversation_part.strip()  # 可调整
                },
                {
                    "from": "gpt",
                    "value": gpt_conversation_part.strip()  # 可调整
                }
            ],
            "system": system
        }
        converted_data.append(conversation)

    return converted_data


def save_dataset(converted_data, output_path):
    if output_path is not None:
        if not output_path.endswith('.json'):
            output_path += '.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
    else:
        return converted_data


def align_data(human_datas, gpt_datas):
    if len(human_datas) != len(gpt_datas):
        raise ValueError("The number of human data and gpt data is not equal")

    # 使用列表推导过滤掉异常数据
    filtered_human_datas = []
    filtered_gpt_datas = []

    for human_data, gpt_data in zip(human_datas, gpt_datas):
        if not isinstance(human_data, Exception) and not isinstance(gpt_data, Exception):
            filtered_human_datas.append(human_data)
            filtered_gpt_datas.append(gpt_data)

    return filtered_human_datas, filtered_gpt_datas


# def merge_converted_data_to_multiple_rounds(converted_data1, converted_data2):
#     if len(converted_data1) != len(converted_data2):
#         raise ValueError("The number of human data and gpt data is not equal")
#
#     merged_data = []
#
#     for data1, data2 in zip(converted_data1, converted_data2):
#         data1['conversations'].extend(data2['conversations'])
#         merged_data.append(data1)
#
#     return merged_data


# def merge_converted_data_to_multiple_rounds(list_of_converted_data):
#     # 确保所有列表长度一致
#     if not all(len(data) == len(list_of_converted_data[0]) for data in list_of_converted_data):
#         raise ValueError("All lists must have the same length")
#
#     # 初始化合并后的数据列表
#     merged_data = []
#
#     # 使用 zip 函数处理每个列表中相应位置的数据
#     for data_items in zip(*list_of_converted_data):
#         # 创建新的字典来存储合并后的结果
#         merged_item = {}
#
#         # 合并 'conversations' 列表
#         merged_item['conversations'] = []
#         for data in data_items:
#             merged_item['conversations'].extend(data['conversations'])
#
#         # 添加其它可能需要合并的字段，这里以 'conversations' 为例
#         # 如果有其他共同字段需要合并，可以在这里添加处理代码
#
#         # 将合并后的数据项添加到结果列表中
#         merged_data.append(merged_item)
#
#     return merged_data


def merge_convert_data_to_multiple_rounds(converted_data):
    if len(converted_data) < 2:
        raise ValueError("The number of rounds is less than 2")

    merged_data = converted_data[0]

    for data in converted_data[1:]:
        merged_data['conversations'].extend(data['conversations'])

    return merged_data


def compare_list_structures(lst1, lst2):
    # 检查两个列表的长度是否相同
    if len(lst1) != len(lst2):
        print("不匹配的情况：")
        print("长度不同")
        return False

    # 遍历列表中的每个元素
    for elem1, elem2 in zip(lst1, lst2):
        # 如果两个元素都是列表，递归比较它们的结构
        if isinstance(elem1, list) and isinstance(elem2, list):
            if not compare_list_structures(elem1, elem2):
                print("不匹配的元素：")
                print(elem1, elem2)
                return False
        # 如果一个是列表而另一个不是，结构不同
        elif isinstance(elem1, list) or isinstance(elem2, list):
            print("不匹配的列表：")
            print(elem1, elem2)
            return False

    # 所有对应元素都匹配，结构相同
    return True


def dataset_sharegpt(worker_dict):
    results = []
    source_list = worker_dict.get('source')
    if source_list is None:
        raise ValueError("No source provided for dataset_sharegpt")
    source_data = worker_dict.get('data')
    if source_data is None:
        raise ValueError("No data provided for dataset_sharegpt")

    conversations = worker_dict['args']['conversations']

    human_conversations = conversations[0::2]
    gpt_conversations = conversations[1::2]

    if len(human_conversations) != len(gpt_conversations):
        raise ValueError("Number of human and gpt conversations do not match")

    converted_data = []

    for human_conversation, gpt_conversation in zip(human_conversations, gpt_conversations):
        human_source = human_conversation['source']
        gpt_source = gpt_conversation['source']
        if human_source not in source_data or gpt_source not in source_data:
            raise ValueError(f"Source {human_source} or {gpt_source} not found in data")

        human_datas = []
        gpt_datas = []
        for human_data, gpt_data in zip(source_data[human_source], source_data[gpt_source]):

            if not compare_list_structures(human_data, gpt_data):
                raise ValueError(f"Data structures for {human_source} and {gpt_source} do not match")

            human_data, gpt_data = align_data(human_data, gpt_data)
            human_datas.extend(human_data)
            gpt_datas.extend(gpt_data)

        system = worker_dict['args'].get('system')
        human_source_tag = human_conversation.get('source_tag')
        human_output_tag = human_conversation.get('output_tag')
        gpt_tag = gpt_conversation.get('output_tag')
        instruction = worker_dict['args'].get('instruction')

        converted_data.append(
            write_dataset_sharegpt(
                system=system,
                human_datas=human_datas,
                gpt_datas=gpt_datas,
                human_source_tag=human_source_tag,
                human_output_tag=human_output_tag,
                gpt_tag=gpt_tag,
                instruction=instruction
            )
        )

    if len(converted_data) > 1:
        converted_data = merge_convert_data_to_multiple_rounds(converted_data)

    results.append(converted_data)

    output_path = worker_dict.get('output_path')

    if output_path is not None:
        save_dataset(results[0], output_path)

    show_log_base(worker_dict, results[0][0][0], worker_dict['name'])

    return results


DATASET_DICT = {"dataset_sharegpt": dataset_sharegpt}


def get_dataset_builder(builder_type):
    return DATASET_DICT[builder_type]
