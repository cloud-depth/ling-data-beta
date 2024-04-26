import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(project_root)


class LingData:

    def __init__(self, databuilder_args_path):
        self.databuilder_args_path = databuilder_args_path
        self.builders_args = self.read_databuilder_args()
        self.workers_dict = self.get_processors()

    def read_databuilder_args(self):
        try:
            with open(self.databuilder_args_path, 'r', encoding='utf-8') as f:
                databuilder_args = json.load(f)
                return databuilder_args
        except Exception as e:
            print(f"读取{self.databuilder_args_path}错误：", e)
            return None

    def get_processors(self):
        from src.reader import get_reader
        from src.spliter import get_spliter
        from src.dataset import get_dataset_builder
        from src.llm import get_llm_processor

        keys = list(self.builders_args.keys())
        workers_dict = self.builders_args

        for key in keys:
            processor = self.builders_args[key]['type']
            if processor.split('_')[0] == 'reader':
                try:
                    reader = get_reader(processor)
                    workers_dict[key]['processor'] = reader
                except Exception as e:
                    print(f"读取{processor}错误：", e)

            elif processor.split('_')[0] == 'spliter':
                try:
                    spliter = get_spliter(processor)
                    workers_dict[key]['processor'] = spliter
                except Exception as e:
                    print(f"读取{processor}错误：", e)

            elif processor.split('_')[0] == 'llm':
                try:
                    llm = get_llm_processor(processor)
                    workers_dict[key]['processor'] = llm
                except Exception as e:
                    print(f"读取{processor}错误：", e)

            elif processor.split('_')[0] == 'dataset':
                try:
                    writer = get_dataset_builder(processor)
                    workers_dict[key]['processor'] = writer
                except Exception as e:
                    print(f"读取{processor}错误：", e)

            else:
                raise KeyError(f"未找到{processor}")

        return workers_dict

    @staticmethod
    def extract_source(source):
        import re
        match = re.match(r"(\w+)(?:\[(\d+):(\d+)\])?", source)
        if match:
            name = match.group(1)  # 提取 'spliter'
            if match.group(2) and match.group(3):  # 确保数字存在
                start_index = int(match.group(2))  # 提取起始数字
                end_index = int(match.group(3))  # 提取结束数字
                return name, start_index, end_index
            return name, None, None

    def run(self, worker):
        processor = self.workers_dict[worker]['processor']
        source = self.workers_dict[worker].get('source')
        if source:
            source_data = {}
            source_name_list = []
            if type(source) is str:
                source = [source]
            elif type(source) is not list:
                raise ValueError(f"{worker}的source数据类型错误：{source}")
            for s in source:
                source_name, start, end = self.extract_source(s)
                source_data[source_name] = [result[start:end] for result in self.workers_dict[source_name]['results']]  # 切分
                source_name_list.append(source_name)
                self.workers_dict[worker]['source'] = source_name_list

            self.workers_dict[worker]['data'] = source_data

        self.workers_dict[worker]['results'] = processor(self.workers_dict[worker])
        return self.workers_dict[worker]

    def run_all(self):
        for key in self.workers_dict:
            try:
                self.run(key)
                print(f"{key}运行完成")
                print("-----------------------------------------------------------------------------------------------")
            except Exception as e:
                print(f"运行{key}错误：", e)
        print(f"已全部运行完成")
        return self.workers_dict


if __name__ == '__main__':
    # 读取数据构建参数
    core = LingData('../databuilder_args/example_arg1.json')
    databuilder_args = core.read_databuilder_args()
    # print(databuilder_args)
    print(core.workers_dict)
    result1 = core.run("reader")
    print(result1['results'][0][0:100])
    result2 = core.run("spliter")
    print(result2['results'][0][0])
    result3 = core.run("process1")
    print(result3['results'][0][0])
    result4 = core.run("dataset_builder")
    print(result4['results'][0][0][0])
