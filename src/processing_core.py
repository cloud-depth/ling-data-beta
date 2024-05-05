import json
import os
import sys
from datetime import datetime
import copy


# 获取当前时间的时间戳
timestamp = datetime.now().timestamp()
dt_object = datetime.fromtimestamp(timestamp)
formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

DEFAULT_CONFIG_DICT = {
    "SAVE_ROOT": "data/saves",
    "DATA_ROOT": "data",
    "LLM_API_BASE": "https://ling-api.com/v1",
    "LLM_API_KEY": "0",
    "SHOW_LOG": True,
    "SAVE_RESULTS": True,
    "SAVE_ARGS": True,
    "PROJECT_NAME": str(formatted_date),
    "SEPERATOR": "⫘",
    "VERSION": "0.0.4 beta",
    "OVERWRITE": False,
}


def set_default_environ():
    for key, value in DEFAULT_CONFIG_DICT.items():
        os.environ[key] = str(value).lower()


def welcome():
    print(f"欢迎使用LingData v{DEFAULT_CONFIG_DICT['VERSION']}")
    print(f"当前时间：{formatted_date}")
    print(f"项目名称：{os.getenv('PROJECT_NAME')}")
    print(f"数据保存路径：{os.getenv('SAVE_DIR')}")
    print(f"数据源路径：{os.getenv('DATA_ROOT')}")
    print(f"模型API地址：{os.getenv('LLM_API_BASE')}")
    print(f"模型API密钥：{os.getenv('LLM_API_KEY')}")
    print(f"是否显示日志：{os.getenv('SHOW_LOG')}")
    print(f"是否保存结果：{os.getenv('SAVE_RESULTS')}")
    print(f"是否保存参数文件：{os.getenv('SAVE_ARGS')}")
    print(f"分隔符：{os.getenv('SEPERATOR')}")
    print(os.getenv('SEPERATOR') * 50)
    print(f"是否覆盖结果：{os.getenv('OVERWRITE')}")


class LingData:

    def __init__(self, databuilder_args_path):
        set_default_environ()
        self.databuilder_args_path = databuilder_args_path
        self.builders_args = self.read_databuilder_args()
        self.read_env_args()
        welcome()
        self.workers_dict = self.get_processors()
        self.mk_dir()
        self.save_args()

    def read_env_args(self):
        from src.environ import get_environ_processor
        keys = list(self.builders_args.keys())
        if self.builders_args[keys[0]]['type'].split('_')[0] == 'environ':
            environ_processor = get_environ_processor(self.builders_args[keys[0]]['type'])
            environ_processor(self.builders_args[keys[0]])
            print('已设置环境变量')
        else:
            raise ValueError("未找到environ参数")

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
        from src.custom import get_custom_processor
        from src.transformer import get_transformer

        workers_dict = copy.deepcopy(self.builders_args)
        keys = list(workers_dict.keys())

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

            elif processor.split('_')[0] == 'custom':
                try:
                    custom = get_custom_processor(processor)
                    workers_dict[key]['processor'] = custom
                except Exception as e:
                    print(f"读取{processor}错误：", e)

            elif processor.split('_')[0] == 'transformer':
                try:
                    transformer = get_transformer(processor)
                    workers_dict[key]['processor'] = transformer
                except Exception as e:
                    print(f"读取{processor}错误：", e)

            elif processor.split('_')[0] == 'environ':
                pass

            else:
                raise KeyError(f"未注册{processor}")

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
                source_data[source_name] = [result[start:end] for result in
                                            self.workers_dict[source_name]['results']]  # 切分
                source_name_list.append(source_name)
                self.workers_dict[worker]['source'] = source_name_list
            self.workers_dict[worker]['data'] = source_data

        self.workers_dict[worker]['name'] = worker

        worker_dict = copy.deepcopy(self.workers_dict[worker])
        self.workers_dict[worker]['results'] = processor(worker_dict)  # 运行processor

        if os.getenv('SHOW_LOG') == 'true':
            print(f"{worker}运行完成")

        if self.workers_dict[worker].get('save_result') is None:
            save_result = os.getenv('SAVE_RESULTS', default='true')
        else:
            save_result = str(self.workers_dict[worker].get('save_result')).lower()
            self.mk_dir(save_result)
        if save_result == 'true':
            try:
                self.save_result(worker)
            except Exception as e:
                print(f"保存{worker}结果错误：", e)

        return self.workers_dict[worker]

    def run_all(self):
        for key in self.workers_dict:
            try:
                self.run(key)
            except Exception as e:
                print(f"运行{key}错误：", e)
        print(f"已全部运行完成")
        return self.workers_dict

    def save_result(self, worker):
        save_dir = os.getenv('SAVE_DIR')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results = self.workers_dict[worker]['results']
        save_path = os.path.join(save_dir, f"{worker}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"{worker}结果已保存至{save_dir}")

    def save_results(self):
        save_dir = self.builders_args.get('SAVE_ROOT', DEFAULT_CONFIG_DICT['SAVE_ROOT'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for key in self.workers_dict:
            results = self.workers_dict[key]['results']
            save_path = os.path.join(save_dir, f"{key}.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"结果已保存至{save_dir}")

    def save_args(self):
        if os.getenv('SAVE_ARGS') == 'true':
            save_dir = os.getenv('SAVE_DIR')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, "args.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.builders_args, f, ensure_ascii=False, indent=4)
            print(f"参数文件已保存至{save_path}")

    def mk_dir(self, save_result="false"):
        save_dir = self.builders_args.get('SAVE_ROOT', DEFAULT_CONFIG_DICT['SAVE_ROOT'])
        save_args = os.getenv('SAVE_ARGS', default='true')
        save_results = os.getenv('SAVE_RESULTS', default='true')
        if save_args or save_results or save_result == 'true':
            project_name = os.getenv('PROJECT_NAME')
            dir_path = os.path.join(save_dir, project_name)
            if not os.path.exists(save_dir):
                os.makedirs(dir_path)
            else:
                if os.getenv('OVERWRITE') == 'true':
                    os.environ['SAVE_DIR'] = dir_path
                else:
                    raise FileExistsError(f"{dir_path}已存在。请设置OVERWRITE为true以覆盖")
            os.environ['SAVE_DIR'] = dir_path
        else:
            pass
