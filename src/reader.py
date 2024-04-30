import re
from langdetect import detect
import os
from src.tools import show_log_base


def make_filepath_list(file_path_list):
    # 初始化一个列表来存储文件路径
    file_paths = []
    for file_path in file_path_list:
        if os.path.isdir(file_path):
            # 使用os.walk()遍历目录
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    # 使用os.path.join()拼接完整的文件路径
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)
        elif os.path.isfile(file_path):
            file_paths.append(file_path)
    return file_paths


def count_words(text, language):
    # 增加对日语的支持
    if language in ['zh-cn', 'zh-tw']:
        # 中文：计算汉字数量
        count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    elif language == 'en':
        # 英文：使用正则表达式匹配英文单词
        words = re.findall(r'\b\w+\b', text)
        count = len(words)
    elif language == 'ja':
        # 日语：计算假名、片假名和汉字的数量
        count = sum(1 for char in text if '\u3040' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff')
    else:
        # 其他语言：简单按空格分割计算“单词”数
        words = text.split()
        count = len(words)
    return count


def txt_reader(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
            language = detect(content)
            count = count_words(content, language)

            # show_log = os.getenv('SHOW_LOG', default='true')
            # if show_log == 'true':
            #     separator = os.getenv('SEPERATOR', default="⫘") * 50
            #     print(f"{separator}\n文件主要语言识别为：{language}")
            #     print(f"文件字数为：{count}\n''\n{separator}")

            return content, count
    except Exception as e:
        print("发生错误：", e)
        return None


def reader_txt(worker_dict):
    results = []
    file_path_list = worker_dict['args']['file_path']

    if type(file_path_list) is str:
        file_path_list = [file_path_list]

    file_path_list = make_filepath_list(file_path_list)

    for file_path in file_path_list:
        results.append(txt_reader(file_path)[0])

    show_log_base(worker_dict, results[0][0:100], worker_dict['name'])

    return results


READER = {"reader_txt": reader_txt}


def get_reader(reader_name):
    return READER[reader_name]
