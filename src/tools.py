import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def show_log_base(worker_dict, show_result, proc_name):
    show_log = os.getenv('SHOW_LOG', default='true')
    if worker_dict.get('show_log') is not None:
        show_log = str(worker_dict.get('show_log')).lower()
    if show_log == 'true':
        separator = os.getenv('SEPERATOR', default="⫘") * 50
        print(f"{separator}\n{proc_name}预览结果：\n{show_result}\n{separator}")


class TokenLen:
    def __init__(self, encoding="qwen"):
        self.encoding = encoding
        if encoding == "qwen":
            from tokenizers import Tokenizer
            self.fast_tokenizer = Tokenizer.from_file("data/source/tokenizer/tokenizer_qwen.json")
        elif encoding == "gpt":
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")

    def __call__(self, text):
        if self.encoding == "qwen":
            t = self.fast_tokenizer.encode(text)
            return len(t.ids)
        elif self.encoding == "gpt":
            return len(self.enc.encode(text))
        elif self.encoding == "words":
            return len(text.split())
        else:
            return len(text)


def draw_len(lengths, title='Distribution of Element Lengths', xlabel='Length of Elements', ylabel='Frequency'):
    import matplotlib.pyplot as plt

    # 绘制长度的直方图
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示图表
    plt.show()
