import os


def show_log_base(worker_dict, show_result, proc_name):
    show_log = os.getenv('SHOW_LOG', default='true')
    if worker_dict.get('show_log') is not None:
        show_log = str(worker_dict.get('show_log')).lower()
    if show_log == 'true':
        separator = os.getenv('SEPERATOR', default="⫘") * 50
        print(f"{separator}\n{proc_name}预览结果：\n{show_result}\n{separator}")


class TokenLen:
    def __init__(self, encoding="qwen"):
        from tokenizers import Tokenizer
        self.encoding = encoding
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tokenizer_path_dict = {
            "qwen": "data/source/tokenizer/tokenizer_qwen.json",
            "gpt": "data/source/tokenizer/tokenizer_gpt.json",
            "gpt4o": "data/source/tokenizer/tokenizer_gpt4o.json",
            # 可拓展
        }
        tokenizer_path = os.path.join(self.project_root, tokenizer_path_dict.get(self.encoding))
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def __call__(self, text):
        t = self.tokenizer.encode(text)
        return len(t.ids)


def draw_len(lengths, title='Distribution of Element Lengths', xlabel='Length of Elements', ylabel='Frequency'):
    import matplotlib.pyplot as plt

    # 绘制长度的直方图
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示图表
    plt.show()
