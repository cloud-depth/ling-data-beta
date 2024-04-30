import os


def show_log_base(worker_dict, show_result, proc_name):
    show_log = os.getenv('SHOW_LOG', default='true')
    if worker_dict.get('show_log') is not None:
        show_log = str(worker_dict.get('show_log')).lower()
    if show_log == 'true':
        separator = os.getenv('SEPERATOR', default="⫘") * 50
        print(f"{separator}\n{proc_name}预览结果：\n{show_result}\n{separator}")
