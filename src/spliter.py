import re

import numpy as np
import tiktoken
import os
from src.tools import show_log_base


def extract_chapters(
        raw_text: str,
        pattern: str = None,
):
    """
    Extract chapters from a novel text using a regular expression pattern to match chapter titles
    :param raw_text: Raw text to extract chapters from
    :param pattern: Regular expression pattern to match chapter titles
    :return: List of tuples containing chapter title and content
    """
    if pattern is None:
        pattern = r'[\s:：]*(第?[0-9一二三四五六七八九十百千零壹两仨贰叁肆伍陆柒捌玖拾佰仟万]+)[:：、\-\s]*章[:：、\-\s]'

    chapters = []
    chapter_contents = []
    current_chapter_title = None
    first_chapter_found = False  # Flag to check if the first chapter has been identified

    chapter_pattern = re.compile(pattern)

    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        match = chapter_pattern.search(line)
        if match:
            if current_chapter_title is not None:
                chapters.append((current_chapter_title, '\n'.join(chapter_contents)))
            if not first_chapter_found:
                first_chapter_found = True  # Mark that the first chapter is found
            current_chapter_title = line  # Update the current chapter title
            chapter_contents = []  # Reset the contents for the new chapter
        elif first_chapter_found:
            chapter_contents.append(line)  # Append line to current chapter contents
        else:
            # If no chapter has been found yet, store lines as part of the preface
            if chapters and chapters[0][0] == "Preface":
                chapters[0][1].append(line)
            else:
                chapters.insert(0, ("Preface", [line]))

    if current_chapter_title is not None:
        chapters.append((current_chapter_title, '\n'.join(chapter_contents)))

    if chapters and chapters[0][0] == "Preface":
        chapters[0] = (chapters[0][0], '\n'.join(chapters[0][1]))

    if len(chapters) == 1:
        chapters.append(('Chapter 1', chapters[0][1]))
        chapters[0] = ('Preface', '')

    return chapters


# 分段函数
def divide_str(s, sep=None):
    if sep is None:
        sep = ['\n', '.', '。']
    mid_len = len(s) // 2  # 中心点位置
    best_sep_pos = len(s) + 1  # 最接近中心点的分隔符位置
    best_sep = None  # 最接近中心点的分隔符
    for curr_sep in sep:
        sep_pos = s.rfind(curr_sep, 0, mid_len)  # 从中心点往左找分隔符
        if sep_pos > 0 and abs(sep_pos - mid_len) < abs(best_sep_pos -
                                                        mid_len):
            best_sep_pos = sep_pos
            best_sep = curr_sep
    if not best_sep:  # 没有找到分隔符
        return s, ''
    return s[:best_sep_pos + 1], s[best_sep_pos + 1:]


def strong_divide(s):
    left, right = divide_str(s)

    if right != '':
        return left, right

    whole_sep = ['\n', '.', '，', '、', ';', ',', '；',
                 '：', '！', '？', '(', ')', '”', '“',
                 '’', '‘', '[', ']', '{', '}', '<', '>',
                 '/', '\\', '|', '-', '=', '+', '*', '%',
                 '$', '#', '@', '&', '^', '_', '`', '~',
                 '·', '…']
    left, right = divide_str(s, sep=whole_sep)

    if right != '':
        return left, right

    mid_len = len(s) // 2
    return s[:mid_len], s[mid_len:]


def split_chunk(
        text,
        max_token_len: int = 1500,
        add_preface: bool = True,
        merge_min: int = 50,
        tokenizer: str = "qwen",
):
    from src.tools import TokenLen
    tokenizer_len = TokenLen(encoding=tokenizer)

    start = 1
    if add_preface:
        start = 0

    chunk_text = []

    for chapter in text[start:]:

        # chapter = chapter[1]

        split_text = chapter.split('\n')

        curr_len = 0
        curr_chunk = ''

        tmp = []

        for line in split_text:
            line_len = tokenizer_len(line)

            if line_len <= max_token_len - 5:
                tmp.append(line)
            else:
                # print('divide line with length = ', line_len)
                path = [line]
                tmp_res = []

                while path:
                    my_str = path.pop()
                    left, right = strong_divide(my_str)

                    len_left = tokenizer_len(left)
                    len_right = tokenizer_len(right)

                    if len_left > max_token_len - 15:
                        path.append(left)
                    else:
                        tmp_res.append(left)

                    if len_right > max_token_len - 15:
                        path.append(right)
                    else:
                        tmp_res.append(right)

                for line in tmp_res:
                    tmp.append(line)

        split_text = tmp

        for line in split_text:
            line_len = tokenizer_len(line)

            if line_len > max_token_len:
                print('warning line_len = ', line_len)

            if curr_len + line_len <= max_token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = line
                curr_len = line_len

        if curr_chunk:
            chunk_text.append(curr_chunk)

    def merge_short_texts(text_list, min_len):
        i = 0
        while i < len(text_list):
            if i > 0 and tokenizer_len(text_list[i]) < min_len:
                text_list[i - 1] += text_list[i]
                text_list.pop(i)
            else:
                i += 1
        return text_list

    chunk_text = merge_short_texts(chunk_text, merge_min)

    return chunk_text


def split_chunk_dist(
        text,
        dist_arg1=100,
        dist_arg2=300,
        add_preface=True,
        merge_min: int = 50,
        distribution='uniform',
        tokenizer: str = "qwen",
):
    # from src.tools import tokenizer_len
    from src.tools import TokenLen
    tokenizer_len = TokenLen(encoding=tokenizer)

    if distribution == 'normal':
        dist_func = np.random.normal
    elif distribution == 'uniform':
        dist_func = np.random.uniform
    else:
        raise ValueError(f"Invalid distribution type: {distribution}")

    start = 0 if add_preface else 1
    chunk_text = []

    for chapter in text[start:]:
        split_text = chapter.split('\n')
        curr_len = 0
        curr_chunk = ''
        tmp = []

        for line in split_text:
            line_len = tokenizer_len(line)
            # 动态确定本次的最大长度
            max_token_len = int(dist_func(dist_arg1, dist_arg2))
            # 确保max_token_len不小于一定值，比如500
            max_token_len = max(max_token_len, 100)

            if line_len <= max_token_len - 5:
                tmp.append(line)
            else:
                path = [line]
                tmp_res = []

                while path:
                    my_str = path.pop()
                    left, right = strong_divide(my_str)

                    len_left = tokenizer_len(left)
                    len_right = tokenizer_len(right)

                    if len_left > max_token_len - 15:
                        path.append(left)
                    else:
                        tmp_res.append(left)

                    if len_right > max_token_len - 15:
                        path.append(right)
                    else:
                        tmp_res.append(right)

                tmp.extend(tmp_res)

        split_text = tmp

        for line in split_text:
            line_len = tokenizer_len(line)
            max_token_len = int(dist_func(dist_arg1, dist_arg2))
            max_token_len = max(max_token_len, 100)

            if curr_len + line_len <= max_token_len:
                curr_chunk += line + '\n'
                curr_len += line_len + 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = line + '\n'
                curr_len = line_len + 1

        if curr_chunk:
            chunk_text.append(curr_chunk)

    def merge_short_texts(text_list, min_len):
        i = 0
        while i < len(text_list):
            if i > 0 and tokenizer_len(text_list[i]) < min_len:
                text_list[i - 1] += text_list[i]
                text_list.pop(i)
            else:
                i += 1
        return text_list

    chunk_text = merge_short_texts(chunk_text, merge_min)

    return chunk_text


def spliter_chapter(worker_dict):
    source_list = worker_dict.get('source')
    results = []

    if source_list is None:
        raise ValueError("No source provided for spliter_chapter")

    args = worker_dict.get('args')
    if args is not None:
        pattern = args.get('pattern', None)
    else:
        pattern = None

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                chapters = extract_chapters(data, pattern)
                results.append(chapters)

    show_log_base(worker_dict, results[0][0], worker_dict['name'])

    return results


def spliter_len(worker_dict):
    source_list = worker_dict.get('source')
    results = []

    if source_list is None:
        raise ValueError("No source provided for spliter_len")

    args = worker_dict.get('args')
    if args is not None:
        max_token_len = worker_dict['args'].get('max_token_len', 1500)
        add_preface = worker_dict['args'].get('preface', False)
        min_len = worker_dict['args'].get('min_len', 50)
        tokenizer = worker_dict['args'].get('tokenizer', 'qwen')
    else:
        max_token_len = 1500
        add_preface = False
        min_len = 50
        tokenizer = 'qwen'

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                results.append(split_chunk(
                    text=data,
                    max_token_len=max_token_len,
                    add_preface=add_preface,
                    min_len=min_len,
                    tokenizer=tokenizer
                ))

    show_log_base(worker_dict, results[0][0], worker_dict['name'])

    return results


def spliter_distribution(worker_dict):
    source_list = worker_dict.get('source')
    results = []

    if source_list is None:
        raise ValueError("No source provided for spliter_len")

    args = worker_dict.get('args')
    if args is not None:
        max_token_range = worker_dict['args'].get('max_token_range', [100, 500])
        add_preface = worker_dict['args'].get('preface', False)
        distribution = worker_dict['args'].get('distribution', 'uniform')
        min_len = worker_dict['args'].get('min_len', 50)
        tokenizer = worker_dict['args'].get('tokenizer', 'qwen')

        if distribution == 'normal':
            dist_arg1 = int(np.mean(max_token_range))
            dist_arg2 = int(np.std(max_token_range))

        elif distribution == 'uniform':
            dist_arg1 = int(max_token_range[0])
            dist_arg2 = int(max_token_range[1])

        else:
            raise ValueError(f"Invalid distribution type: {distribution}")

    else:
        dist_arg1 = 100
        dist_arg2 = 500
        add_preface = False
        distribution = 'uniform'
        min_len = 50
        tokenizer = 'qwen'

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                results.append(split_chunk_dist(
                    text=data,
                    dist_arg1=dist_arg1,
                    dist_arg2=dist_arg2,
                    add_preface=add_preface,
                    distribution=distribution,
                    merge_min=min_len,
                    tokenizer=tokenizer
                ))

    show_log_base(worker_dict, results[0][0], worker_dict['name'])

    return results


SPLITER_DICT = {
    "spliter_chapter": spliter_chapter,
    "spliter_len": spliter_len,
    "spliter_distribution": spliter_distribution,
}


def get_spliter(spliter_name):
    return SPLITER_DICT[spliter_name]
