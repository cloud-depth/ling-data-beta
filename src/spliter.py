import re
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
        max_token_len=1500,
        add_preface=True
):
    enc = tiktoken.get_encoding("cl100k_base")

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
            line_len = len(enc.encode(line))

            if line_len <= max_token_len - 5:
                tmp.append(line)
            else:
                # print('divide line with length = ', line_len)
                path = [line]
                tmp_res = []

                while path:
                    my_str = path.pop()
                    left, right = strong_divide(my_str)

                    len_left = len(enc.encode(left))
                    len_right = len(enc.encode(right))

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
            line_len = len(enc.encode(line))

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

    return chunk_text


def spliter_novel(worker_dict):
    """
    已弃用，将在v0.0.4中删除
    :param worker_dict:
    :return:
    """
    source_list = worker_dict.get('source')
    results = []

    if source_list is None:
        raise ValueError("No source provided for spliter_novel")

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                chapters = extract_chapters(data)
                max_token_len = worker_dict['args'].get('max_token_len', 1500)
                add_preface = worker_dict['args'].get('preface', False)
                results.append(split_chunk(chapters, max_token_len, add_preface))

    print("spliter_novel预览结果：")
    print(results[0][0])
    print("-----------------------------------------------------------------------------------------------------------")

    return results


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
    else:
        max_token_len = 1500
        add_preface = False

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                results.append(split_chunk(data, max_token_len, add_preface))

    show_log_base(worker_dict, results[0][0], worker_dict['name'])

    return results


SPLITER_DICT = {
    "spliter_novel": spliter_novel,
    "spliter_chapter": spliter_chapter,
    "spliter_len": spliter_len,
}


def get_spliter(spliter_name):
    return SPLITER_DICT[spliter_name]
