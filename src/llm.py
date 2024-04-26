from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

client = OpenAI(base_url="http://192.168.31.10:9997/v1",
                api_key="0")


def llm_base(
        user_input: str,
        system_input: str = "You are a helpful assistant.",
        model=None,
        temperature=None,
        top_p=None,
) -> str:
    chat_response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system_input},
                  {'role': 'user', 'content': user_input}],
        temperature=temperature,
        top_p=top_p,
    )
    return chat_response.choices[0].message.content


def single_request(
        chunk_text,
        sys_prompt=None,
        model=None,
        example=None,
        instruction=None,
        source_tag="文本",
        output_tag="输出",
        temperature=None,
        top_p=None,
) -> (str, str):
    prompt = ""
    if instruction is not None:
        prompt += instruction
    if example is not None:
        prompt += f"\n例子：\n{example}"

    prompt += f"\n{example}\n\n{source_tag}:\n{chunk_text}\n{output_tag}:"

    result = llm_base(prompt, sys_prompt, model=model, temperature=temperature, top_p=top_p)

    return result


def multi_request(
        chunk_texts: list[str],
        sys_prompt=None,
        model=None,
        example=None,
        instruction=None,
        source_tag="文本",
        output_tag="输出",
        temperature=None,
        top_p=None,
        workers=1,
) -> list[(str, str)]:
    """
    使用多线程并发请求
    :param workers:
    :param sys_prompt:
    :param model:
    :param example:
    :param instruction:
    :param source_tag:
    :param output_tag:
    :param temperature:
    :param top_p:
    :param chunk_texts:
    :return: 列表，每个元素是一个元组，包含原始文本和处理后的文本
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    # 使用with语句自动管理线程池
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 使用enumerate获取索引并将任务提交到线程池
        futures = {executor.submit(
            single_request,
            chunk_text,
            sys_prompt,
            model,
            example,
            instruction,
            source_tag,
            output_tag,
            temperature,
            top_p): index for index, chunk_text in enumerate(chunk_texts)}

        # 初始化一个足够大的列表，用None填充，保证有足够的空间存储每个结果
        results = [None] * len(chunk_texts)
        preview_count = 5  # 预览的结果数量
        preview_shown = False  # 控制是否已显示预览的标志

        # 使用tqdm来显示进度条，并通过futures字典获取每个任务的原始索引
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing requests"):
            index = futures[future]  # 获取这个future对应的原始索引
            try:
                result = future.result()  # 获取结果
            except Exception as e:
                result = str(e)  # 如果有异常，保存异常信息为字符串
            results[index] = result  # 将结果放在正确的位置

            # 如果已经完成了足够多的任务，并且尚未显示预览
            completed_results = [res for res in results if res is not None]
            if len(completed_results) >= preview_count and not preview_shown:
                print("Preview of first few results:")
                print("Preview of first few results:")
                for res in completed_results[:preview_count]:
                    print(res)
                    print("-------------------------------------------------------------------------------------------")
                preview_shown = True  # 设置标志为True，不再打印后续结果

    return results


def llm_instruction_001(worker_dict):
    results = []
    source_list = worker_dict.get('source')
    if source_list is None:
        raise ValueError("No source provided for llm_instruction")
    source_data = worker_dict.get('data')
    if source_data is None:
        raise ValueError("No data provided for llm_instruction")

    model = worker_dict['args'].get('model')
    instruction = worker_dict['args'].get('instruction')
    example = worker_dict['args'].get('example')
    sys_prompt = worker_dict['args'].get('sys_prompt')
    source_tag = worker_dict['args'].get('source_tag')
    output_tag = worker_dict['args'].get('output_tag')
    temperature = worker_dict['args'].get('temperature', 0.7)
    top_p = worker_dict['args'].get('top_p', 0.8)
    worker = worker_dict['args'].get('worker')

    for source in source_list:
        if source not in source_data:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in source_data[source]:
                results.append(
                    multi_request(
                        chunk_texts=data,
                        sys_prompt=sys_prompt,
                        model=model,
                        example=example,
                        instruction=instruction,
                        source_tag=source_tag,
                        output_tag=output_tag,
                        temperature=temperature,
                        top_p=top_p,
                        workers=worker
                    )
                )

    return results


LLM_PROCESSOR_DICT = {"llm_001": llm_instruction_001}


def get_llm_processor(processor_name: str):
    return LLM_PROCESSOR_DICT[processor_name]
