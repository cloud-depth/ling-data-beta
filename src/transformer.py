def transformer_id(worker_dict):
    source_list = worker_dict.get('source')
    results = []

    if source_list is None:
        raise ValueError("No source provided for transformer_id")

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                results.append(data)

    return results


def transformer_chapter_len(worker_dict):
    source_list = worker_dict.get('source')
    results = []

    if source_list is None:
        raise ValueError("No source provided for transformer_chapter_len")

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                results.append([item[1] for item in data])

    return results


def transformer_llm_001_dataset(worker_dict):
    source_list = worker_dict.get('source')
    results = []

    if source_list is None:
        raise ValueError("No source provided for transformer_llm_001_dataset")

    for source in source_list:
        if source not in worker_dict['data']:
            raise ValueError(f"Source {source} not found in data")
        else:
            for data in worker_dict['data'][source]:
                results.append([item[1] for item in data])

    return results


TRANSFORMER_DICT = {
    "transformer_id": transformer_id,
    "transformer_spliter_chapter_spliter": transformer_chapter_len,
    "transformer_llm_001_dataset": transformer_llm_001_dataset
}


def get_transformer(processor):
    return TRANSFORMER_DICT[processor]
