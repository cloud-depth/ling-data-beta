import os


def environ_set(worker_dict):
    args = worker_dict.get('args')
    if args is None:
        raise ValueError("No args provided for environ_set")
    for key, value in args.items():
        os.environ[key] = str(value).lower()
    pass


ENVIRON_PROCESSOR_DICT = {
    "environ_set": environ_set,
}


def get_environ_processor(processor):
    return ENVIRON_PROCESSOR_DICT[processor]
