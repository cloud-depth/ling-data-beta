# Custom.py
# This file contains the custom processors.


# 在此之上，我们可以自定义处理器，只需要在custom.py中定义一个函数，然后在CUSTOM_PROCESSOR_DICT中注册即可。
CUSTOM_PROCESSOR_DICT = {}


def get_custom_processor(processor_name: str):
    return CUSTOM_PROCESSOR_DICT[processor_name]
