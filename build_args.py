import json

args_example = {
    "environ": {
        "type": "environ_set",
        "args": {
            "PROJECT_NAME": "qiongming_v2",
            "SAVE_ROOT": "data/saves",
            "LLM_API_BASE": "http://192.168.31.10:9997/v1",
            "LLM_API_KEY": "0",
            "SHOW_LOG": True,
            "SAVE_ARGS": True,
            "SAVE_RESULTS": True
        }
    },
    "reader1": {
        "type": "reader_txt",
        "args": {
            "file_path": "data/source/qiongming.txt"
        }
    },
    "spliter1": {
        "type": "spliter_chapter",
        "source": "reader1"
    },
    "transformer1": {
        "type": "transformer_spliter_chapter_spliter",
        "source": "spliter1[0:20]"
    },
    "spliter2": {
        "type": "spliter_len",
        "source": "transformer1",
        "args": {
            "max_token_len": 300,
            "preface": False
        }
    },
    "llm1": {
        "type": "llm_001",
        "source": "spliter2[0:100]",
        "args": {
            "model": "qwen1.5-110b",
            "instruction": "提取{时间}{地点}{人物}{起因}{经过}{结果}",
            "example": "例子：\n文本：\n叶临渊在一个幽静的暗室中醒来，身边放着一柄生锈的剑。\n石壁之上镶嵌着青铜古灯，壁上绘画繁复，彩绘的笔画保存完好，栩栩如生，没有丝毫的剥落。\n一袭白衣古静如素，那张年轻的少年脸庞在昏暗的石室间清秀如同少女。\n他看着那柄锈迹斑斑，毫无灵气的古朴长剑，默然许久，他终于幽幽叹了一口气：“临渊羡鱼，终于被深渊吞噬了。”\n他推开石门，走进了光里。\n\n梗概：\n{时间}：未提及具体时间\n{地点}：幽静的暗室\n{人物}：叶临渊\n{起因}：叶临渊在暗室中醒来，身边放着一柄生锈的剑\n{经过}：\n1. \n叶临渊观察了石壁上的青铜古灯和彩绘\n2. 他看着那柄锈迹斑斑的古朴长剑，默然许久\n3. 他幽幽叹了一口气，说出\"临渊羡鱼，终于被深渊吞噬了\"\n{结果}：叶临渊推开石门，走进了光里",
            "sys_prompt": "You are a story extractor.",
            "source_tag": "文本：",
            "output_tag": "梗概：",
            "temperature": 0.7,
            "top_p": 0.8,
            "workers": 16
        },
        "show_prompt": True
    },
    "transformer2": {
        "type": "transformer_llm_001_dataset",
        "source": "llm1"
    },
    "dataset1": {
        "type": "dataset_sharegpt",
        "source": [
            "spliter2[0:100]",
            "transformer2"
        ],
        "args": {
            "conversations": [
                {
                    "from": "human",
                    "source": "transformer2",
                    "instruction": "根据梗概写故事",
                    "source_tag": "梗概：",
                    "output_tag": "故事："
                },
                {
                    "from": "gpt",
                    "source": "spliter2"
                }
            ],
            "system": "You are a story writer with {琼明神女录} style."
        },
        "output_path": "data/datasets/qiongming_v2.json"
    }
}

with open('databuilder_args/v0.0.3example.json', 'w', encoding='utf-8') as f:
    json.dump(args_example, f, ensure_ascii=False, indent=4)
