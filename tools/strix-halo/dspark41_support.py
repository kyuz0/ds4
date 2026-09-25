#!/usr/bin/env python3
"""Build the DeepSeek V4.1 DSpark support GGUF with the gguf-tools of antirez/ds4#1073,
from the official checkpoint's mtp.* shards only (44-46 of 48, about 8 GB).

deepseek41_quantize.py reads tokenizer metadata for the main model; the DSpark
stages do not need it, so this wrapper skips it and only allows --dspark-out.

usage: dspark41_support.py --gguf-tools ../ds4-pr1073/gguf-tools --hf DIR --dspark-out OUT.gguf [other deepseek41_quantize.py options]
DIR holds config.json, model.safetensors.index.json and model-0004{4,5,6}-of-00048.safetensors.
"""
import json, os, sys


def main() -> None:
    args = sys.argv[1:]
    if '--gguf-tools' not in args:
        raise SystemExit(__doc__)
    i = args.index('--gguf-tools')
    tools = os.path.abspath(args[i + 1])
    del args[i:i + 2]
    if '--out' in args or '--dspark-out' not in args:
        raise SystemExit("only --dspark-out is supported")
    sys.path.insert(0, tools)
    import deepseek41_quantize as q

    def metadata_no_tokenizer(hf_dir, revision):
        with open(os.path.join(hf_dir, "config.json")) as fp:
            config = json.load(fp)
        if config["model_type"] != "deepseek_v41":
            raise ValueError("not a DeepSeek V4.1 source checkpoint")
        return config, []

    q.metadata = metadata_no_tokenizer
    sys.argv = [os.path.join(tools, 'deepseek41_quantize.py')] + args
    q.main()


if __name__ == '__main__':
    main()
