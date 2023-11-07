#!/usr/bin/env python

import sys
import gguf
import torch
import numpy as np
from typing import Dict

STRUCT_TEMPLATE = """struct {struct_name} {{
    {struct_fields}
}}"""


# https://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


DEFAULT_MODEL_NAME = "untitled"
GGML_TENSOR_PTR = "struct ggml_tensor*"
MAIN_STRUCT: str = "module"
TAB_WIDTH = 4
TAB = " " * TAB_WIDTH


def make_structs(o: Dict) -> Dict[str, str]:
    """
    :param o: the model
    :return: the structs used to represent the model in c
    """

    structs = AutoVivification()

    def get_module_add_structs(struct_name_to_struct_fields):
        _main_struct = None
        for struct_name, struct_fields in struct_name_to_struct_fields.items():
            if struct_name == MAIN_STRUCT:
                _main_struct = struct_fields
                continue
            structs[struct_name] = struct_fields  # FIXME: override is bad
        return _main_struct

    for k in o.keys():
        if type(o[k]) is dict:
            if all(map(lambda x: x.isnumeric(), o[k].keys())):
                arr_length = int(max(o[k].keys())) + 1
                structs[MAIN_STRUCT][k] = f"struct {k.capitalize()}[{arr_length}]"
                structs[k.capitalize()] = get_module_add_structs(make_structs(o[k]["0"]))
            else:
                structs[MAIN_STRUCT][k] = f"struct {k.capitalize()}"
                structs[k.capitalize()] = get_module_add_structs(make_structs(o[k]))
        else:
            structs[MAIN_STRUCT][k] = type(o[k])

    return structs


def c_gen(state_dict, model_name):
    code = '#include <stdio.h>\n#include <stdlib.h>\n#include "ggml/ggml.h"\n\n\n'

    def k(line, tabs=0):
        nonlocal code
        code += f"{TAB * tabs}{line}\n"

    # gen model AST
    model = {}
    for param_name, v in state_dict.items():
        parts = param_name.split(".")

        last = model
        for i in range(len(parts)):
            p = parts[i]
            if p not in last:
                last[p] = {}
            # last p
            if i == len(parts) - 1:
                last[p] = v.numpy()
            else:
                last = last[p]

    # gen model structs
    model_structs = make_structs(model)
    model_structs[MAIN_STRUCT]["ctx"] = "struct ggml_context*"
    for struct_name, struct_fields in model_structs.items():
        if struct_name == MAIN_STRUCT:
            struct_name = model_name

        struct_fields = {k: (GGML_TENSOR_PTR if type is np.ndarray else type) for k, type in struct_fields.items()}

        struct_fields_str = "\n    ".join(map(lambda x: f"{x[1]} {x[0]};", struct_fields.items()))
        code += STRUCT_TEMPLATE.format(struct_name=struct_name, struct_fields=struct_fields_str)
        code += "\n\n"

    # gen model load function
    code += f"""
// returns a pointer to a loaded model struct. user is responsible of freeing it later
struct {model_name}* {model_name}_model_load(const char *model_file, mnist_model & model) {{
    struct {model_name} *model = malloc(sizeof(*model));
    if (!model) {{
        fprintf(stderr, "%s: malloc(sizeof(*model)) failed, Out of memory\\n", __func__);
        return NULL;
    }}

    struct gguf_init_params params = {{
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    }};
    gguf_context *ctx = gguf_init_from_file(model_file, params);
    if (!ctx) {{
        fprintf(stderr, "%s: gguf_init_from_file() failed\\n", __func__);
        return NULL;
    }}
"""

    for param_name in state_dict.keys():
        param_name_to_accessor = ""
        for p in param_name.split("."):
            if p.isnumeric():
                param_name_to_accessor += f"[{p}]"
            else:
                param_name_to_accessor += f".{p}"

        k(f'model{param_name_to_accessor} = ggml_get_tensor(model.ctx, "{param_name}");', tabs=1)

    k("")
    k("return model;\n}", tabs=1)

    return code


def convert(model_path, output_model_path, model_name=DEFAULT_MODEL_NAME):
    if model_name == DEFAULT_MODEL_NAME:
        print(f"Warning: no provided model_name, default={DEFAULT_MODEL_NAME}")

    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    code = c_gen(state_dict, model_name)
    c_output_path = f"./{model_name}.c"
    with open(c_output_path, "w") as f:
        f.write(code)
        print(f"Model ggml code generated and saved to '{c_output_path}'")

    gguf_writer = gguf.GGUFWriter(output_model_path, model_name)
    for param_name, param_value in state_dict.items():
        gguf_writer.add_tensor(param_name, param_value.numpy())

    # kernel1 = model.layers[0].weights[0].numpy()
    # kernel1 = np.moveaxis(kernel1, [2, 3], [0, 1])
    # kernel1 = kernel1.astype(np.float16)
    # gguf_writer.add_tensor("kernel1", kernel1, raw_shape=(32, 1, 3, 3))
    #
    # bias1 = model.layers[0].weights[1].numpy()
    # bias1 = np.repeat(bias1, 26 * 26)
    # gguf_writer.add_tensor("bias1", bias1, raw_shape=(1, 32, 26, 26))
    #
    # kernel2 = model.layers[2].weights[0].numpy()
    # kernel2 = np.moveaxis(kernel2, [0, 1, 2, 3], [2, 3, 1, 0])
    # kernel2 = kernel2.astype(np.float16)
    # gguf_writer.add_tensor("kernel2", kernel2, raw_shape=(64, 32, 3, 3))
    #
    # bias2 = model.layers[2].weights[1].numpy()
    # bias2 = np.repeat(bias2, 11 * 11)
    # gguf_writer.add_tensor("bias2", bias2, raw_shape=(1, 64, 11, 11))
    #
    # dense_w = model.layers[-1].weights[0].numpy()
    # dense_w = dense_w.transpose()
    # gguf_writer.add_tensor("dense_w", dense_w, raw_shape=(10, 1600))
    #
    # dense_b = model.layers[-1].weights[1].numpy()
    # gguf_writer.add_tensor("dense_b", dense_b)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Model converted and saved to '{output_model_path}'")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: ./{sys.argv[0]} <path_to_pt_model> <path_to_output> [model_name]")
        sys.exit(1)

    if len(sys.argv) < 4:
        convert(sys.argv[1], sys.argv[2])
    else:
        convert(sys.argv[1], sys.argv[2], sys.argv[3])
