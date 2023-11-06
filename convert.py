import sys

import torch
import gguf
import numpy

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


def make_structs(o):
    structs = AutoVivification()
    for k in o.keys():
        if type(o[k]) is dict and all(map(lambda x: x.isnumeric(), o[k].keys())):
            arr_length = int(max(o[k].keys())) + 1
            structs["main"][k] = f"{k.capitalize()}[{arr_length}]"
            structs[k.capitalize()] = make_structs(o[k]["0"])["main"]
        else:
            if type(o[k]) is not dict:
                structs["main"][k] = type(o[k])
            else:
                structs["main"][k] = k.capitalize()
                structs[k.capitalize()] = make_structs(o[k])["main"]

    return structs


def c_gen(state_dict, model_name):
    struct_members = {k: 'const ggml_tensor*;' for k in state_dict.keys()}
    struct_members_str = "\n    ".join(map(lambda x: f"{x[0]}: {x[1]}", struct_members.items()))

    # gen model AST
    model = {}
    for k, v in state_dict.items():
        parts = k.split(".")

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

    model_structs = make_structs(model)
    print()
    for struct_name, struct_fields in model_structs.items():
        struct_fields_str = "\n    ".join(map(lambda x: f"{x[0]}: {x[1]};", struct_fields.items()))
        print(STRUCT_TEMPLATE.format(struct_name=struct_name, struct_fields=struct_fields_str))

    # def obj_to_struct(o, name):
    #
    #     if o is list:
    #
    #
    #     code = STRUCT_TEMPLATE.format(struct_name=name, _struct_members_str)

    code = STRUCT_TEMPLATE.format(struct_name=model_name, struct_fields=struct_members_str)

    return code


def convert(model_path, output_model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    gguf_writer = gguf.GGUFWriter(output_model_path, output_model_path)

    code = c_gen(state_dict, "mymodel")
    print(code)
    return

    for param_name, param_value in state_dict.items():
        print(param_name)
        print(type(param_value.numpy()))
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
    convert(sys.argv[1], sys.argv[2])
