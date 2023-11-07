# :fire: Torch to ggml

a python tool to convert any (hopefully) pytorch model file to a gguf file 
and generate as much of the c code to use it as possible.

## Usage

```sh
./convert.py <path_to_pt_model> [model_name]
```

This will generate a model_name.gguf model file and a model_name.c file.
