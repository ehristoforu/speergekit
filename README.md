# speergekit

This is a library for merging pretrained speech models (TTS & ASR). It is a refactor of the original `speergekit` with a more powerful and flexible YAML-based configuration system, inspired by [`mergekit`](https://github.com/arcee-ai/mergekit).

## Installation

To install the library, clone the repository and install it in editable mode:

```bash
git clone https://github.com/ehristoforu/speergekit.git
cd speergekit
pip install -e .
```

## Usage

To perform a merge, you need to create a YAML configuration file that specifies the models to merge, the merge method, and other parameters. Then, you can run the merge script:

```bash
speergekit path/to/your/config.yml ./output-directory --cuda
```

This will merge the models according to your configuration and save the resulting model in the `./output-directory`.

## Merge Configuration

The YAML configuration file has the following structure:

-   `models`: A list of models to merge. Each model can be specified by a local path or a HuggingFace repository ID (e.g., `openai/whisper-tiny`).
-   `base_model`: The model to use as the base for the merge.
-   `merge_method`: The merge method to use. Currently, `linear` and `task_arithmetic` are supported.
-   `parameters`: A dictionary of parameters for the merge method.
-   `dtype`: The data type to use for the merge. `float16` is recommended.

### Example Configurations

You can find example configurations in the [`examples/`](./examples) directory:

-   [`linear.yml`](./examples/linear.yml): A simple weighted average of two models.
-   [`task_arithmetic.yml`](./examples/task_arithmetic.yml): A task arithmetic merge. This method applies the formula `(model_1 - base_model) * scaling_factor + model_2`. The `models` list must contain exactly two models: the finetuned model first, and the target model second.
