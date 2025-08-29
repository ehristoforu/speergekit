import click
import yaml

from speergekit.config import MergeConfiguration
from speergekit.merge import run_merge
from speergekit.options import MergeOptions


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
@click.option("--cuda", is_flag=True, help="Use CUDA for merging")
@click.option("--lazy-unpickle", is_flag=True, help="Enable lazy unpickling")
@click.option("--low-cpu-memory", is_flag=True, help="Enable low CPU memory mode")
def main(
    config_path: str,
    out_path: str,
    cuda: bool,
    lazy_unpickle: bool,
    low_cpu_memory: bool,
):
    """
    Merge PyTorch models for text-to-speech (TTS) based on a YAML configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    merge_config = MergeConfiguration.model_validate(config_data)

    options = MergeOptions(
        cuda=cuda,
        lazy_unpickle=lazy_unpickle,
        low_cpu_memory=low_cpu_memory,
    )

    run_merge(merge_config, out_path, options)


if __name__ == "__main__":
    main()