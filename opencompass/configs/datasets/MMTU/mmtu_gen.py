from mmengine.config import read_base

with read_base():
    from .mmtu_gen_a3b5c7 import (  # noqa: F401
        mmtu_datasets,
        mmtu_summary_groups,
    )
