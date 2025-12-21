from mmengine.config import read_base

with read_base():
    from .mmtu_gen_b2f12e import (  # noqa: F401
        mmtu_datasets,
        mmtu_summary_groups,
    )
