# Copyright (c) 2022 Agora
# Licensed under The MIT License [see LICENSE for details]
from zeta.utils.benchmark import (
    benchmark,
    print_cuda_memory_usage,
    save_memory_snapshot,
)
from zeta.utils.cuda_memory_wrapper import track_cuda_memory_usage

#######
from zeta.utils.cuda_wrapper import (
    append_nvcc_threads,
    check_cuda,
    check_cuda_torch_binary_vs_bare_metal,
    get_cuda_bare_metal_version,
    raise_if_cuda_home_none,
)
from zeta.utils.disable_logging import disable_warnings_and_logs
from zeta.utils.enforce_types import enforce_types
from zeta.utils.main import (
    cast_if_src_dtype,
    cast_tuple,
    cosine_beta_schedule,
    default,
    eval_decorator,
    exists,
    get_sinusoid_encoding_table,
    gif_to_tensor,
    group_by_key_prefix,
    group_dict_by_key,
    gumbel_noise,
    init_zero_,
    interpolate_pos_encoding_2d,
    l2norm,
    log,
    maybe,
    once,
    pad_at_dim,
    pick_and_pop,
    string_begins_with,
    top_a,
    top_k,
    top_p,
    video_tensor_to_gift,
)
from zeta.utils.module_device import module_device
from zeta.utils.params import print_main, print_num_params
from zeta.utils.save_load_wrapper import save_load
from zeta.utils.verbose_execution import VerboseExecution

####
__all__ = [
    "track_cuda_memory_usage",
    "benchmark",
    "print_cuda_memory_usage",
    "save_memory_snapshot",
    "disable_warnings_and_logs",
    "print_main",
    "module_device",
    "save_load",
    "exists",
    "default",
    "once",
    "eval_decorator",
    "cast_tuple",
    "maybe",
    "init_zero_",
    "pick_and_pop",
    "group_dict_by_key",
    "string_begins_with",
    "group_by_key_prefix",
    "top_p",
    "top_k",
    "top_a",
    "log",
    "gumbel_noise",
    "print_num_params",
    "video_tensor_to_gift",
    "gif_to_tensor",
    "l2norm",
    "pad_at_dim",
    "cosine_beta_schedule",
    "cast_if_src_dtype",
    "get_sinusoid_encoding_table",
    "interpolate_pos_encoding_2d",
    "enforce_types",
    "get_cuda_bare_metal_version",
    "check_cuda_torch_binary_vs_bare_metal",
    "raise_if_cuda_home_none",
    "append_nvcc_threads",
    "check_cuda",
    "VerboseExecution",
]
