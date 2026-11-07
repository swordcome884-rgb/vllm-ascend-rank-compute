#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import os
import sys

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from vllm_ascend import envs


class _NpuAll2AllManager:
    """No-op all2all_manager for NPU. Used by vLLM main's fault-tolerance
    check (data_parallel_size > 1 and is_moe); NPU does not register a real
    one because it uses mc2 / all_gather for MoE communication.
    """

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            use_all2all=use_all2all,
        )
        self.device = torch.npu.current_device()
        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        self.all2all_manager = _NpuAll2AllManager()
        self._experimental_ar = None
        self._custom_ar_active_logged = False
        self._custom_ar_enabled = (
            envs.VLLM_ASCEND_ENABLE_CUSTOM_AR
            and self.unique_name.startswith("tp:")
            and dist.get_world_size(self.cpu_group) in (8, 16)
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self._custom_ar_enabled:
            if self._experimental_ar is None:
                sys.path.insert(0, envs.VLLM_ASCEND_CUSTOM_AR_PATH)
                from custom_ar import AscendCustomAllreduce

                rank = dist.get_rank(self.cpu_group)
                world = dist.get_world_size(self.cpu_group)
                self._experimental_ar = AscendCustomAllreduce(
                    rank, world, self.device, envs.VLLM_ASCEND_CUSTOM_AR_STORE_URL
                )
                print(
                    f"IQUEST_CUSTOM_AR_READY rank={rank} world={world} algorithm=two_shot_batched_flags",
                    flush=True,
                )
            if (
                input_.device.type == "npu"
                and input_.dtype == torch.bfloat16
                and input_.is_contiguous()
                and 0 < input_.numel() * 2 <= 2 * 1024 * 1024
                and input_.numel() % 16 == 0
            ):
                if not self._custom_ar_active_logged:
                    print(
                        f"IQUEST_CUSTOM_AR_ACTIVE rank={dist.get_rank(self.cpu_group)} "
                        f"bytes={input_.numel() * 2} algorithm=two_shot_batched_flags",
                        flush=True,
                    )
                    self._custom_ar_active_logged = True
                return self._experimental_ar.all_reduce(input_, algorithm="two_shot_batched_flags")
        return super().all_reduce(input_)
