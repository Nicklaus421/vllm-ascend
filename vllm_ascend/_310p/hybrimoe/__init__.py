#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""HybriMoE: hybrid CPU-NPU MoE inference for Ascend 310P.

Implements the HybriMoE algorithm (https://arxiv.org/abs/2504.05897):
  1. HSS  - dynamic intra-layer hybrid CPU/NPU scheduling of experts
  2. impact-driven inter-layer expert prefetching
  3. MRS  - score-aware (Minus Recent Score) expert cache management
"""
