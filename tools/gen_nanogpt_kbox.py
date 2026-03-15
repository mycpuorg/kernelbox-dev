#!/usr/bin/env python3
"""Generate 32 kernelbox test files + h5 fixtures for the nanoGPT forward pass.

Usage:
    uv run python3 tools/gen_nanogpt_kbox.py [--output-dir DIR]
"""

import argparse
import os
import textwrap
from collections import Counter

import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
B = 2
T = 16
C = 32
N_HEAD = 2
HEAD_DIM = 16
VOCAB_SIZE = 64
MLP_HIDDEN = 128


# ---------------------------------------------------------------------------
# Primals generation
# ---------------------------------------------------------------------------
def generate_primals(device):
    """Create all model weight tensors (primals_1 .. primals_31)."""
    torch.manual_seed(42)
    p = {}
    p['primals_1'] = torch.randint(0, VOCAB_SIZE, (B, T), dtype=torch.int64, device=device)
    p['primals_2'] = torch.randn(VOCAB_SIZE, C, device=device)        # wte
    p['primals_3'] = torch.randn(T, C, device=device)                 # wpe
    # Block 0
    p['primals_4'] = torch.ones(C, device=device)                     # ln_1 weight
    p['primals_5'] = torch.zeros(C, device=device)                    # ln_1 bias
    p['primals_6'] = torch.randn(3 * C, C, device=device)             # c_attn weight
    p['primals_7'] = torch.randn(3 * C, device=device)                # c_attn bias
    p['primals_8'] = torch.tril(torch.ones(1, 1, T, T, device=device))  # causal mask
    p['primals_9'] = torch.randn(C, C, device=device)                 # c_proj weight
    p['primals_10'] = torch.randn(C, device=device)                   # c_proj bias
    p['primals_11'] = torch.ones(C, device=device)                    # ln_2 weight
    p['primals_12'] = torch.zeros(C, device=device)                   # ln_2 bias
    p['primals_13'] = torch.randn(MLP_HIDDEN, C, device=device)       # fc1 weight
    p['primals_14'] = torch.randn(MLP_HIDDEN, device=device)          # fc1 bias
    p['primals_15'] = torch.randn(C, MLP_HIDDEN, device=device)       # fc2 weight
    p['primals_16'] = torch.randn(C, device=device)                   # fc2 bias
    # Block 1
    p['primals_17'] = torch.ones(C, device=device)                    # ln_1 weight
    p['primals_18'] = torch.zeros(C, device=device)                   # ln_1 bias
    p['primals_19'] = torch.randn(3 * C, C, device=device)            # c_attn weight
    p['primals_20'] = torch.randn(3 * C, device=device)               # c_attn bias
    p['primals_21'] = torch.tril(torch.ones(1, 1, T, T, device=device))  # causal mask
    p['primals_22'] = torch.randn(C, C, device=device)                # c_proj weight
    p['primals_23'] = torch.randn(C, device=device)                   # c_proj bias
    p['primals_24'] = torch.ones(C, device=device)                    # ln_2 weight
    p['primals_25'] = torch.zeros(C, device=device)                   # ln_2 bias
    p['primals_26'] = torch.randn(MLP_HIDDEN, C, device=device)       # fc1 weight
    p['primals_27'] = torch.randn(MLP_HIDDEN, device=device)          # fc1 bias
    p['primals_28'] = torch.randn(C, MLP_HIDDEN, device=device)       # fc2 weight
    p['primals_29'] = torch.randn(C, device=device)                   # fc2 bias
    # Final LN
    p['primals_30'] = torch.ones(C, device=device)                    # ln_f weight
    p['primals_31'] = torch.zeros(C, device=device)                   # ln_f bias
    return p


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
def run_forward(p, device):
    """Execute the full nanoGPT forward pass using torch.ops.aten.* calls.

    Returns a dict of ALL named intermediate tensors.
    """
    t = {}

    # fw_000
    t['arange'] = torch.ops.aten.arange.start(0, T, dtype=torch.int64, device=device, pin_memory=False)

    # fw_001
    t['embedding'] = torch.ops.aten.embedding.default(p['primals_2'], p['primals_1'])

    # fw_002
    t['embedding_1'] = torch.ops.aten.embedding.default(p['primals_3'], t['arange'])

    # fw_003
    t['add'] = torch.ops.aten.add.Tensor(t['embedding'], t['embedding_1'])

    # fw_004 LN part
    ln_0 = torch.ops.aten.native_layer_norm.default(t['add'], [C], p['primals_4'], p['primals_5'], 1e-05)
    t['getitem'] = ln_0[0]
    t['getitem_1'] = ln_0[1]
    t['getitem_2'] = ln_0[2]

    # fw_005
    view = torch.ops.aten.view.default(t['getitem'], [B * T, C])
    t_0 = torch.ops.aten.t.default(p['primals_6'])
    addmm = torch.ops.aten.addmm.default(p['primals_7'], view, t_0)
    t['view_1'] = torch.ops.aten.view.default(addmm, [B, T, 3 * C])

    # fw_006
    split = torch.ops.aten.split.Tensor(t['view_1'], C, 2)
    t['getitem_3'] = split[0]
    t['getitem_4'] = split[1]
    t['getitem_5'] = split[2]

    # fw_007
    view_2 = torch.ops.aten.view.default(t['getitem_3'], [B, T, N_HEAD, HEAD_DIM])
    t['transpose'] = torch.ops.aten.transpose.int(view_2, 1, 2)

    # fw_008
    view_3 = torch.ops.aten.view.default(t['getitem_4'], [B, T, N_HEAD, HEAD_DIM])
    t['transpose_1'] = torch.ops.aten.transpose.int(view_3, 1, 2)

    # fw_009
    view_4 = torch.ops.aten.view.default(t['getitem_5'], [B, T, N_HEAD, HEAD_DIM])
    t['transpose_2'] = torch.ops.aten.transpose.int(view_4, 1, 2)

    # fw_010
    transpose_3 = torch.ops.aten.transpose.int(t['transpose_1'], -2, -1)
    expand = torch.ops.aten.expand.default(t['transpose'], [B, N_HEAD, T, HEAD_DIM])
    expand_1 = torch.ops.aten.expand.default(transpose_3, [B, N_HEAD, HEAD_DIM, T])
    clone = torch.ops.aten.clone.default(expand, memory_format=torch.contiguous_format)
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format=torch.contiguous_format)
    uv0 = torch.ops.aten._unsafe_view.default(clone, [B * N_HEAD, T, HEAD_DIM])
    uv1 = torch.ops.aten._unsafe_view.default(clone_1, [B * N_HEAD, HEAD_DIM, T])
    bmm = torch.ops.aten.bmm.default(uv0, uv1)
    view_5 = torch.ops.aten.view.default(bmm, [B, N_HEAD, T, T])
    t['mul'] = torch.ops.aten.mul.Tensor(view_5, 0.25)

    # fw_011
    alias = torch.ops.aten.alias.default(p['primals_8'])
    eq = torch.ops.aten.eq.Scalar(alias, 0)
    t['masked_fill'] = torch.ops.aten.masked_fill.Scalar(t['mul'], eq, float('-inf'))

    # fw_012
    t['_softmax'] = torch.ops.aten._softmax.default(t['masked_fill'], -1, False)
    t['detach'] = torch.ops.aten.detach.default(t['_softmax'])

    # fw_013
    expand_2 = torch.ops.aten.expand.default(t['_softmax'], [B, N_HEAD, T, T])
    expand_3 = torch.ops.aten.expand.default(t['transpose_2'], [B, N_HEAD, T, HEAD_DIM])
    view_6 = torch.ops.aten.view.default(expand_2, [B * N_HEAD, T, T])
    clone_2 = torch.ops.aten.clone.default(expand_3, memory_format=torch.contiguous_format)
    uv2 = torch.ops.aten._unsafe_view.default(clone_2, [B * N_HEAD, T, HEAD_DIM])
    bmm_1 = torch.ops.aten.bmm.default(view_6, uv2)
    t['view_7'] = torch.ops.aten.view.default(bmm_1, [B, N_HEAD, T, HEAD_DIM])

    # fw_014
    transpose_4 = torch.ops.aten.transpose.int(t['view_7'], 1, 2)
    clone_3 = torch.ops.aten.clone.default(transpose_4, memory_format=torch.contiguous_format)
    t['view_8'] = torch.ops.aten.view.default(clone_3, [B, T, C])

    # fw_015
    view_9 = torch.ops.aten.view.default(t['view_8'], [B * T, C])
    t_1 = torch.ops.aten.t.default(p['primals_9'])
    addmm_1 = torch.ops.aten.addmm.default(p['primals_10'], view_9, t_1)
    t['view_10'] = torch.ops.aten.view.default(addmm_1, [B, T, C])

    # fw_004 residual
    t['add_1'] = torch.ops.aten.add.Tensor(t['add'], t['view_10'])

    # fw_016 (MLP block 0)
    ln_1 = torch.ops.aten.native_layer_norm.default(t['add_1'], [C], p['primals_11'], p['primals_12'], 1e-05)
    getitem_6 = ln_1[0]
    view_11 = torch.ops.aten.view.default(getitem_6, [B * T, C])
    t_2 = torch.ops.aten.t.default(p['primals_13'])
    addmm_2 = torch.ops.aten.addmm.default(p['primals_14'], view_11, t_2)
    view_12 = torch.ops.aten.view.default(addmm_2, [B, T, MLP_HIDDEN])
    gelu = torch.ops.aten.gelu.default(view_12)
    view_13 = torch.ops.aten.view.default(gelu, [B * T, MLP_HIDDEN])
    t_3 = torch.ops.aten.t.default(p['primals_15'])
    addmm_3 = torch.ops.aten.addmm.default(p['primals_16'], view_13, t_3)
    view_14 = torch.ops.aten.view.default(addmm_3, [B, T, C])
    t['add_2'] = torch.ops.aten.add.Tensor(t['add_1'], view_14)

    # fw_017 LN part
    ln_2 = torch.ops.aten.native_layer_norm.default(t['add_2'], [C], p['primals_17'], p['primals_18'], 1e-05)
    t['getitem_9'] = ln_2[0]
    t['getitem_10'] = ln_2[1]
    t['getitem_11'] = ln_2[2]

    # fw_018
    view_15 = torch.ops.aten.view.default(t['getitem_9'], [B * T, C])
    t_4 = torch.ops.aten.t.default(p['primals_19'])
    addmm_4 = torch.ops.aten.addmm.default(p['primals_20'], view_15, t_4)
    t['view_16'] = torch.ops.aten.view.default(addmm_4, [B, T, 3 * C])

    # fw_019
    split_1 = torch.ops.aten.split.Tensor(t['view_16'], C, 2)
    t['getitem_12'] = split_1[0]
    t['getitem_13'] = split_1[1]
    t['getitem_14'] = split_1[2]

    # fw_020
    view_17 = torch.ops.aten.view.default(t['getitem_12'], [B, T, N_HEAD, HEAD_DIM])
    t['transpose_5'] = torch.ops.aten.transpose.int(view_17, 1, 2)

    # fw_021
    view_18 = torch.ops.aten.view.default(t['getitem_13'], [B, T, N_HEAD, HEAD_DIM])
    t['transpose_6'] = torch.ops.aten.transpose.int(view_18, 1, 2)

    # fw_022
    view_19 = torch.ops.aten.view.default(t['getitem_14'], [B, T, N_HEAD, HEAD_DIM])
    t['transpose_7'] = torch.ops.aten.transpose.int(view_19, 1, 2)

    # fw_023
    transpose_8 = torch.ops.aten.transpose.int(t['transpose_6'], -2, -1)
    expand_4 = torch.ops.aten.expand.default(t['transpose_5'], [B, N_HEAD, T, HEAD_DIM])
    expand_5 = torch.ops.aten.expand.default(transpose_8, [B, N_HEAD, HEAD_DIM, T])
    clone_4 = torch.ops.aten.clone.default(expand_4, memory_format=torch.contiguous_format)
    clone_5 = torch.ops.aten.clone.default(expand_5, memory_format=torch.contiguous_format)
    uv3 = torch.ops.aten._unsafe_view.default(clone_4, [B * N_HEAD, T, HEAD_DIM])
    uv4 = torch.ops.aten._unsafe_view.default(clone_5, [B * N_HEAD, HEAD_DIM, T])
    bmm_2 = torch.ops.aten.bmm.default(uv3, uv4)
    view_20 = torch.ops.aten.view.default(bmm_2, [B, N_HEAD, T, T])
    t['mul_1'] = torch.ops.aten.mul.Tensor(view_20, 0.25)

    # fw_024
    alias_1 = torch.ops.aten.alias.default(p['primals_21'])
    eq_1 = torch.ops.aten.eq.Scalar(alias_1, 0)
    t['masked_fill_1'] = torch.ops.aten.masked_fill.Scalar(t['mul_1'], eq_1, float('-inf'))

    # fw_025
    t['_softmax_1'] = torch.ops.aten._softmax.default(t['masked_fill_1'], -1, False)
    t['detach_1'] = torch.ops.aten.detach.default(t['_softmax_1'])

    # fw_026
    expand_6 = torch.ops.aten.expand.default(t['_softmax_1'], [B, N_HEAD, T, T])
    expand_7 = torch.ops.aten.expand.default(t['transpose_7'], [B, N_HEAD, T, HEAD_DIM])
    view_21 = torch.ops.aten.view.default(expand_6, [B * N_HEAD, T, T])
    clone_6 = torch.ops.aten.clone.default(expand_7, memory_format=torch.contiguous_format)
    uv5 = torch.ops.aten._unsafe_view.default(clone_6, [B * N_HEAD, T, HEAD_DIM])
    bmm_3 = torch.ops.aten.bmm.default(view_21, uv5)
    t['view_22'] = torch.ops.aten.view.default(bmm_3, [B, N_HEAD, T, HEAD_DIM])

    # fw_027
    transpose_9 = torch.ops.aten.transpose.int(t['view_22'], 1, 2)
    clone_7 = torch.ops.aten.clone.default(transpose_9, memory_format=torch.contiguous_format)
    t['view_23'] = torch.ops.aten.view.default(clone_7, [B, T, C])

    # fw_028
    view_24 = torch.ops.aten.view.default(t['view_23'], [B * T, C])
    t_5 = torch.ops.aten.t.default(p['primals_22'])
    addmm_5 = torch.ops.aten.addmm.default(p['primals_23'], view_24, t_5)
    t['view_25'] = torch.ops.aten.view.default(addmm_5, [B, T, C])

    # fw_017 residual
    t['add_3'] = torch.ops.aten.add.Tensor(t['add_2'], t['view_25'])

    # fw_029 (MLP block 1)
    ln_3 = torch.ops.aten.native_layer_norm.default(t['add_3'], [C], p['primals_24'], p['primals_25'], 1e-05)
    getitem_15 = ln_3[0]
    view_26 = torch.ops.aten.view.default(getitem_15, [B * T, C])
    t_6 = torch.ops.aten.t.default(p['primals_26'])
    addmm_6 = torch.ops.aten.addmm.default(p['primals_27'], view_26, t_6)
    view_27 = torch.ops.aten.view.default(addmm_6, [B, T, MLP_HIDDEN])
    gelu_1 = torch.ops.aten.gelu.default(view_27)
    view_28 = torch.ops.aten.view.default(gelu_1, [B * T, MLP_HIDDEN])
    t_7 = torch.ops.aten.t.default(p['primals_28'])
    addmm_7 = torch.ops.aten.addmm.default(p['primals_29'], view_28, t_7)
    view_29 = torch.ops.aten.view.default(addmm_7, [B, T, C])
    t['add_4'] = torch.ops.aten.add.Tensor(t['add_3'], view_29)

    # fw_030
    ln_f = torch.ops.aten.native_layer_norm.default(t['add_4'], [C], p['primals_30'], p['primals_31'], 1e-05)
    t['getitem_18'] = ln_f[0]
    t['getitem_19'] = ln_f[1]
    t['getitem_20'] = ln_f[2]

    # fw_031
    t_8 = torch.ops.aten.t.default(p['primals_2'])
    view_30 = torch.ops.aten.view.default(t['getitem_18'], [B * T, C])
    mm = torch.ops.aten.mm.default(view_30, t_8)
    t['_unsafe_view_6'] = torch.ops.aten._unsafe_view.default(mm, [B, T, VOCAB_SIZE])

    return t


# ---------------------------------------------------------------------------
# OPS list  (32 entries)
# ---------------------------------------------------------------------------
OPS = [
    {
        'id': 'fw_000', 'name': 'arange', 'line': 'model.py:67', 'module': '',
        'desc': 'arange',
        'input_keys': [], 'output_keys': ['arange'],
        'needs_operator': False,
        'ops_desc': 'arange x1', 'n_ops': 1,
        'run_body': textwrap.dedent("""\
            arange = torch.ops.aten.arange.start(0, 16, dtype=torch.int64, device="cuda", pin_memory=False)
            return [arange]"""),
    },
    {
        'id': 'fw_001', 'name': 'embedding', 'line': 'model.py:68', 'module': '[wte]',
        'desc': 'embedding',
        'input_keys': ['primals_2', 'primals_1'], 'output_keys': ['embedding'],
        'needs_operator': False,
        'ops_desc': 'embedding x1', 'n_ops': 1,
        'run_body': textwrap.dedent("""\
            embedding = torch.ops.aten.embedding.default(inputs.primals_2, inputs.primals_1)
            return [embedding]"""),
    },
    {
        'id': 'fw_002', 'name': 'embedding', 'line': 'model.py:68', 'module': '[wpe]',
        'desc': 'embedding',
        'input_keys': ['primals_3', 'arange'], 'output_keys': ['embedding_1'],
        'needs_operator': False,
        'ops_desc': 'embedding x1', 'n_ops': 1,
        'run_body': textwrap.dedent("""\
            embedding_1 = torch.ops.aten.embedding.default(inputs.primals_3, inputs.arange)
            return [embedding_1]"""),
    },
    {
        'id': 'fw_003', 'name': 'add', 'line': 'model.py:68', 'module': '',
        'desc': 'add',
        'input_keys': ['embedding', 'embedding_1'], 'output_keys': ['add'],
        'needs_operator': False,
        'ops_desc': 'add x1', 'n_ops': 1,
        'run_body': textwrap.dedent("""\
            add = torch.ops.aten.add.Tensor(inputs.embedding, inputs.embedding_1)
            return [add]"""),
    },
    {
        'id': 'fw_004', 'name': 'add', 'line': 'model.py:48', 'module': '[blocks.0]',
        'desc': 'add (ln_1 residual)',
        'input_keys': ['add', 'primals_4', 'primals_5', 'view_10'], 'output_keys': ['add_1'],
        'needs_operator': True,
        'ops_desc': 'native_layer_norm x1, getitem x3, add x1', 'n_ops': 5,
        'run_body': textwrap.dedent("""\
            native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add, [32], inputs.primals_4, inputs.primals_5, 1e-05)
            getitem = operator.getitem(native_layer_norm, 0)
            getitem_1 = operator.getitem(native_layer_norm, 1)
            getitem_2 = operator.getitem(native_layer_norm, 2)
            add_1 = torch.ops.aten.add.Tensor(inputs.add, inputs.view_10)
            return [add_1]"""),
    },
    {
        'id': 'fw_005', 'name': 'view', 'line': 'model.py:22', 'module': '[blocks.0]',
        'desc': 'view (qkv linear)',
        'input_keys': ['getitem', 'primals_6', 'primals_7'], 'output_keys': ['view_1'],
        'needs_operator': False,
        'ops_desc': 'view x2, t x1, addmm x1', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            view = torch.ops.aten.view.default(inputs.getitem, [32, 32])
            t = torch.ops.aten.t.default(inputs.primals_6)
            addmm = torch.ops.aten.addmm.default(inputs.primals_7, view, t)
            view_1 = torch.ops.aten.view.default(addmm, [2, 16, 96])
            return [view_1]"""),
    },
    {
        'id': 'fw_006', 'name': 'split', 'line': 'model.py:23', 'module': '[blocks.0]',
        'desc': 'split (q,k,v)',
        'input_keys': ['view_1'], 'output_keys': ['getitem_3', 'getitem_4', 'getitem_5'],
        'needs_operator': True,
        'ops_desc': 'split x1, getitem x3', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            split = torch.ops.aten.split.Tensor(inputs.view_1, 32, 2)
            getitem_3 = operator.getitem(split, 0)
            getitem_4 = operator.getitem(split, 1)
            getitem_5 = operator.getitem(split, 2)
            return [getitem_3, getitem_4, getitem_5]"""),
    },
    {
        'id': 'fw_007', 'name': 'transpose', 'line': 'model.py:24', 'module': '[blocks.0]',
        'desc': 'transpose (q reshape)',
        'input_keys': ['getitem_3'], 'output_keys': ['transpose'],
        'needs_operator': False,
        'ops_desc': 'view x1, transpose x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            view_2 = torch.ops.aten.view.default(inputs.getitem_3, [2, 16, 2, 16])
            transpose = torch.ops.aten.transpose.int(view_2, 1, 2)
            return [transpose]"""),
    },
    {
        'id': 'fw_008', 'name': 'transpose', 'line': 'model.py:25', 'module': '[blocks.0]',
        'desc': 'transpose (k reshape)',
        'input_keys': ['getitem_4'], 'output_keys': ['transpose_1'],
        'needs_operator': False,
        'ops_desc': 'view x1, transpose x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            view_3 = torch.ops.aten.view.default(inputs.getitem_4, [2, 16, 2, 16])
            transpose_1 = torch.ops.aten.transpose.int(view_3, 1, 2)
            return [transpose_1]"""),
    },
    {
        'id': 'fw_009', 'name': 'transpose', 'line': 'model.py:26', 'module': '[blocks.0]',
        'desc': 'transpose (v reshape)',
        'input_keys': ['getitem_5'], 'output_keys': ['transpose_2'],
        'needs_operator': False,
        'ops_desc': 'view x1, transpose x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            view_4 = torch.ops.aten.view.default(inputs.getitem_5, [2, 16, 2, 16])
            transpose_2 = torch.ops.aten.transpose.int(view_4, 1, 2)
            return [transpose_2]"""),
    },
    {
        'id': 'fw_010', 'name': 'mul', 'line': 'model.py:27', 'module': '[blocks.0]',
        'desc': 'mul (QK^T scaled)',
        'input_keys': ['transpose_1', 'transpose'], 'output_keys': ['mul'],
        'needs_operator': False,
        'ops_desc': 'transpose x1, expand x2, clone x2, _unsafe_view x2, bmm x1, view x1, mul x1', 'n_ops': 10,
        'run_body': textwrap.dedent("""\
            transpose_3 = torch.ops.aten.transpose.int(inputs.transpose_1, -2, -1)
            expand = torch.ops.aten.expand.default(inputs.transpose, [2, 2, 16, 16])
            expand_1 = torch.ops.aten.expand.default(transpose_3, [2, 2, 16, 16])
            clone = torch.ops.aten.clone.default(expand, memory_format=torch.contiguous_format)
            clone_1 = torch.ops.aten.clone.default(expand_1, memory_format=torch.contiguous_format)
            _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [4, 16, 16])
            _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_1, [4, 16, 16])
            bmm = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
            view_5 = torch.ops.aten.view.default(bmm, [2, 2, 16, 16])
            mul = torch.ops.aten.mul.Tensor(view_5, 0.25)
            return [mul]"""),
    },
    {
        'id': 'fw_011', 'name': 'masked_fill', 'line': 'model.py:28', 'module': '[blocks.0]',
        'desc': 'masked_fill (causal mask)',
        'input_keys': ['primals_8', 'mul'], 'output_keys': ['masked_fill'],
        'needs_operator': False,
        'ops_desc': 'alias x1, eq x1, masked_fill x1', 'n_ops': 3,
        'run_body': textwrap.dedent("""\
            alias = torch.ops.aten.alias.default(inputs.primals_8)
            eq = torch.ops.aten.eq.Scalar(alias, 0)
            masked_fill = torch.ops.aten.masked_fill.Scalar(inputs.mul, eq, float('-inf'))
            return [masked_fill]"""),
    },
    {
        'id': 'fw_012', 'name': 'detach', 'line': 'model.py:29', 'module': '[blocks.0]',
        'desc': 'detach (softmax)',
        'input_keys': ['masked_fill'], 'output_keys': ['detach'],
        'needs_operator': False,
        'ops_desc': '_softmax x1, detach x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            _softmax = torch.ops.aten._softmax.default(inputs.masked_fill, -1, False)
            detach = torch.ops.aten.detach.default(_softmax)
            return [detach]"""),
    },
    {
        'id': 'fw_013', 'name': 'view', 'line': 'model.py:30', 'module': '[blocks.0]',
        'desc': 'view (att @ v)',
        'input_keys': ['_softmax', 'transpose_2'], 'output_keys': ['view_7'],
        'needs_operator': False,
        'ops_desc': 'expand x2, view x1, clone x1, _unsafe_view x1, bmm x1, view x1', 'n_ops': 7,
        'run_body': textwrap.dedent("""\
            expand_2 = torch.ops.aten.expand.default(inputs._softmax, [2, 2, 16, 16])
            expand_3 = torch.ops.aten.expand.default(inputs.transpose_2, [2, 2, 16, 16])
            view_6 = torch.ops.aten.view.default(expand_2, [4, 16, 16])
            clone_2 = torch.ops.aten.clone.default(expand_3, memory_format=torch.contiguous_format)
            _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_2, [4, 16, 16])
            bmm_1 = torch.ops.aten.bmm.default(view_6, _unsafe_view_2)
            view_7 = torch.ops.aten.view.default(bmm_1, [2, 2, 16, 16])
            return [view_7]"""),
    },
    {
        'id': 'fw_014', 'name': 'view', 'line': 'model.py:31', 'module': '[blocks.0]',
        'desc': 'view (y reshape)',
        'input_keys': ['view_7'], 'output_keys': ['view_8'],
        'needs_operator': False,
        'ops_desc': 'transpose x1, clone x1, view x1', 'n_ops': 3,
        'run_body': textwrap.dedent("""\
            transpose_4 = torch.ops.aten.transpose.int(inputs.view_7, 1, 2)
            clone_3 = torch.ops.aten.clone.default(transpose_4, memory_format=torch.contiguous_format)
            view_8 = torch.ops.aten.view.default(clone_3, [2, 16, 32])
            return [view_8]"""),
    },
    {
        'id': 'fw_015', 'name': 'view', 'line': 'model.py:32', 'module': '[blocks.0]',
        'desc': 'view (c_proj)',
        'input_keys': ['view_8', 'primals_9', 'primals_10'], 'output_keys': ['view_10'],
        'needs_operator': False,
        'ops_desc': 'view x2, t x1, addmm x1', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            view_9 = torch.ops.aten.view.default(inputs.view_8, [32, 32])
            t_1 = torch.ops.aten.t.default(inputs.primals_9)
            addmm_1 = torch.ops.aten.addmm.default(inputs.primals_10, view_9, t_1)
            view_10 = torch.ops.aten.view.default(addmm_1, [2, 16, 32])
            return [view_10]"""),
    },
    {
        'id': 'fw_016', 'name': 'add', 'line': 'model.py:49', 'module': '[blocks.0]',
        'desc': 'add (MLP residual, ln_2)',
        'input_keys': ['add_1', 'primals_11', 'primals_12', 'primals_13', 'primals_14', 'primals_15', 'primals_16'],
        'output_keys': ['add_2'],
        'needs_operator': True,
        'ops_desc': 'native_layer_norm x1, getitem x3, view x4, t x2, addmm x2, gelu x1, add x1', 'n_ops': 14,
        'run_body': textwrap.dedent("""\
            native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_1, [32], inputs.primals_11, inputs.primals_12, 1e-05)
            getitem_6 = operator.getitem(native_layer_norm, 0)
            getitem_7 = operator.getitem(native_layer_norm, 1)
            getitem_8 = operator.getitem(native_layer_norm, 2)
            view_11 = torch.ops.aten.view.default(getitem_6, [32, 32])
            t_2 = torch.ops.aten.t.default(inputs.primals_13)
            addmm_2 = torch.ops.aten.addmm.default(inputs.primals_14, view_11, t_2)
            view_12 = torch.ops.aten.view.default(addmm_2, [2, 16, 128])
            gelu = torch.ops.aten.gelu.default(view_12)
            view_13 = torch.ops.aten.view.default(gelu, [32, 128])
            t_3 = torch.ops.aten.t.default(inputs.primals_15)
            addmm_3 = torch.ops.aten.addmm.default(inputs.primals_16, view_13, t_3)
            view_14 = torch.ops.aten.view.default(addmm_3, [2, 16, 32])
            add_2 = torch.ops.aten.add.Tensor(inputs.add_1, view_14)
            return [add_2]"""),
    },
    {
        'id': 'fw_017', 'name': 'add', 'line': 'model.py:48', 'module': '[blocks.1]',
        'desc': 'add (ln_1 residual)',
        'input_keys': ['add_2', 'primals_17', 'primals_18', 'view_25'], 'output_keys': ['add_3'],
        'needs_operator': True,
        'ops_desc': 'native_layer_norm x1, getitem x3, add x1', 'n_ops': 5,
        'run_body': textwrap.dedent("""\
            native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_2, [32], inputs.primals_17, inputs.primals_18, 1e-05)
            getitem_9 = operator.getitem(native_layer_norm, 0)
            getitem_10 = operator.getitem(native_layer_norm, 1)
            getitem_11 = operator.getitem(native_layer_norm, 2)
            add_3 = torch.ops.aten.add.Tensor(inputs.add_2, inputs.view_25)
            return [add_3]"""),
    },
    {
        'id': 'fw_018', 'name': 'view', 'line': 'model.py:22', 'module': '[blocks.1]',
        'desc': 'view (qkv linear)',
        'input_keys': ['getitem_9', 'primals_19', 'primals_20'], 'output_keys': ['view_16'],
        'needs_operator': False,
        'ops_desc': 'view x2, t x1, addmm x1', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            view_15 = torch.ops.aten.view.default(inputs.getitem_9, [32, 32])
            t_4 = torch.ops.aten.t.default(inputs.primals_19)
            addmm_4 = torch.ops.aten.addmm.default(inputs.primals_20, view_15, t_4)
            view_16 = torch.ops.aten.view.default(addmm_4, [2, 16, 96])
            return [view_16]"""),
    },
    {
        'id': 'fw_019', 'name': 'split', 'line': 'model.py:23', 'module': '[blocks.1]',
        'desc': 'split (q,k,v)',
        'input_keys': ['view_16'], 'output_keys': ['getitem_12', 'getitem_13', 'getitem_14'],
        'needs_operator': True,
        'ops_desc': 'split x1, getitem x3', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            split_1 = torch.ops.aten.split.Tensor(inputs.view_16, 32, 2)
            getitem_12 = operator.getitem(split_1, 0)
            getitem_13 = operator.getitem(split_1, 1)
            getitem_14 = operator.getitem(split_1, 2)
            return [getitem_12, getitem_13, getitem_14]"""),
    },
    {
        'id': 'fw_020', 'name': 'transpose', 'line': 'model.py:24', 'module': '[blocks.1]',
        'desc': 'transpose (q reshape)',
        'input_keys': ['getitem_12'], 'output_keys': ['transpose_5'],
        'needs_operator': False,
        'ops_desc': 'view x1, transpose x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            view_17 = torch.ops.aten.view.default(inputs.getitem_12, [2, 16, 2, 16])
            transpose_5 = torch.ops.aten.transpose.int(view_17, 1, 2)
            return [transpose_5]"""),
    },
    {
        'id': 'fw_021', 'name': 'transpose', 'line': 'model.py:25', 'module': '[blocks.1]',
        'desc': 'transpose (k reshape)',
        'input_keys': ['getitem_13'], 'output_keys': ['transpose_6'],
        'needs_operator': False,
        'ops_desc': 'view x1, transpose x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            view_18 = torch.ops.aten.view.default(inputs.getitem_13, [2, 16, 2, 16])
            transpose_6 = torch.ops.aten.transpose.int(view_18, 1, 2)
            return [transpose_6]"""),
    },
    {
        'id': 'fw_022', 'name': 'transpose', 'line': 'model.py:26', 'module': '[blocks.1]',
        'desc': 'transpose (v reshape)',
        'input_keys': ['getitem_14'], 'output_keys': ['transpose_7'],
        'needs_operator': False,
        'ops_desc': 'view x1, transpose x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            view_19 = torch.ops.aten.view.default(inputs.getitem_14, [2, 16, 2, 16])
            transpose_7 = torch.ops.aten.transpose.int(view_19, 1, 2)
            return [transpose_7]"""),
    },
    {
        'id': 'fw_023', 'name': 'mul', 'line': 'model.py:27', 'module': '[blocks.1]',
        'desc': 'mul (QK^T scaled)',
        'input_keys': ['transpose_6', 'transpose_5'], 'output_keys': ['mul_1'],
        'needs_operator': False,
        'ops_desc': 'transpose x1, expand x2, clone x2, _unsafe_view x2, bmm x1, view x1, mul x1', 'n_ops': 10,
        'run_body': textwrap.dedent("""\
            transpose_8 = torch.ops.aten.transpose.int(inputs.transpose_6, -2, -1)
            expand_4 = torch.ops.aten.expand.default(inputs.transpose_5, [2, 2, 16, 16])
            expand_5 = torch.ops.aten.expand.default(transpose_8, [2, 2, 16, 16])
            clone_4 = torch.ops.aten.clone.default(expand_4, memory_format=torch.contiguous_format)
            clone_5 = torch.ops.aten.clone.default(expand_5, memory_format=torch.contiguous_format)
            _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_4, [4, 16, 16])
            _unsafe_view_4 = torch.ops.aten._unsafe_view.default(clone_5, [4, 16, 16])
            bmm_2 = torch.ops.aten.bmm.default(_unsafe_view_3, _unsafe_view_4)
            view_20 = torch.ops.aten.view.default(bmm_2, [2, 2, 16, 16])
            mul_1 = torch.ops.aten.mul.Tensor(view_20, 0.25)
            return [mul_1]"""),
    },
    {
        'id': 'fw_024', 'name': 'masked_fill', 'line': 'model.py:28', 'module': '[blocks.1]',
        'desc': 'masked_fill (causal mask)',
        'input_keys': ['primals_21', 'mul_1'], 'output_keys': ['masked_fill_1'],
        'needs_operator': False,
        'ops_desc': 'alias x1, eq x1, masked_fill x1', 'n_ops': 3,
        'run_body': textwrap.dedent("""\
            alias_1 = torch.ops.aten.alias.default(inputs.primals_21)
            eq_1 = torch.ops.aten.eq.Scalar(alias_1, 0)
            masked_fill_1 = torch.ops.aten.masked_fill.Scalar(inputs.mul_1, eq_1, float('-inf'))
            return [masked_fill_1]"""),
    },
    {
        'id': 'fw_025', 'name': 'detach', 'line': 'model.py:29', 'module': '[blocks.1]',
        'desc': 'detach (softmax)',
        'input_keys': ['masked_fill_1'], 'output_keys': ['detach_1'],
        'needs_operator': False,
        'ops_desc': '_softmax x1, detach x1', 'n_ops': 2,
        'run_body': textwrap.dedent("""\
            _softmax_1 = torch.ops.aten._softmax.default(inputs.masked_fill_1, -1, False)
            detach_1 = torch.ops.aten.detach.default(_softmax_1)
            return [detach_1]"""),
    },
    {
        'id': 'fw_026', 'name': 'view', 'line': 'model.py:30', 'module': '[blocks.1]',
        'desc': 'view (att @ v)',
        'input_keys': ['_softmax_1', 'transpose_7'], 'output_keys': ['view_22'],
        'needs_operator': False,
        'ops_desc': 'expand x2, view x1, clone x1, _unsafe_view x1, bmm x1, view x1', 'n_ops': 7,
        'run_body': textwrap.dedent("""\
            expand_6 = torch.ops.aten.expand.default(inputs._softmax_1, [2, 2, 16, 16])
            expand_7 = torch.ops.aten.expand.default(inputs.transpose_7, [2, 2, 16, 16])
            view_21 = torch.ops.aten.view.default(expand_6, [4, 16, 16])
            clone_6 = torch.ops.aten.clone.default(expand_7, memory_format=torch.contiguous_format)
            _unsafe_view_5 = torch.ops.aten._unsafe_view.default(clone_6, [4, 16, 16])
            bmm_3 = torch.ops.aten.bmm.default(view_21, _unsafe_view_5)
            view_22 = torch.ops.aten.view.default(bmm_3, [2, 2, 16, 16])
            return [view_22]"""),
    },
    {
        'id': 'fw_027', 'name': 'view', 'line': 'model.py:31', 'module': '[blocks.1]',
        'desc': 'view (y reshape)',
        'input_keys': ['view_22'], 'output_keys': ['view_23'],
        'needs_operator': False,
        'ops_desc': 'transpose x1, clone x1, view x1', 'n_ops': 3,
        'run_body': textwrap.dedent("""\
            transpose_9 = torch.ops.aten.transpose.int(inputs.view_22, 1, 2)
            clone_7 = torch.ops.aten.clone.default(transpose_9, memory_format=torch.contiguous_format)
            view_23 = torch.ops.aten.view.default(clone_7, [2, 16, 32])
            return [view_23]"""),
    },
    {
        'id': 'fw_028', 'name': 'view', 'line': 'model.py:32', 'module': '[blocks.1]',
        'desc': 'view (c_proj)',
        'input_keys': ['view_23', 'primals_22', 'primals_23'], 'output_keys': ['view_25'],
        'needs_operator': False,
        'ops_desc': 'view x2, t x1, addmm x1', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            view_24 = torch.ops.aten.view.default(inputs.view_23, [32, 32])
            t_5 = torch.ops.aten.t.default(inputs.primals_22)
            addmm_5 = torch.ops.aten.addmm.default(inputs.primals_23, view_24, t_5)
            view_25 = torch.ops.aten.view.default(addmm_5, [2, 16, 32])
            return [view_25]"""),
    },
    {
        'id': 'fw_029', 'name': 'add', 'line': 'model.py:49', 'module': '[blocks.1]',
        'desc': 'add (MLP residual, ln_2)',
        'input_keys': ['add_3', 'primals_24', 'primals_25', 'primals_26', 'primals_27', 'primals_28', 'primals_29'],
        'output_keys': ['add_4'],
        'needs_operator': True,
        'ops_desc': 'native_layer_norm x1, getitem x3, view x4, t x2, addmm x2, gelu x1, add x1', 'n_ops': 14,
        'run_body': textwrap.dedent("""\
            native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_3, [32], inputs.primals_24, inputs.primals_25, 1e-05)
            getitem_15 = operator.getitem(native_layer_norm, 0)
            getitem_16 = operator.getitem(native_layer_norm, 1)
            getitem_17 = operator.getitem(native_layer_norm, 2)
            view_26 = torch.ops.aten.view.default(getitem_15, [32, 32])
            t_6 = torch.ops.aten.t.default(inputs.primals_26)
            addmm_6 = torch.ops.aten.addmm.default(inputs.primals_27, view_26, t_6)
            view_27 = torch.ops.aten.view.default(addmm_6, [2, 16, 128])
            gelu_1 = torch.ops.aten.gelu.default(view_27)
            view_28 = torch.ops.aten.view.default(gelu_1, [32, 128])
            t_7 = torch.ops.aten.t.default(inputs.primals_28)
            addmm_7 = torch.ops.aten.addmm.default(inputs.primals_29, view_28, t_7)
            view_29 = torch.ops.aten.view.default(addmm_7, [2, 16, 32])
            add_4 = torch.ops.aten.add.Tensor(inputs.add_3, view_29)
            return [add_4]"""),
    },
    {
        'id': 'fw_030', 'name': 'native_layer_norm', 'line': 'model.py:71', 'module': '[ln_f]',
        'desc': 'native_layer_norm (final LN)',
        'input_keys': ['add_4', 'primals_30', 'primals_31'],
        'output_keys': ['getitem_18', 'getitem_19', 'getitem_20'],
        'needs_operator': True,
        'ops_desc': 'native_layer_norm x1, getitem x3', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            native_layer_norm = torch.ops.aten.native_layer_norm.default(inputs.add_4, [32], inputs.primals_30, inputs.primals_31, 1e-05)
            getitem_18 = operator.getitem(native_layer_norm, 0)
            getitem_19 = operator.getitem(native_layer_norm, 1)
            getitem_20 = operator.getitem(native_layer_norm, 2)
            return [getitem_18, getitem_19, getitem_20]"""),
    },
    {
        'id': 'fw_031', 'name': '_unsafe_view', 'line': 'model.py:72', 'module': '[lm_head]',
        'desc': '_unsafe_view (logits)',
        'input_keys': ['primals_2', 'getitem_18'], 'output_keys': ['_unsafe_view_6'],
        'needs_operator': False,
        'ops_desc': 't x1, view x1, mm x1, _unsafe_view x1', 'n_ops': 4,
        'run_body': textwrap.dedent("""\
            t_8 = torch.ops.aten.t.default(inputs.primals_2)
            view_30 = torch.ops.aten.view.default(inputs.getitem_18, [32, 32])
            mm = torch.ops.aten.mm.default(view_30, t_8)
            _unsafe_view_6 = torch.ops.aten._unsafe_view.default(mm, [2, 16, 64])
            return [_unsafe_view_6]"""),
    },
]


# ---------------------------------------------------------------------------
# H5 saving
# ---------------------------------------------------------------------------
def _tensor_to_numpy(tensor):
    """Convert a tensor to numpy, handling non-contiguous tensors."""
    t = tensor.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    return t.numpy()


def save_op_h5(path, input_tensors_dict, output_tensors_list):
    """Save input tensors and expected output(s) to an h5 file.

    Args:
        path: Output h5 file path.
        input_tensors_dict: dict of {key_name: tensor} for inputs.
        output_tensors_list: list of (key_name, tensor) for outputs.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        # Save inputs
        for key, tensor in input_tensors_dict.items():
            arr = _tensor_to_numpy(tensor)
            ds = f.create_dataset(key, data=arr)
            ds.attrs['torch_dtype'] = str(tensor.dtype).replace('torch.', '')

        # Save outputs
        if len(output_tensors_list) == 1:
            _name, tensor = output_tensors_list[0]
            arr = _tensor_to_numpy(tensor)
            ds = f.create_dataset('expected', data=arr)
            ds.attrs['torch_dtype'] = str(tensor.dtype).replace('torch.', '')
        else:
            for i, (_name, tensor) in enumerate(output_tensors_list):
                arr = _tensor_to_numpy(tensor)
                ds = f.create_dataset(f'expected_{i}', data=arr)
                ds.attrs['torch_dtype'] = str(tensor.dtype).replace('torch.', '')


# ---------------------------------------------------------------------------
# Shape / size formatting helpers
# ---------------------------------------------------------------------------
def _dtype_str(tensor):
    """Return dtype string like 'float32' or 'int64'."""
    return str(tensor.dtype).replace('torch.', '')


def _shape_str(tensor):
    """Return shape string like 'float32[2x16x32]'."""
    dims = 'x'.join(str(d) for d in tensor.shape)
    return f"{_dtype_str(tensor)}[{dims}]"


def _byte_size(tensor):
    """Return human-readable byte size."""
    nbytes = tensor.nelement() * tensor.element_size()
    if nbytes >= 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    elif nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    else:
        return f"{nbytes} B"


# ---------------------------------------------------------------------------
# Test file generation
# ---------------------------------------------------------------------------
def generate_test_file(op_def, all_tensors, primals, output_dir):
    """Generate a single kernelbox .py test file."""
    op_id = op_def['id']
    op_name = op_def['name']
    filename = f"{op_id}_{op_name}.py"
    filepath = os.path.join(output_dir, filename)

    # Build the header line: NNN_model.py:LINE [module] | desc
    num = op_id.replace('fw_', '')
    module_str = f" {op_def['module']}" if op_def['module'] else ''
    header = f"{num}_{op_def['line']}{module_str} | {op_def['desc']}"

    # Collect input tensor info
    input_lines = []
    total_input_bytes = 0
    for key in op_def['input_keys']:
        if key.startswith('primals_'):
            tensor = primals[key]
        else:
            tensor = all_tensors[key]
        shape = _shape_str(tensor)
        size = _byte_size(tensor)
        total_input_bytes += tensor.nelement() * tensor.element_size()
        input_lines.append(f"  {key}  {shape}  {size}")

    # Collect output tensor info
    output_lines = []
    total_output_bytes = 0
    for key in op_def['output_keys']:
        tensor = all_tensors[key]
        shape = _shape_str(tensor)
        size = _byte_size(tensor)
        total_output_bytes += tensor.nelement() * tensor.element_size()
        output_lines.append(f"  {key}  {shape}  {size}")

    # Format total sizes
    def _fmt_total(nbytes):
        if nbytes >= 1024 * 1024:
            return f"{nbytes / (1024 * 1024):.1f} MB"
        elif nbytes >= 1024:
            return f"{nbytes / 1024:.1f} KB"
        else:
            return f"{nbytes} B"

    n_inputs = len(op_def['input_keys'])
    n_outputs = len(op_def['output_keys'])
    input_total_str = _fmt_total(total_input_bytes)
    output_total_str = _fmt_total(total_output_bytes)

    # Build docstring
    doc_parts = [f'"""{header}\n']

    if input_lines:
        doc_parts.append(f'\nInputs ({input_total_str} total):')
        for line in input_lines:
            doc_parts.append(line)
    else:
        doc_parts.append(f'\nInputs (0 B total):')
        doc_parts.append('  (none)')

    doc_parts.append(f'Outputs ({output_total_str} total):')
    for line in output_lines:
        doc_parts.append(line)

    doc_parts.append(f'Ops: {op_def["ops_desc"]}  ({op_def["n_ops"]} ops)')
    doc_parts.append(f'\n    kbox iterate {filename}')
    doc_parts.append('"""')

    docstring = '\n'.join(doc_parts)

    # Build imports
    imports = ['import torch']
    if op_def['needs_operator']:
        imports.append('import operator')

    # Build run body with proper indentation
    run_body_lines = op_def['run_body'].split('\n')
    indented_body = '\n'.join('    ' + line for line in run_body_lines)

    # Assemble file
    parts = [
        docstring,
        '\n'.join(imports),
        '',
        '',
        'def init_once():',
        f'    return {{"h5": "data/{op_id}_{op_name}.h5"}}',
        '',
        '',
        'def run(inputs):',
        indented_body,
        '',
    ]

    content = '\n'.join(parts)

    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate nanoGPT kernelbox test files')
    parser.add_argument('--output-dir', default='outputs/nanogpt/nanogpt_kbox_per_aten',
                        help='Output directory (default: outputs/nanogpt/nanogpt_kbox_per_aten)')
    args = parser.parse_args()

    output_dir = args.output_dir
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    device = 'cpu'

    # Generate primals and run forward pass
    primals = generate_primals(device)
    all_tensors = run_forward(primals, device)

    n_py = 0
    n_h5 = 0

    for op_def in OPS:
        op_id = op_def['id']
        op_name = op_def['name']

        # Collect input tensors for h5
        input_tensors = {}
        for key in op_def['input_keys']:
            if key.startswith('primals_'):
                input_tensors[key] = primals[key]
            else:
                input_tensors[key] = all_tensors[key]

        # Collect output tensors for h5
        output_tensors = []
        for key in op_def['output_keys']:
            output_tensors.append((key, all_tensors[key]))

        # Save h5
        h5_path = os.path.join(data_dir, f'{op_id}_{op_name}.h5')
        save_op_h5(h5_path, input_tensors, output_tensors)
        n_h5 += 1

        # Generate .py
        generate_test_file(op_def, all_tensors, primals, output_dir)
        n_py += 1

    # Print summary
    print(f"Generated {n_py} kbox files in {output_dir}/")
    print(f"Generated {n_h5} h5 fixtures in {data_dir}/")

    # Count op names
    name_counts = Counter(op['name'] for op in OPS)
    ops_parts = []
    for name, count in name_counts.items():
        if count == 1:
            ops_parts.append(name)
        else:
            ops_parts.append(f"{name} x{count}")
    print(f"Ops covered: {', '.join(ops_parts)}")


if __name__ == '__main__':
    main()
