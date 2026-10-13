import json
from pathlib import Path

import torch

from neurreps.training import resolve_device


def test_v10_focused_configs_use_three_paired_seeds_and_auto_device():
    expected = {
        'configs/neurreps_sha_test_focused_main.json': 9,
        'configs/neurreps_sha_test_focused_capacity_s32.json': 6,
        'configs/neurreps_sha_test_focused_padding_s32.json': 9,
    }
    for path, count in expected.items():
        cfg = json.loads(Path(path).read_text(encoding='utf-8'))
        assert cfg['S_values'] == [32]
        assert cfg['seeds'] == [11, 37, 71]
        assert cfg['device'] == 'auto'
        assert cfg['device_policy'] == 'auto_cuda_if_available_else_cpu'
        assert cfg['expected_unique_cnn_trainings'] == count


def test_resolve_device_auto_cuda_else_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device(None) == 'cpu'
    assert resolve_device('auto') == 'cpu'
    assert resolve_device('cuda') == 'cpu'


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert resolve_device(None) == 'cuda'
    assert resolve_device('auto') == 'cuda'
    assert resolve_device('cpu') == 'cpu'
