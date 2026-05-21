"""Tests for KV Knowledge Packs — runs without GPU using mock model.

Validates the composition logic, fact store, and Muse values pipeline
without requiring a real transformer model. Integration tests with
actual models run on the Mac Studio.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch
import pytest

from kv_packs import (
    CacheBlock, KVPackBuilder, CacheComposer, FactStore,
    generate_with_kvpack,
)


def make_mock_kv(num_layers=4, seq_len=10, hidden=32, num_heads=4):
    """Create realistic mock KV cache tensors."""
    head_dim = hidden // num_heads
    kv = tuple(
        (
            torch.randn(1, num_heads, seq_len, head_dim),
            torch.randn(1, num_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )
    return kv


def make_mock_model(num_layers=4, hidden=32, num_heads=4):
    model = MagicMock()
    model.device = 'cpu'

    def mock_forward(**kwargs):
        input_ids = kwargs.get('input_ids')
        past = kwargs.get('past_key_values')

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        else:
            seq_len = 1

        if past is not None:
            prefix_len = past[0][0].shape[2]
            total_len = prefix_len + seq_len
        else:
            total_len = seq_len

        result = MagicMock()
        result.past_key_values = make_mock_kv(num_layers, total_len, hidden, num_heads)
        return result

    model.side_effect = mock_forward
    model.return_value = mock_forward(input_ids=torch.tensor([[1, 2, 3]]))
    return model


def make_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer.apply_chat_template.return_value = "<|im_start|>system\ntest<|im_end|>"
    tokenizer.decode.return_value = "Generated response text"
    return tokenizer


class TestCacheBlock:
    def test_age(self):
        block = CacheBlock(
            key_values=make_mock_kv(seq_len=5),
            seq_length=5,
            source_hash='abc123',
        )
        assert block.age >= 0
        assert block.age < 1

    def test_metadata(self):
        block = CacheBlock(
            key_values=make_mock_kv(),
            seq_length=10,
            source_hash='def456',
            label='test-block',
        )
        assert block.label == 'test-block'
        assert block.source_hash == 'def456'


class TestCacheComposer:
    def test_single_block(self):
        builder = MagicMock()
        composer = CacheComposer(builder)
        block = CacheBlock(
            key_values=make_mock_kv(seq_len=5),
            seq_length=5,
            source_hash='a',
        )
        result = composer.compose(block)
        assert result is block

    def test_two_blocks(self):
        builder = MagicMock()
        composer = CacheComposer(builder)
        b1 = CacheBlock(key_values=make_mock_kv(seq_len=5), seq_length=5, source_hash='a')
        b2 = CacheBlock(key_values=make_mock_kv(seq_len=3), seq_length=3, source_hash='b')
        result = composer.compose(b1, b2)
        assert result.seq_length == 8
        for k, v in result.key_values:
            assert k.shape[2] == 8
            assert v.shape[2] == 8

    def test_three_blocks(self):
        builder = MagicMock()
        composer = CacheComposer(builder)
        blocks = [
            CacheBlock(key_values=make_mock_kv(seq_len=s), seq_length=s, source_hash=str(i))
            for i, s in enumerate([10, 5, 3])
        ]
        result = composer.compose(*blocks)
        assert result.seq_length == 18
        assert 'composed(3 blocks)' in result.label

    def test_empty_raises(self):
        builder = MagicMock()
        composer = CacheComposer(builder)
        with pytest.raises(ValueError):
            composer.compose()


class TestFactStore:
    def test_create_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FactStore(Path(tmpdir))
            bank = store.create_bank('test', ['fact 1', 'fact 2'], {'topic': 'testing'})
            assert len(bank.facts) == 2

            loaded = store.get_bank('test')
            assert loaded.facts == ['fact 1', 'fact 2']

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = FactStore(Path(tmpdir))
            store1.create_bank('persist', ['saved fact'])

            store2 = FactStore(Path(tmpdir))
            loaded = store2.get_bank('persist')
            assert loaded is not None
            assert loaded.facts == ['saved fact']

    def test_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FactStore(Path(tmpdir))
            store.create_bank('update', ['old'])
            store.update_bank('update', ['new1', 'new2'])
            assert store.get_bank('update').facts == ['new1', 'new2']

    def test_list_banks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FactStore(Path(tmpdir))
            store.create_bank('a', ['fact'])
            store.create_bank('b', ['fact'])
            assert sorted(store.list_banks()) == ['a', 'b']

    def test_hash_deterministic(self):
        from kv_packs import FactBank
        b1 = FactBank(facts=['a', 'b'])
        b2 = FactBank(facts=['a', 'b'])
        assert b1.hash() == b2.hash()

        b3 = FactBank(facts=['b', 'a'])
        assert b3.hash() == b1.hash()  # sorted before hashing


class TestMuseValues:
    def test_default_consent_framework(self):
        from muse_values import MuseValuesConfig
        config = MuseValuesConfig(values_path=Path('/tmp/fake.json'))
        assert 'red' in config.safewords
        assert 'safeword' in config.safewords

    def test_values_formatting(self):
        from muse_values import MuseValuesInjector, MuseValuesConfig
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'principles': ['Be kind', 'Be honest'],
                'boundaries': ['No harm', 'No manipulation'],
                'safewords': ['red', 'stop'],
            }, f)
            f.flush()

            config = MuseValuesConfig(values_path=Path(f.name))
            builder = MagicMock()
            injector = MuseValuesInjector(builder, config)
            formatted = injector._format_values(json.loads(Path(f.name).read_text()))

            assert 'Be kind' in formatted
            assert 'No harm' in formatted
            assert 'red' in formatted


class TestTGSBridge:
    @patch('tgs_kvpack_bridge.retrieve_memories')
    def test_empty_retrieval(self, mock_retrieve):
        mock_retrieve.return_value = []

        from tgs_kvpack_bridge import TGSKVPackBridge
        builder = MagicMock()
        system = CacheBlock(key_values=make_mock_kv(), seq_length=10, source_hash='sys')
        bridge = TGSKVPackBridge(builder, system_block=system)

        result = bridge.retrieve_and_inject("test query")
        assert result is system

    def test_stats(self):
        from tgs_kvpack_bridge import TGSKVPackBridge
        builder = MagicMock()
        builder.cache_stats.return_value = {'blocks': 0, 'total_tokens': 0, 'oldest_age': 0}
        bridge = TGSKVPackBridge(builder)
        stats = bridge.stats()
        assert 'cached_retrievals' in stats
        assert 'builder_stats' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
