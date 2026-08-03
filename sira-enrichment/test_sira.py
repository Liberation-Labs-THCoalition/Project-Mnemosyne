"""Tests for SIRA Enrichment — runs without LLM using domain mapping mode."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from sira import (
    SIRAIndex, LLMEnricher, DomainMapper, QueryExpander,
    MemoryEnricher, extract_terms,
)


class TestExtractTerms:
    def test_valid_json(self):
        assert extract_terms('{"terms": ["sql injection", "sqli"]}') == ["sql injection", "sqli"]

    def test_with_thinking_tags(self):
        resp = '<think>reasoning here</think>{"terms": ["term1", "term2"]}'
        assert extract_terms(resp) == ["term1", "term2"]

    def test_empty(self):
        assert extract_terms("") == []
        assert extract_terms(None) == []

    def test_invalid_json(self):
        assert extract_terms("not json at all") == []

    def test_filters_short_terms(self):
        assert extract_terms('{"terms": ["a", "ab", "abc"]}') == ["ab", "abc"]

    def test_lowercases(self):
        assert extract_terms('{"terms": ["SQL", "XSS"]}') == ["sql", "xss"]


class TestSIRAIndex:
    def test_add_and_search(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            doc_id = idx.add_document("KV cache geometry research", "doc1", "test")
            idx.set_enrichment(doc_id, ["attention", "transformer", "eigenvalue"])
            idx.build_term_stats()

            results = idx.search("attention geometry")
            assert len(results) > 0
            assert results[0]["doc_key"] == "doc1"
            idx.close()

    def test_duplicate_doc_key(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            id1 = idx.add_document("content one", "same_key", "test")
            id2 = idx.add_document("content two", "same_key", "test")
            assert id1 == id2
            idx.close()

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            idx.add_document("doc one", "d1")
            idx.add_document("doc two", "d2")
            s = idx.stats()
            assert s["total_docs"] == 2
            assert s["enriched_docs"] == 0
            idx.close()

    def test_term_validation(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            for i in range(10):
                did = idx.add_document(f"document {i} about python", f"d{i}")
                idx.set_enrichment(did, ["python", f"unique_{i}"])
            idx.build_term_stats()

            valid, reason = idx.validate_term("python", tau=0.5)
            assert not valid
            assert "too_common" in reason

            valid, reason = idx.validate_term("unique_3", tau=0.5)
            assert valid

            valid, reason = idx.validate_term("nonexistent", tau=0.5)
            assert not valid
            assert reason == "not_in_index"
            idx.close()

    def test_unenriched(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            idx.add_document("not enriched", "d1")
            d2 = idx.add_document("enriched", "d2")
            idx.set_enrichment(d2, ["term"])

            unenriched = idx.get_unenriched()
            assert len(unenriched) == 1
            assert unenriched[0][1] == "not enriched"
            idx.close()


class TestDomainMapper:
    def test_basic_mapping(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            did = idx.add_document("Thomas built the sewing machine interface", "d1")

            mapper = DomainMapper(idx, {
                "example_user": ["example_handle", "example user"],
                "sewing machine": ["singer futura", "embroidery"],
            })
            terms = mapper.enrich_document(did, "Thomas built the sewing machine interface")
            assert "example_handle" in terms
            assert "singer futura" in terms
            idx.close()

    def test_no_match(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            did = idx.add_document("unrelated content", "d1")
            mapper = DomainMapper(idx, {"python": ["cpython"]})
            terms = mapper.enrich_document(did, "unrelated content")
            assert terms == []
            idx.close()

    def test_save_load_mappings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/mappings.json"
            mapper = DomainMapper(None, {"kv cache": ["key-value", "attention"]})
            mapper.save_mappings(path)

            mapper2 = DomainMapper(None)
            mapper2.load_mappings(path)
            assert "kv cache" in mapper2.mappings
            assert "attention" in mapper2.mappings["kv cache"]

    def test_enrich_all(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            idx.add_document("Thomas wrote code", "d1")
            idx.add_document("Vera designed Oracle", "d2")
            idx.add_document("unrelated stuff", "d3")

            mapper = DomainMapper(idx, {
                "example_user": ["example_handle"],
                "vera": ["oracle designer"],
            })
            result = mapper.enrich_all()
            assert result["enriched"] == 2
            assert result["total"] == 3

            s = idx.stats()
            assert s["enriched_docs"] == 2
            idx.close()


class TestQueryExpander:
    @patch("sira.llm_generate")
    def test_expansion_with_validation(self, mock_llm):
        mock_llm.return_value = '{"terms": ["rare_term", "common_term", "missing_term"]}'

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            idx = SIRAIndex(f.name)
            for i in range(10):
                did = idx.add_document(f"doc {i} common_term", f"d{i}")
                if i == 0:
                    idx.set_enrichment(did, ["rare_term", "common_term"])
                else:
                    idx.set_enrichment(did, ["common_term"])
            idx.build_term_stats()

            expander = QueryExpander(idx, tau=0.5)
            result = expander.expand("test query")

            assert "rare_term" in result["added_terms"]
            assert any(t == "missing_term" for t, _ in result["rejected_terms"])
            idx.close()


class TestMemoryEnricher:
    def test_domain_enrichment(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)
            conn.execute("""
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    search_terms TEXT
                )
            """)
            conn.execute("INSERT INTO memories (content) VALUES (?)",
                         ("Thomas worked on the sewing machine",))
            conn.execute("INSERT INTO memories (content) VALUES (?)",
                         ("unrelated content about weather",))
            conn.commit()
            conn.close()

            enricher = MemoryEnricher(
                f.name, "domain",
                {"example_user": ["example_handle", "example user"]}
            )
            result = enricher.enrich()
            assert result["enriched"] == 1
            assert result["terms"] >= 2

            conn = sqlite3.connect(f.name)
            row = conn.execute(
                "SELECT search_terms FROM memories WHERE id = 1"
            ).fetchone()
            assert "example_handle" in row[0]
            conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
