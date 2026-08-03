"""TGS-RAG Bridge: Text-Graph Synergistic retrieval.

Fuses full-text search (FTS5) with knowledge graph retrieval (HippoRAG)
using bidirectional verification:
  Graph→Text: Graph entity mentions re-rank text search results
  Text→Graph: Top text entities seed additional graph retrievals (orphan bridging)

Scoring formula (Global Voting):
  Score = alpha * Norm(text_similarity) + (1-alpha) * Norm(entity_count)

Based on arXiv:2605.05643 with adaptations for agent memory infrastructure.

Designed as a standalone HTTP service. Configure via environment:
  HIPPORAG_URL    — HippoRAG endpoint (default: http://hipporag-memory:11235)
  MEMORY_DB       — SQLite memory database path
  TGS_ALPHA       — Text vs graph weight balance (default: 0.5)
  TGS_ORPHAN_CAP  — Max orphan entities for graph bridging (default: 3)
  TGS_EPSILON     — Score discount for bridge results (default: 0.4)
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import aiohttp

logging.basicConfig(level=logging.INFO, format='[tgs-bridge] %(message)s')
logger = logging.getLogger(__name__)

HIPPORAG_URL = os.environ.get('HIPPORAG_URL', 'http://hipporag-memory:11235')
MEMORY_DB = os.environ.get('MEMORY_DB', '')
ALPHA = float(os.environ.get('TGS_ALPHA', '0.5'))
ORPHAN_CAP = int(os.environ.get('TGS_ORPHAN_CAP', '3'))
EPSILON = float(os.environ.get('TGS_EPSILON', '0.4'))


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    if rng == 0:
        return [1.0] * len(scores)
    return [(s - mn) / rng for s in scores]


def extract_entities(text: str) -> set[str]:
    """Extract candidate entity mentions from text.
    Capitalized phrases, acronyms, and filenames. No model call."""
    entities = set()
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
        ent = m.group(1)
        if len(ent) > 2 and ent not in {'The', 'This', 'That', 'What', 'When',
                                          'Where', 'Which', 'Here', 'There'}:
            entities.add(ent.lower())
    for m in re.finditer(r'\b([A-Z]{2,}(?:-[A-Z]+)*)\b', text):
        entities.add(m.group(1).lower())
    for m in re.finditer(r'\b([\w-]+\.(?:py|ts|md|sql|json|yaml))\b', text):
        entities.add(m.group(1).lower())
    return entities


def search_memory_fts(db_path: str, query: str, limit: int = 20) -> list[dict]:
    """Full-text search against memory SQLite with SIRA enrichment support."""
    if not db_path:
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        fts_query = ' OR '.join(f'"{w}"' for w in query.split() if len(w) > 2)
        if not fts_query:
            return []
        rows = conn.execute("""
            SELECT m.id, m.content, m.memory_type, m.tags,
                   fts.rank as fts_rank
            FROM memory_content_fts fts
            JOIN memories m ON m.id = fts.rowid
            WHERE memory_content_fts MATCH ?
            UNION
            SELECT m.id, m.content, m.memory_type, m.tags,
                   -10.0 as fts_rank
            FROM memories m
            WHERE m.search_terms LIKE '%' || ? || '%'
              AND m.id NOT IN (
                  SELECT rowid FROM memory_content_fts WHERE memory_content_fts MATCH ?
              )
            ORDER BY fts_rank
            LIMIT ?
        """, (fts_query, query.split()[0].lower() if query.split() else '', fts_query, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f'FTS search failed: {e}')
        return []
    finally:
        conn.close()


async def query_hipporag(url: str, query: str, num: int = 10) -> dict:
    """Query HippoRAG for graph-based retrieval."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f'{url}/query',
                json={'queries': [query], 'num_to_retrieve': num}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get('results', [])
                    if results:
                        return results[0]
        except Exception as e:
            logger.error(f'HippoRAG query failed: {e}')
    return {'docs': [], 'doc_scores': []}


async def tgs_retrieve(query: str, num_results: int = 10,
                       alpha: Optional[float] = None,
                       hipporag_url: str = None,
                       memory_db: str = None) -> list[dict]:
    """TGS-RAG retrieval: fuse graph and text results with bidirectional voting."""
    if alpha is None:
        alpha = ALPHA
    hip_url = hipporag_url or HIPPORAG_URL
    db = memory_db or MEMORY_DB

    graph_task = query_hipporag(hip_url, query, num_results * 2)
    text_results = search_memory_fts(db, query, num_results * 3)
    graph_results = await graph_task

    graph_docs = graph_results.get('docs', [])
    graph_scores = graph_results.get('doc_scores', [])

    logger.info(f'Query: "{query[:60]}" -> graph:{len(graph_docs)} text:{len(text_results)}')

    graph_entities = set()
    for doc in graph_docs:
        graph_entities.update(extract_entities(doc))

    scored_text = []
    for mem in text_results:
        content = mem.get('content', '')
        text_entities = extract_entities(content)
        content_lower = content.lower()
        rec_count = sum(1 for ge in graph_entities if ge in content_lower)
        scored_text.append({
            'id': mem.get('id'),
            'content': content,
            'memory_type': mem.get('memory_type', ''),
            'fts_rank': mem.get('fts_rank', 0),
            'rec_count': rec_count,
            'entities': text_entities,
            'source': 'text',
        })

    if scored_text:
        fts_scores = [-m['fts_rank'] for m in scored_text]
        rec_scores = [m['rec_count'] for m in scored_text]
        norm_fts = normalize(fts_scores)
        norm_rec = normalize(rec_scores)
        for i, mem in enumerate(scored_text):
            mem['tgs_score'] = alpha * norm_fts[i] + (1 - alpha) * norm_rec[i]

    text_entity_pool = set()
    for mem in scored_text[:10]:
        text_entity_pool.update(mem.get('entities', set()))

    orphan_entities = text_entity_pool - graph_entities
    orphan_docs = []

    if orphan_entities:
        top_orphans = list(orphan_entities)[:ORPHAN_CAP]
        logger.info(f'Orphan entities for graph bridging: {top_orphans}')
        for orphan in top_orphans:
            bridge_result = await query_hipporag(hip_url, orphan, 3)
            bridge_docs = bridge_result.get('docs', [])
            bridge_scores = bridge_result.get('doc_scores', [])
            for j, doc in enumerate(bridge_docs):
                score = bridge_scores[j] if j < len(bridge_scores) else 0.5
                orphan_docs.append({
                    'content': doc,
                    'tgs_score': score * EPSILON,
                    'source': 'bridge',
                    'bridge_entity': orphan,
                })

    all_results = []
    for i, doc in enumerate(graph_docs):
        score = graph_scores[i] if i < len(graph_scores) else 0.5
        all_results.append({'content': doc, 'tgs_score': score, 'source': 'graph'})
    all_results.extend(scored_text)
    all_results.extend(orphan_docs)

    seen = set()
    deduped = []
    for r in all_results:
        key = r['content'][:100].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    deduped.sort(key=lambda x: x.get('tgs_score', 0), reverse=True)
    return deduped[:num_results]


class TGSHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == '/retrieve':
            query = body.get('query', '')
            num = body.get('num_results', 10)
            alpha = body.get('alpha')
            loop = asyncio.new_event_loop()
            results = loop.run_until_complete(tgs_retrieve(query, num, alpha))
            loop.close()
            self._respond(200, {'query': query, 'results': results})

        elif self.path == '/retrieve/text':
            query = body.get('query', '')
            results = search_memory_fts(MEMORY_DB, query, body.get('limit', 20))
            self._respond(200, {'results': results})

        elif self.path == '/retrieve/graph':
            query = body.get('query', '')
            loop = asyncio.new_event_loop()
            results = loop.run_until_complete(
                query_hipporag(HIPPORAG_URL, query, body.get('num_results', 10))
            )
            loop.close()
            self._respond(200, {'results': results})

        else:
            self._respond(404, {'error': 'not found'})

    def do_GET(self):
        if self.path == '/health':
            self._respond(200, {'status': 'ok', 'mode': 'tgs-rag'})
        else:
            self._respond(404, {'error': 'not found'})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        out = json.dumps(data, default=str)
        self.wfile.write(out.encode())

    def log_message(self, fmt, *args):
        logger.info(fmt % args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=11236)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--hipporag', default=HIPPORAG_URL)
    parser.add_argument('--memory-db', default=MEMORY_DB)
    args = parser.parse_args()

    HIPPORAG_URL = args.hipporag
    MEMORY_DB = args.memory_db

    logger.info(f'TGS-RAG Bridge on {args.host}:{args.port}')
    logger.info(f'HippoRAG: {HIPPORAG_URL} | Memory DB: {MEMORY_DB}')
    logger.info(f'alpha={ALPHA} orphan_cap={ORPHAN_CAP} epsilon={EPSILON}')
    HTTPServer((args.host, args.port), TGSHandler).serve_forever()
