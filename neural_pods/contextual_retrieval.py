"""Contextual Retrieval helpers: contextual text for dense and lexical retrieval."""
from __future__ import annotations
from typing import Any, Mapping

def document_context(metadata: Mapping[str,Any]) -> str:
    tags=' '.join(sorted(str(x) for x in metadata.get('tags',[]) if x))
    entities=' '.join(sorted(str(x) for x in metadata.get('entity_ids',metadata.get('entities',[])) if x))
    return ' '.join(x for x in [f"pod_type:{metadata.get('pod_type',metadata.get('type',''))}",f"domain:{metadata.get('domain','')}",f"role:{metadata.get('semantic_role','')}",f"entities:{entities}",f"tags:{tags}",f"knowledge_key:{metadata.get('knowledge_key','')}"] if x.split(':',1)[-1])

def contextual_document(text: str, metadata: Mapping[str,Any]) -> str:
    return f"Context: {document_context(metadata)}\nDocument: {text}"

def contextual_query(text: str, *, pod_type: str|None=None, domain: str|None=None, tags=(), entities=()) -> str:
    meta={'pod_type':pod_type or '','domain':domain or '','tags':tags,'entities':entities}
    return f"Context: {document_context(meta)}\nQuery: {text}"
