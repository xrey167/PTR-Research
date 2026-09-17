"""Registry-backed alias candidate index with spaCy KB-like API."""
from __future__ import annotations
class RegistryLookupKB:
    """Small adapter exposing spaCy's alias/candidate shape without authority bypass."""
    def __init__(self, registry): self.registry=registry; self._aliases={}
    def add_alias(self, alias, knowledge_key, prior=1.0):
        if not alias or not knowledge_key or not 0 <= float(prior) <= 1: raise ValueError('invalid alias candidate')
        self._aliases.setdefault(alias.casefold(), []).append((knowledge_key,float(prior)))
    def get_alias_candidates(self, alias, principal='local'):
        out=[]
        for key,prior in self._aliases.get(alias.casefold(),[]):
            try: node=self.registry.head(key); self.registry.snapshot([node],principal); out.append({'knowledge_key':key,'prior':prior,'generation_key':node})
            except Exception: continue
        return out
    def get_candidates_batch(self, aliases, principal='local'):
        return [self.get_alias_candidates(a,principal) for a in aliases]
