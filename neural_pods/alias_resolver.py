"""Fast canonical alias resolution before hybrid retrieval."""
from __future__ import annotations
import re
class AliasResolver:
    def __init__(self, aliases=None): self._map={str(k).casefold():str(v) for k,v in (aliases or {}).items()}
    def add(self, alias:str, canonical:str):
        if not alias.strip() or not canonical.strip(): raise ValueError('alias and canonical are required')
        self._map[alias.casefold()]=canonical
    def resolve(self, text:str)->str:
        out=text
        for alias,canonical in sorted(self._map.items(),key=lambda kv:-len(kv[0])):
            out=re.sub(re.escape(alias), canonical, out, flags=re.IGNORECASE)
        return out
    def resolve_with_metadata(self,text:str):
        resolved=self.resolve(text); return {'query':resolved,'resolved':resolved!=text,'aliases':sorted(a for a in self._map if a in text.casefold())}

