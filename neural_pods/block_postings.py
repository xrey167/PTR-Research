"""Compact fixed-size posting lists for high-cardinality filters."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class PostingBlock:
    first:int; last:int; values:bytes; count:int

def _enc(vals):
 out=bytearray(); prev=0
 for v in vals:
  d=v-prev
  while d>=128: out.append((d&127)|128); d >>= 7
  out.append(d); prev=v
 return bytes(out)
def _dec(buf):
 vals=[]; cur=0; shift=0
 for b in buf:
  cur |= (b&127)<<shift
  if b<128:
   vals.append(cur if not vals else vals[-1]+cur); cur=0; shift=0
  else: shift += 7
 return vals
class BlockPostings:
 def __init__(self,block_size=256):
  if block_size<32: raise ValueError('block_size too small')
  self.block_size=block_size; self._blocks={}
 def add(self,term,doc_ids):
  vals=sorted(set(int(x) for x in doc_ids)); self._blocks[term]=tuple(PostingBlock(v[0],v[-1],_enc(v),len(v)) for v in (vals[i:i+self.block_size] for i in range(0,len(vals),self.block_size)) if v)
 def union(self,terms):
  out=set()
  for term in terms:
   for block in self._blocks.get(term,()): out.update(_dec(block.values))
  return sorted(out)
 def stats(self):
  blocks=[b for v in self._blocks.values() for b in v]; raw=sum(b.count*8 for b in blocks); packed=sum(len(b.values) for b in blocks); return {'terms':len(self._blocks),'blocks':len(blocks),'postings':sum(b.count for b in blocks),'raw_bytes':raw,'packed_bytes':packed,'compression_ratio':raw/packed if packed else 1.0}


class ClusterPostings:
    """FTS-v1-style postings partitioned by external vector-cluster ID."""
    def __init__(self): self._parts={}
    def add(self,term,cluster_ids,doc_ids):
        parts={}
        for c,d in zip(cluster_ids,doc_ids): parts.setdefault(int(c),[]).append(int(d))
        self._parts[term]={c:PostingBlock(v[0],v[-1],_enc(sorted(set(v))),len(set(v))) for c,v in parts.items() if v}
    def union(self,terms):
        out=set()
        for t in terms:
            for b in self._parts.get(t,{}).values(): out.update(_dec(b.values))
        return sorted(out)
    def stats(self):
        blocks=[b for v in self._parts.values() for b in v.values()]; raw=sum(b.count*8 for b in blocks); packed=sum(len(b.values) for b in blocks); return {'terms':len(self._parts),'blocks':len(blocks),'postings':sum(b.count for b in blocks),'raw_bytes':raw,'packed_bytes':packed,'compression_ratio':raw/packed if packed else 1.0}

