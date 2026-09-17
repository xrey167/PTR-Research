"""Optional spaCy-compatible semantic enrichment for Pod retrieval."""
from __future__ import annotations
import re
from typing import Any
class SpacySemanticEnricher:
    def __init__(self, model: str | None = None, nlp=None):
        self.backend='fallback'
        if nlp is not None: self.nlp=nlp; self.backend='spacy'
        else:
            try:
                import spacy
                self.nlp=spacy.load(model or 'en_core_web_sm'); self.backend='spacy'
            except Exception: self.nlp=None
    def annotate(self,text:str)->dict[str,Any]:
        if not isinstance(text,str) or not text.strip(): return {'backend':self.backend,'entities':[],'sentences':[]}
        if self.nlp is not None:
            doc=self.nlp(text); entities=[{'text':e.text,'label':e.label_,'start':e.start_char,'end':e.end_char} for e in doc.ents]; sentences=[s.text for s in doc.sents]
        else:
            entities=[]
            for m in re.finditer(r'\b[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,3}',text): entities.append({'text':m.group(0),'label':'ENTITY','start':m.start(),'end':m.end()})
            sentences=[x.strip() for x in re.split(r'(?<=[.!?])\s+',text) if x.strip()]
        return {'backend':self.backend,'entities':entities,'sentences':sentences,'soft':True,'warning':'Annotations are advisory and never lifecycle authority.'}
    def enrich_metadata(self,text:str,metadata:dict[str,Any])->dict[str,Any]:
        out=dict(metadata); ann=self.annotate(text); out['semantic_annotations']=ann; out['entity_mentions']=sorted({e['text'] for e in ann['entities']}); return out
