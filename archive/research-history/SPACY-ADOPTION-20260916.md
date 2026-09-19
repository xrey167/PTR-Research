# spaCy adoption

spaCy provides pipeline components for tokenization, sentence boundaries, NER, entity linking, dependency parsing, matchers and efficient `Doc`/`DocBin` containers. We use it as an optional semantic enrichment layer. Its entity mentions and sentence segmentation are stored as soft annotations for retrieval only; canonical identity, provenance, ACL, validity and revocation remain registry-owned. This follows spaCy's single-source `Doc` annotation model and composable pipeline architecture.

The implementation in `neural_pods/spacy_enrichment.py` uses a loaded spaCy pipeline when available and a deterministic fallback otherwise. It never writes lifecycle authority.

## InMemoryLookupKB mapping

spaCy's InMemoryLookupKB uses exact alias candidates and priors. RegistryLookupKB mirrors that lookup shape but resolves every candidate through the Registry head/snapshot first, so revoked or stale generations are omitted. Alias candidates remain advisory; the registry remains authoritative.

## Contextual Retrieval

neural_pods/contextual_retrieval.py builds deterministic context prefixes from typed metadata for contextual embeddings and BM25. Raw document text remains in the record; context is an indexed retrieval representation only.
