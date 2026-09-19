# Research-Pod Tool Contract

The next Pod type gets a typed tool manifest before model training. The manifest currently contains `ann_search`, `bm25_search`, `metadata_filter`, `graph_neighbors`, `registry_snapshot`, and `calculator`, plus `workspace_search` for bounded streaming literal/regex search with file-type and context controls. Each call has explicit required/optional arguments and carries namespace, Pod identity, generation, principal, and trace context.

The runtime validates the contract before dispatch. A generation cannot be supplied without a stable Pod identity, and retrieval tools require a namespace. This keeps tool use compatible with the lifecycle barrier and makes tool-choice SFT measurable.

Training data should contain positive and negative tool-choice examples: lexical lookup vs ANN, metadata filtering before retrieval, graph traversal for multi-hop questions, snapshot validation before using a Pod, and calculator use for numeric answers.

The workspace tool follows ripgrep-inspired guardrails: explicit path/type scope, bounded results, Unicode-safe text handling, and streaming search rather than loading an entire corpus. It is a contract only; execution must still enforce the Pod ACL and approved roots.
