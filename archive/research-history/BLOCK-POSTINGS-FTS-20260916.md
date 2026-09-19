# Fixed-size block postings

The local FTS index supports fixed-size posting blocks (default 256) with delta encoding. `LocalSearchBackend.build_block_postings(namespace, branch, field)` builds a field-specific index for high-cardinality arrays such as permissions and tags. Blocks expose first/last/count metadata for future skip decisions.

Cluster-based partitions remain the ANN index structure; they are not used for BM25 postings. PostgreSQL deployment uses the same logical model in `pod_postings`.
