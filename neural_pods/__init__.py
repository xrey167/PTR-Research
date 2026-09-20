"""Provenance-controlled neural memory. Research prototype.

The re-exports below are grouped by the layer each module belongs to
(`neural_pods.architecture.LAYER_OF`, which is the layer model of
ARCHITECTURE-MASTER section 1). They used to be interleaved with a dozen
`__all__ +=` statements in import order, so the public surface of the
package could only be read by executing it.

Optional heavy dependencies (torch, numpy) stay optional: the modules that
need them are imported in a try/except and their names resolve to None when
the dependency is absent, so metadata-, routing- and transport-only
processes keep working.
"""

# --- layer 1: kern - registry, types, contracts, consensus, serving --------
from .execution_manifest import ExecutionManifest
from .fault_injection import FailAction, FailureInjector, InjectedFailure
from .placement import (FencedLeader, LeaderLease, PlacementDriver,
                        RegionPlacement, ReplicaNode)
from .pod_builder import PodBuilder, PodMetadata, resolve_pod_identity
from .pod_profiles import PROFILES, apply_profile, profile_defaults
from .pod_taxonomy import PodDescriptor, route_candidates
from .pod_types import PodHeader, PodType, typed_artifact
from .raft_backend import DurableRaftNode, RaftNode, raft_binding_available
from .raft_transport import (PersistentRaftClient, PersistentRaftServer,
                             RaftFrameError, RaftPeer, decode_raft_frame,
                             encode_raft_frame)
from .replication import PostgresReplica, QuorumReplicator, ReplicaResult
from .resource_runtime import (GPUInfo, HardwareSnapshot, RequestReceipt,
                               RequestTracker, ResidencyLease,
                               ResourceAdmissionError, ResourceBoundHandler,
                               ResourceBudget, ResourceGovernor, ResourceLease,
                               probe_hardware)
from .vllm_router import VllmReplica, VllmReplicaRouter, VllmRouterMetrics

# --- layer 2: storage - the four tiers and what indexes into them ----------
from .alias_resolver import AliasResolver
from .block_postings import BlockPostings, PostingBlock
from .contextual_retrieval import (contextual_document, contextual_query,
                                   document_context)
from .local_search import LocalSearchBackend, SearchHit
from .pod_cache import PodCache
from .postgres_store import PostgresNamespaceStore
from .ranking import TensorRankProfile, onnx_ranker
from .registry_lookup_kb import RegistryLookupKB
from .snapshot_store import SnapshotRef, SnapshotStore
from .spacy_enrichment import SpacySemanticEnricher

# --- layer 4: nervensystem - protocol, sockets, streams, batching ----------
from .adaptive_batcher import AdaptiveBatcher, BatchRequest, BatchedPodHandler
from .pod_protocol import PodFanout, PodRequest, PodResponse, PodTransport
from .pod_socket import PodSocketClient, PodSocketServer, PodSocketSession
from .pod_streams import (DuplexSession, HypothesisBranch, MergedPodResult,
                          PodEvent, merge_branches)

# --- layer 5: pod-arm - reflex, routing, agentic search --------------------
from .ngu_sampling import NGUItem, never_give_up
from .research_tools import (RESEARCH_TOOLS, ResearchTool, ResearchToolCall,
                             research_tool_manifest)
from .search_agent import LocalSearchAgent, SearchAction, SearchEpisode
from .symlink import TemporalPortPlane

# --- optional heavy dependencies ------------------------------------------
try:
    from .lifecycle_transport import (TransportedLifecycleToken, apply_token,
                                      compose, transported_delete_token,
                                      validate_registry_binding)
except ImportError:  # torch is optional for metadata/routing-only deployments
    TransportedLifecycleToken = apply_token = compose = None
    transported_delete_token = validate_registry_binding = None
try:
    from .continuous_concepts import ContinuousConceptMixer, DecoderConceptInjector
except ImportError:  # torch is optional for routing/contract-only processes
    ContinuousConceptMixer = DecoderConceptInjector = None
try:
    from .vector_index import HNSWVectorIndex, PersistentVectorIndex
except ImportError:  # the optional NumPy tier
    PersistentVectorIndex = HNSWVectorIndex = None

__all__ = [
    # layer 1
    "ExecutionManifest",
    "FailAction", "FailureInjector", "InjectedFailure",
    "FencedLeader", "LeaderLease", "PlacementDriver", "RegionPlacement",
    "ReplicaNode",
    "PodBuilder", "PodMetadata", "resolve_pod_identity",
    "PodDescriptor", "route_candidates",
    "PodHeader", "PodType", "typed_artifact",
    "DurableRaftNode", "RaftNode", "raft_binding_available",
    "PersistentRaftClient", "PersistentRaftServer", "RaftFrameError",
    "RaftPeer", "decode_raft_frame", "encode_raft_frame",
    "PostgresReplica", "QuorumReplicator", "ReplicaResult",
    "GPUInfo", "HardwareSnapshot", "RequestReceipt", "RequestTracker",
    "ResidencyLease", "ResourceAdmissionError", "ResourceBoundHandler",
    "ResourceBudget", "ResourceGovernor", "ResourceLease", "probe_hardware",
    "VllmReplica", "VllmReplicaRouter", "VllmRouterMetrics",
    # layer 2
    "LocalSearchBackend", "SearchHit",
    "PodCache",
    "TensorRankProfile", "onnx_ranker",
    "SnapshotRef", "SnapshotStore",
    # layer 4
    "AdaptiveBatcher", "BatchRequest", "BatchedPodHandler",
    "DuplexSession", "HypothesisBranch", "MergedPodResult", "PodEvent",
    "merge_branches",
    # layer 5
    "LocalSearchAgent", "SearchAction", "SearchEpisode",
    "TemporalPortPlane",
    # optional
    "TransportedLifecycleToken", "apply_token", "compose",
    "transported_delete_token", "validate_registry_binding",
    "ContinuousConceptMixer", "DecoderConceptInjector",
]
