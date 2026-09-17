"""Provenance-controlled neural memory. Research prototype."""

from .local_search import LocalSearchBackend, SearchHit
from .pod_types import PodHeader, PodType, typed_artifact
from .search_agent import LocalSearchAgent, SearchAction, SearchEpisode
from .execution_manifest import ExecutionManifest
from .ranking import TensorRankProfile, onnx_ranker
from .symlink import TemporalPortPlane
try:
    from .lifecycle_transport import (TransportedLifecycleToken, apply_token, compose,
                                      transported_delete_token, validate_registry_binding)
except ImportError:  # torch is optional for metadata/routing-only deployments
    TransportedLifecycleToken = apply_token = compose = transported_delete_token = validate_registry_binding = None
try:
    from .continuous_concepts import ContinuousConceptMixer, DecoderConceptInjector
except ImportError:  # torch is optional for routing/contract-only processes
    ContinuousConceptMixer = DecoderConceptInjector = None
from .pod_taxonomy import PodDescriptor, route_candidates
from .pod_builder import PodBuilder, PodMetadata, resolve_pod_identity
from .pod_cache import PodCache

__all__ = ["LocalSearchBackend", "SearchHit", "PodHeader", "PodType", "typed_artifact",
           "LocalSearchAgent", "SearchAction", "SearchEpisode", "ExecutionManifest",
           "TensorRankProfile", "onnx_ranker"]
__all__.append("TemporalPortPlane")
__all__ += ["TransportedLifecycleToken", "apply_token", "compose", "transported_delete_token",
            "validate_registry_binding"]
__all__ += ["ContinuousConceptMixer", "DecoderConceptInjector"]
__all__ += ["PodDescriptor", "route_candidates"]
__all__ += ["PodBuilder", "PodMetadata", "resolve_pod_identity"]
__all__ += ["PodCache"]
from .pod_streams import HypothesisBranch, MergedPodResult, merge_branches, PodEvent, DuplexSession
from .pod_protocol import PodRequest, PodResponse, PodTransport, PodFanout
try:
    from .vector_index import PersistentVectorIndex, HNSWVectorIndex
except ImportError:
    # Metadata/transport clients should work without the optional NumPy tier.
    PersistentVectorIndex = HNSWVectorIndex = None
from .pod_socket import PodSocketServer, PodSocketClient, PodSocketSession
from .replication import QuorumReplicator, ReplicaResult, PostgresReplica
from .snapshot_store import SnapshotStore, SnapshotRef
__all__ += ["HypothesisBranch", "MergedPodResult", "merge_branches", "PodEvent", "DuplexSession"]
__all__ += ["QuorumReplicator", "ReplicaResult", "PostgresReplica", "SnapshotStore", "SnapshotRef"]

from .spacy_enrichment import SpacySemanticEnricher

from .registry_lookup_kb import RegistryLookupKB

from .contextual_retrieval import contextual_document, contextual_query, document_context

from .alias_resolver import AliasResolver

from .block_postings import BlockPostings, PostingBlock

from .postgres_store import PostgresNamespaceStore

from .pod_profiles import PROFILES, profile_defaults, apply_profile
from .ngu_sampling import NGUItem, never_give_up
from .research_tools import RESEARCH_TOOLS, ResearchTool, ResearchToolCall, research_tool_manifest
from .adaptive_batcher import AdaptiveBatcher, BatchRequest, BatchedPodHandler
__all__ += ["AdaptiveBatcher", "BatchRequest", "BatchedPodHandler"]
from .resource_runtime import (GPUInfo, HardwareSnapshot, ResourceBudget,
                               ResourceGovernor, ResourceLease, ResourceAdmissionError,
                               ResourceBoundHandler, ResidencyLease, RequestReceipt,
                               RequestTracker,
                               probe_hardware)
__all__ += ["GPUInfo", "HardwareSnapshot", "ResourceBudget", "ResourceGovernor",
            "ResourceLease", "ResidencyLease", "ResourceAdmissionError", "ResourceBoundHandler",
            "RequestReceipt", "RequestTracker", "probe_hardware"]
from .placement import FencedLeader, LeaderLease, PlacementDriver, RegionPlacement, ReplicaNode
__all__ += ["FencedLeader", "LeaderLease", "PlacementDriver", "RegionPlacement", "ReplicaNode"]
from .vllm_router import VllmReplica, VllmReplicaRouter, VllmRouterMetrics
__all__ += ["VllmReplica", "VllmReplicaRouter", "VllmRouterMetrics"]
from .raft_backend import DurableRaftNode, RaftNode, raft_binding_available
__all__ += ["DurableRaftNode", "RaftNode", "raft_binding_available"]
from .fault_injection import FailAction, FailureInjector, InjectedFailure
__all__ += ["FailAction", "FailureInjector", "InjectedFailure"]
from .raft_transport import (PersistentRaftClient, PersistentRaftServer, RaftFrameError, RaftPeer,
                             decode_raft_frame, encode_raft_frame)
__all__ += ["PersistentRaftClient", "PersistentRaftServer", "RaftFrameError", "RaftPeer",
            "decode_raft_frame", "encode_raft_frame"]

