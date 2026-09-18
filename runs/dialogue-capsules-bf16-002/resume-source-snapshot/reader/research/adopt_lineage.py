"""Adopt a verified local DAG snapshot into an isolated experiment registry.

Destination becomes the experiment's local authority. Later source changes do
NOT propagate: this is deliberately not a distributed revocation protocol.
Artifact file copying/verification remains the caller's responsibility.
"""
from neural_pods.registry import InvalidState, digest


def adopt_lineage(source, destination, artifact, principal='buyer'):
    if source.db is destination.db:raise ValueError('Use distinct registries')
    source_path=source.db.execute('PRAGMA database_list').fetchone()[2]
    target_path=destination.db.execute('PRAGMA database_list').fetchone()[2]
    if source_path and source_path==target_path:raise ValueError('Use distinct registry files')
    # Freeze the source snapshot while verifying and adopting it. This scope is
    # local SQLite; no claim of a cross-host transaction or signature.
    with source.transaction():
        source._valid(artifact,principal)
        nodes={n['id']:n for n in source.ancestors(artifact)}
        parents={key:sorted(r[0] for r in source.db.execute('SELECT parent FROM edges WHERE child=?',(key,))) for key in nodes}
        for key,node in nodes.items():
            expected=node['kind']+':'+digest({'kind':node['kind'],'payload':node['payload'],'parents':parents[key]})
            if key!=expected:raise InvalidState('Source lineage content hash mismatch')
        ordered=[];done=set();active=set()
        def visit(key):
            if key in done:return
            if key in active:raise InvalidState('Cyclic lineage')
            active.add(key)
            for parent in parents[key]:visit(parent)
            active.remove(key);done.add(key);ordered.append(key)
        visit(artifact)
        with destination.transaction():
            for key in ordered:
                node=nodes[key];payload=node['payload']
                if destination.db.execute('SELECT 1 FROM nodes WHERE id=?',(key,)).fetchone():
                    existing=destination.node(key)
                    existing_parents=sorted(r[0] for r in destination.db.execute('SELECT parent FROM edges WHERE child=?',(key,)))
                    if existing['revoked'] or existing['kind']!=node['kind'] or existing['payload']!=payload or existing_parents!=parents[key]:
                        raise InvalidState('Conflicting destination lineage')
                if destination._add(node['kind'],payload,parents[key])!=key:raise InvalidState('Adoption changed identity')
                if node['kind']=='knowledge':
                    head=destination.db.execute('SELECT node_id,generation FROM heads WHERE knowledge_key=?',(payload['knowledge_key'],)).fetchone()
                    if head and (head['node_id']!=key or head['generation']!=payload['generation']):
                        raise InvalidState('Conflicting destination knowledge head')
                    destination.db.execute('INSERT OR IGNORE INTO heads VALUES(?,?,?)',(payload['knowledge_key'],payload['generation'],key))
            destination._valid(artifact,principal)
            destination._event('adopt_local_snapshot',{'artifact':artifact,'nodes':len(ordered),'policy':'local-authority-copy:no-future-source-sync'})
    return artifact
