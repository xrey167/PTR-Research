"""Optional PostgreSQL namespace store for production-scale Pods.

The module does not require psycopg at import time. It provides schema and
parameterized operations; lifecycle authorization remains in the Registry.
"""
from __future__ import annotations
try:
 from psycopg.types.json import Jsonb
except ImportError:
 Jsonb = lambda value: value
SCHEMA='''
CREATE TABLE IF NOT EXISTS pod_items(namespace text NOT NULL, branch text NOT NULL DEFAULT 'main', item_key text NOT NULL, text_value text NOT NULL, vector jsonb, metadata jsonb NOT NULL, deleted boolean NOT NULL DEFAULT false, PRIMARY KEY(namespace,branch,item_key));
CREATE INDEX IF NOT EXISTS pod_items_meta_gin ON pod_items USING gin(metadata);
CREATE INDEX IF NOT EXISTS pod_items_namespace_branch ON pod_items(namespace,branch) WHERE deleted=false;
CREATE TABLE IF NOT EXISTS pod_postings(namespace text NOT NULL, branch text NOT NULL, term text NOT NULL, block_no integer NOT NULL, first_doc bigint NOT NULL, last_doc bigint NOT NULL, postings bytea NOT NULL, count integer NOT NULL, PRIMARY KEY(namespace,branch,term,block_no));
CREATE INDEX IF NOT EXISTS pod_postings_term ON pod_postings(namespace,branch,term);
CREATE TABLE IF NOT EXISTS pod_namespaces(namespace text PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(), revision bigint NOT NULL DEFAULT 0, read_only boolean NOT NULL DEFAULT false, schema_json jsonb NOT NULL DEFAULT '{}'::jsonb);
CREATE TABLE IF NOT EXISTS pod_branches(namespace text NOT NULL, branch text NOT NULL, parent text, PRIMARY KEY(namespace,branch));
CREATE INDEX IF NOT EXISTS pod_branches_parent ON pod_branches(namespace,parent);
'''
class PostgresNamespaceStore:
 def __init__(self, connection): self.connection=connection
 def initialize(self):
  with self.connection.cursor() as cur:
   cur.execute(SCHEMA)
   cur.execute("INSERT INTO pod_namespaces(namespace) VALUES(%s) ON CONFLICT DO NOTHING", ('default',))
   cur.execute("INSERT INTO pod_branches(namespace,branch) VALUES(%s,%s) ON CONFLICT DO NOTHING", ('default','main'))
  self.connection.commit()
 def upsert(self, namespace, key, text, metadata, branch='main', vector=None):
  with self.connection.cursor() as cur:
   cur.execute("INSERT INTO pod_namespaces(namespace) VALUES(%s) ON CONFLICT DO NOTHING", (namespace,))
   cur.execute("INSERT INTO pod_branches(namespace,branch) VALUES(%s,%s) ON CONFLICT DO NOTHING", (namespace,branch))
   cur.execute('''INSERT INTO pod_items(namespace,branch,item_key,text_value,vector,metadata,deleted) VALUES(%s,%s,%s,%s,%s,%s,false) ON CONFLICT(namespace,branch,item_key) DO UPDATE SET text_value=EXCLUDED.text_value,vector=EXCLUDED.vector,metadata=EXCLUDED.metadata,deleted=false''',(namespace,branch,key,text,Jsonb(vector) if vector is not None else None,Jsonb(metadata)))
   cur.execute("UPDATE pod_namespaces SET updated_at=now(),revision=revision+1 WHERE namespace=%s", (namespace,))
  self.connection.commit()
 def delete(self, namespace, key, branch='main'):
  with self.connection.cursor() as cur:
   cur.execute("UPDATE pod_items SET deleted=true WHERE namespace=%s AND branch=%s AND item_key=%s", (namespace,branch,key))
   cur.execute("UPDATE pod_namespaces SET updated_at=now(),revision=revision+1 WHERE namespace=%s", (namespace,))
  self.connection.commit()
 def branch(self, namespace, source='main', target=None):
  import uuid
  target = target or 'branch-' + uuid.uuid4().hex[:12]
  with self.connection.cursor() as cur:
   cur.execute("INSERT INTO pod_branches(namespace,branch,parent) VALUES(%s,%s,%s)", (namespace,target,source))
  self.connection.commit(); return target
 def namespace_metadata(self, namespace):
  with self.connection.cursor() as cur:
   cur.execute("SELECT created_at,updated_at,revision,read_only,schema_json FROM pod_namespaces WHERE namespace=%s", (namespace,)); row=cur.fetchone()
   if row is None: raise KeyError(namespace)
   cur.execute("SELECT branch,parent FROM pod_branches WHERE namespace=%s", (namespace,)); branches=cur.fetchall()
  return {'created_at': row[0].isoformat() if hasattr(row[0], 'isoformat') else row[0], 'updated_at': row[1].isoformat() if hasattr(row[1], 'isoformat') else row[1], 'revision': row[2], 'read_only': row[3], 'schema': row[4], 'branches': [{'branch': x[0], 'parent': x[1]} for x in branches]}
 def close(self): self.connection.close()
