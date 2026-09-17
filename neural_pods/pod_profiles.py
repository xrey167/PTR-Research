"""Preset Pod profiles for common knowledge domains."""
from __future__ import annotations
from copy import deepcopy
PROFILES={
 'private': {'domain':'private','tags':['scope:private'],'sensitivity':'high','acl':['owner'],'cache_namespace':'private'},
 'hobby': {'domain':'hobby','tags':['scope:hobby'],'sensitivity':'normal','acl':['owner'],'cache_namespace':'hobby'},
 'beruf': {'domain':'work','tags':['scope:work','role:knowledge'],'sensitivity':'confidential','acl':['owner','workgroup'],'cache_namespace':'work'},
 'programmierung': {'domain':'programming','tags':['scope:programming','role:technical'],'sensitivity':'normal','acl':['owner','team'],'cache_namespace':'programming'},
}
def profile_defaults(name:str)->dict:
 key=name.casefold().strip()
 if key not in PROFILES: raise KeyError(f'unknown Pod profile: {name}')
 return deepcopy(PROFILES[key])
def apply_profile(metadata:dict,name:str)->dict:
 out=profile_defaults(name); out.update(metadata); out['tags']=sorted(set(profile_defaults(name)['tags'])|set(metadata.get('tags',[]))); return out
