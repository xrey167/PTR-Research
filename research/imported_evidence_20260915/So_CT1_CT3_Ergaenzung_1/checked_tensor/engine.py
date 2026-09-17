"""Fixed tensor expression language; no arbitrary code execution facility."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

ABI = 'ct1/fp64/c-order/numpy-' + np.__version__


class Invalid(Exception):
    pass


def canon(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(x).hexdigest()


def tensor_digest(x):
    if not isinstance(x,np.ndarray) or x.dtype!=np.dtype('<f8') or x.ndim!=2:
        raise Invalid('rank-2 little-endian FP64 required')
    if any(d<1 or d>4096 for d in x.shape) or x.size>1048576 or not np.isfinite(x).all():
        raise Invalid('tensor bound/nonfinite')
    return digest(canon({'shape':list(x.shape),'dtype':'<f8'})+x.tobytes(order='C'))


def snapshot(tensors,epochs,namespace='research/ct1'):
    if set(tensors)!=set(epochs):raise Invalid('epochs')
    if any(type(v) is not int or v<1 for v in epochs.values()):raise Invalid('epoch')
    records={k:{'epoch':epochs[k],'tensor_digest':tensor_digest(v)} for k,v in tensors.items()}
    body={'namespace':namespace,'inputs':records}
    return body,digest(canon(body))


def run(program,tensors):
    if not isinstance(program,list) or not 1<=len(program)<=128:raise Invalid('program bound')
    values=[];deps=[]
    for i,node in enumerate(program):
        if not isinstance(node,dict):raise Invalid('node')
        op=node.get('op')
        if op=='input':
            if set(node)!={'op','name'} or node['name'] not in tensors:raise Invalid('leaf')
            name=node['name'];x=tensors[name];tensor_digest(x)
            values.append(x.copy());deps.append({name});continue
        if op not in ('add','sub','matmul') or set(node)!={'op','a','b'}:raise Invalid('operation')
        a,b=node['a'],node['b']
        if type(a) is not int or type(b) is not int or not (0<=a<i and 0<=b<i):raise Invalid('reference')
        x,y=values[a],values[b]
        if op=='matmul':
            if x.shape[1]!=y.shape[0] or x.shape[0]*y.shape[1]>1048576:raise Invalid('shape')
            z=x@y
        else:
            if x.shape!=y.shape:raise Invalid('shape')
            z=x+y if op=='add' else x-y
        tensor_digest(z);values.append(z);deps.append(deps[a]|deps[b])
    # Require every declared expression input to contribute to the output.
    leaves={n['name'] for n in program if n.get('op')=='input'}
    if leaves!=deps[-1]:raise Invalid('dead leaf')
    return values[-1],sorted(deps[-1])


def build(program,tensors,epochs):
    body,root=snapshot(tensors,epochs)
    output,reads=run(program,tensors)
    if set(reads)!=set(tensors):raise Invalid('unused snapshot inputs')
    record={'schema':'ct1','abi':ABI,'program_digest':digest(canon(program)),
      'snapshot_root':root,'input_manifest':body,'reads':reads,'output_digest':tensor_digest(output)}
    return output,record


def verify(program,tensors,epochs,output,evidence):
    try:
        actual,record=build(program,tensors,epochs)
        return canon(record)==canon(evidence) and tensor_digest(output)==record['output_digest'] and np.array_equal(actual,output)
    except (Invalid,ValueError,KeyError,TypeError,OverflowError):
        return False


def transport_program():
    return [dict(op='input',name=n) for n in ('B','S','R','C','Y','X')]+[
      dict(op='sub',a=4,b=5),dict(op='matmul',a=3,b=6),
      dict(op='add',a=0,b=1),dict(op='add',a=8,b=2),dict(op='add',a=9,b=7)]
