"""Repair grouped attention mask expansion without changing weights or I/O."""
import copy,hashlib,json,math,mmap,pathlib
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.tools import flatbuffer_utils as fbu

def rewrite(model):
 reports=[]
 reshape_code=next(i for i,c in enumerate(model.operatorCodes) if c.builtinCode==s.BuiltinOperator.RESHAPE)
 for gi,g in enumerate(model.subgraphs):
  mask_ops=[op for op in g.operators if model.operatorCodes[op.opcodeIndex].builtinCode==s.BuiltinOperator.CONCATENATION
            and len(op.inputs)>1 and len(set(op.inputs))==1 and g.tensors[op.inputs[0]].name.endswith(b'_mask_global')]
  if not mask_ops:continue
  replacements={};removed=set();shape_tensors={}
  def temporary(index,shape,suffix):
   t=copy.deepcopy(g.tensors[index]);t.name=t.name+suffix;t.shape=list(shape);t.shapeSignature=None;t.buffer=0
   g.tensors.append(t);return len(g.tensors)-1
  def reshape(src,dst,shape):
   key=tuple(shape)
   if key not in shape_tensors:
    b=s.BufferT();b.data=np.asarray(shape,dtype=np.int32).view(np.uint8);model.buffers.append(b)
    t=s.TensorT();t.name=('mask_broadcast_shape_'+str(len(shape_tensors))).encode();t.type=s.TensorType.INT32
    t.shape=[len(shape)];t.buffer=len(model.buffers)-1;g.tensors.append(t);shape_tensors[key]=len(g.tensors)-1
   op=s.OperatorT();op.opcodeIndex=reshape_code;op.inputs=[src,shape_tensors[key]];op.outputs=[dst]
   op.builtinOptionsType=s.BuiltinOptions.ReshapeOptions;op.builtinOptions=s.ReshapeOptionsT();op.builtinOptions.newShape=list(shape)
   return op
  for mask_op in mask_ops:
   mask_in,mask_out=mask_op.inputs[0],mask_op.outputs[0];factor=len(mask_op.inputs)
   tin,tout=g.tensors[mask_in],g.tensors[mask_out]
   if tin.type!=s.TensorType.INT16 or tout.type!=tin.type or mask_op.builtinOptions.axis!=2 or mask_op.builtinOptions.fusedActivationFunction:
    raise ValueError('Unsupported mask concat form')
   if not np.array_equal(tin.quantization.scale,tout.quantization.scale) or not np.array_equal(tin.quantization.zeroPoint,tout.quantization.zeroPoint):
    raise ValueError('Mask concat requantizes its input')
   consumers=[op for op in g.operators if mask_out in op.inputs]
   if not consumers:raise ValueError('Mask has no consumer')
   for add in consumers:
    if model.operatorCodes[add.opcodeIndex].builtinCode!=s.BuiltinOperator.ADD or len(add.inputs)!=2:
     raise ValueError('Unsupported mask consumer')
    score=next(v for v in add.inputs if v!=mask_out);destination=add.outputs[0]
    old_shape=list(g.tensors[score].shape)
    if len(old_shape)!=4 or old_shape[2]%factor:raise ValueError('Unsupported grouped query shape')
    new_shape=[old_shape[0],old_shape[1]*factor,old_shape[2]//factor,old_shape[3]]
    if list(tin.shape)!=[1,1,new_shape[2],new_shape[3]] or math.prod(old_shape)!=math.prod(new_shape):
     raise ValueError('Mask broadcast dimensions differ')
    score_view=temporary(score,new_shape,b'_explicit_query_heads')
    sum_view=temporary(destination,new_shape,b'_explicit_query_heads')
    new_add=copy.deepcopy(add);new_add.inputs=[score_view,mask_in];new_add.outputs=[sum_view]
    replacements[id(add)]=[reshape(score,score_view,new_shape),new_add,reshape(sum_view,destination,old_shape)]
   removed.add(id(mask_op))
   reports.append({'subgraph':gi,'mask_input':tin.name.decode(),'groups':factor,'rewritten_adds':len(consumers)})
  g.operators=[part for op in g.operators if id(op) not in removed for part in replacements.get(id(op),[op])]
 return reports

def buffer_bytes(buffer):
 return memoryview(buffer.data).cast('B') if buffer.data is not None else memoryview(b'')

def write_large(model,path):
 external={}
 metadata_buffers={v.buffer for v in (model.metadata or [])}
 for i,b in enumerate(model.buffers):
  data=buffer_bytes(b)
  if data.nbytes>=1024*1024 and i not in metadata_buffers:
   external[i]=data;b.data=None;b.offset=1;b.size=data.nbytes
 header=fbu.convert_object_to_bytearray(model)
 packed=s.Model.GetRootAsModel(header,0)
 position=(len(header)+63)//64*64;offsets={}
 for i,data in external.items():
  offsets[i]=position;fbu.update_packed_buffer(packed.Buffers(i),offset=position,size=data.nbytes)
  position=(position+data.nbytes+63)//64*64
 with path.open('wb') as f:
  f.write(header)
  for i,data in external.items():
   f.write(b'\0'*(offsets[i]-f.tell()));f.write(data)

def repair(source,destination,report_path):
 if destination.exists():raise ValueError('Refusing to overwrite a prior repair')
 model=fbu.read_model(str(source))
 previous=[{'bytes':buffer_bytes(b).nbytes,'sha256':hashlib.sha256(buffer_bytes(b)).hexdigest()} for b in model.buffers]
 changes=rewrite(model)
 if not changes:raise ValueError('No supported mask expansion found')
 write_large(model,destination)
 with destination.open('rb') as f:
  data=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);m=s.Model.GetRootAsModel(data,0)
  for i,expected in enumerate(previous):
   b=m.Buffers(i);offset=b._tab.Vector(b._tab.Offset(4)) if b.DataLength() else b.Offset();size=b.DataLength() or b.Size()
   actual=hashlib.sha256(memoryview(data)[offset:offset+size]).hexdigest()
   if size!=expected['bytes'] or actual!=expected['sha256']:raise ValueError('Original model buffer changed: '+str(i))
 with destination.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 report={'changes':changes,'original_buffers_verified':len(previous),'all_original_buffer_bytes_preserved':True,
         'sha256':digest,'bytes':destination.stat().st_size,'scope':'Graph rewrite; functional validation is separate.'}
 report_path.write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--report',type=pathlib.Path,required=True)
 a=p.parse_args();print(json.dumps(repair(a.input,a.output,a.report)),flush=True)
