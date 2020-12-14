import operator
import numpy as np
global_track=True
def list_mul(x,y):
  return list(map(add,x,y))
ndarray=type(np.array([1]))
def mul_back(ob, context):
  ob.grad=ob.grad*context
def return_array(x):
  if isinstance(x,tensor):
    return x.item
  else:
    return x
def relu(x):
  negative=(x.item<0).astype(int)
  return x*negative
def lrelu(x,c):
  inv=abs((x.item<0).astype(int)-1)*c
  negative=(x.item<0).astype(int)
  return x*(negative+inv)
def lreu_c_back(ob,context):
  ob.grad=ob.grad*(context*(abs((context.item<0).astype(int)-1)))
def relu_back(ob,context):
  ob.grad=ob.grad*(context.item<0).astype(int)
def lrelu_back(ob,context):
  inv=abs((context.item<0).astype(int)-1)*c
  negative=(context.item<0).astype(int)
  ob.grad=ob.grad*(inv+negative)
def mm(a,b,do_grad=False):
  result=tensor([[sum(a[i]*b[:,j]) for j in range(b.shape[1])] for i in range(a.shape[0])])
def sum_to(tens,axis,new_dim):
  idx=[None for i in range(len(tens.shape))]
  idx[new_dim]=slice(None)
def bmm_back(ob,context):
  ob.grad=bmm(ob.grad,np.stack([i.T for i in context]))
def rbmm_back(ob,context):
  ob.grad=bmm(np.stack([i.T for i in context]),ob.grad)
def bmm(a,b):
  result=tensor(np.stack([mm(a[i],b[i]) for i in range(len(a))]))
  if global_track:
    a.grads.update({result.id:(b,bmm_back)})
    b.grads.update({result.id:(a,rbmm_back)})
    result.parents=tuple(filter(lambda x: x.requires_grad,(a,b)))
    result.requires_grad=len(result.parents)>0
    return result
  result.requires_grad=False
  result.parents=()
  return result

def sum_compress(thing,dim,new_length):
  idx=[slice(None) for i in range(len(thing.shape))]
  idx[dim]=slice(0,new_length)
  starting=thing[idx]
  for i in range(thing.shape[dim]//new_length-1):
    start=new_length
    idx=[slice(None) for i in range(len(thing.shape))]
    idx[dim]=slice(start,start+new_length)
    starting+=thing[idx]
    start+=new_length
  if global_track:
    thing.grads.update({id(starting):(thing.shape,resize_back)})
    thing.parents=(thing,)
  else:
    thing.parents=()
  thing.requires_grad=len(thing.parents)>0
  return thing

def resize_back(ob,context):
  for i in range(len(context)):
    if context[i]!=ob.grad.shape[i]:
      ob.grad=sum_compress(ob.grad,i,context[i])

def rmatmul_back(ob, context):
  if isinstance(context,tensor):
    #print(ob.grad.shape,context.item.T.shape)
    ob.grad=mm(context.item.T,ob.grad)
  else:
    ob.grad=mm(context.T,ob.grad)
def matmul_back(ob,context):
  if isinstance(context, tensor):
    ob.grad=mm(ob.grad,context.item.T)
  else:
    ob.grad=mm(ob.grad,context.T)
def pow_back(ob,context):
  ob.grad=ob.grad*(context*ob.item)**(context-1)
def div_back(ob,context):
  ob.grad=ob.grad/context
#TODO transpose backward should transpose all the grads that come after it i think
#reshape backward has to reshape all the new gradients to fit the original tensor
#use numpy or tensor.item when you don't want gradients tracked
def add_back(ob,*args):
    pass
def sigmoid(x):
  if isinstance(x,tensor):
    result=1/(1+np.exp(-x.item))
    x.grads.insert(0,(sig_back,x.item))
    return result
  else:
    return 1/(1+np.exp(x))
def sig_back(ob, context):
  ob.grad=ob.grad*sigmoid(context)*(1-sigmoid(context))

def transpose_back(ob, context):
  ob.grad=ob.grad.T

def reshape_back(ob, context):
  ob.grad=ob.grad.reshape(*context)
def mseloss_back(ob,context):
  ob.grad=ob.grad*context
def mseloss(x,y):
  #print(x,'x')
  #print(y,'y')
  result=tensor(x.item-y.item)
  x.grads.insert(0,(mseloss_back,result))
  result.parents=(x,y)
  return result

def handle(ob, other, thing, parents:tuple, grads=None,do_inherit=False):
  if do_inherit:
    result=tensor(thing,parents=parents)
    if grads is not None:
      ob.grads.insert(0,(grads,other))
  else:
    return tensor(thing)
def bmm_back(ob,context):
  for i in range(len(a)):
    result.append(mm(a[i],b[i]))
  return result
def rdiv_back(ob,context):
  num,denom=context
  ob.grad=num*-(1/(denom**2))*ob.grad
def div_grad(ob,context):
  ob.grad=ob.grad/context

###############################
class placeholder_list:
  def __init__(self,data=()):
    self.item=data
  def append(self,*args):
    pass
  def __add__(self,other):
    pass
  def __radd__(self,other):
    pass
  def __iadd__(self,other):
    pass
  def __sub__(self,other):
    pass
  def __repr__(self):
    return "placeholder_list"
  def __iter__(self):
    return self.item.__iter__()
  def insert(self,*args):
    pass
  def update(self,*args):
    pass

def create_container(data=[],mutable=True):
  return data if mutable else placeholder_list(data)
def find_needy(*tensors):
  ldict={tensors[i]:i for i in range(len(tensors))}
  result=[]
  for i in filter(lambda x: x.requires_grad,tensors):
    result.append([i,ldict[i]])
  return result
def get_needs_grad(tensors):
  return tuple(filter(lambda x: isinstance(x,tensor) and x.requires_grad,tensors))

def handle_op(result, *others, grads=[]):
  if global_track:
    #gradsc=grads*((len(others)-len(grads))*2)
    parents=get_needs_grad(others)
    out=tensor(result,parents=parents,requires_grad=len(parents)>0)
    for i in range(len(grads)):
      others[i].grads.update({out.id:grads[i]})
  else:
    tensor(result,parents=(),requires_grad=False)
  
def mseloss(pred,yhat):
  assert pred.shape==yhat.shape
  if len(pred.shape)>2:
    result=(pred.item-yhat.item)/pred.shape[0]
  else:
    result=pred.item-yhat.item
  return result
#order matters grads
matmul_grads=[matmul_back,rmatmul_back]
div_grads=[div_back,rdiv_back]
native_python_types=[int,float,type(1j)]

#TODO turn grads attribute to dictionary with {id(child):grad}

class tensor:
  def __init__(self, data, requires_grad=True, parents=()):
    self.requires_grad=requires_grad
    if not global_track:
      self.requires_grad=False
    self.grad=0
    self.gradv=0
    self.grads=create_container(data={},mutable=requires_grad)
    self.tensors=[]
    self.parents=parents
    #self.children=[]
    self.item=np.array(data) if not isinstance(data,tensor) else data.item
    self.shape=self.item.shape
    self.T=self.item.T
    self.id=id(self)
  def size(self):
    return self.item.shape
  def col(self,col_num):
    return [i[col_num] for i in self.item]
  def __mul__(self, other):
    print('mul')
    if type(other) in native_python_types:
      return handle_op(self.item*other,self,grads=[(mul_back,other)])
    return handle_op(self.item*other.item,self,other,grads=[(mul_back,other.item),(mul_back,self.item)])
  def __rmul__(self,other):
    '''this is only required when multiplying with non tensors which you shouldn`t do'''
    print('rmul')
    return handle_op(other.item*self.item,self,grads=[(mul_back,other)])
  def __add__(self,other):
    if not isinstance(other,tensor):
      return handle_op(self.item*other,self,grads=[])
    return handle_op(self.item*other.item,self,other,grads=[])
  def backward(self,incoming_grads=[]):#incoming grads should be list of tuples
    self.grads+=incoming_grads
    '''print('################')
    print(self)
    print(self.grads)
    print('################')'''
    if len(self.grads)>0:
      self.grad=self.grads[-1][1]
    self.grad=self.grads[-1][1]
    for i,j in reversed(self.grads[:-1]):
      i(self, j)
    for parent in self.parents:
      #print(parent)
      parent.backward(self.grads)
    if self.grad is not None:
      self.gradv+=self.grad
  def backward2(self,incoming_grad=None,child_id=None):
    self.grad=None
    if incoming_grad is not None:
      self.grad=incoming_grad
      context,func=self.grads[child_id]
      func(self,context)
    elif len(self.grads)>0:
      self.grad=self.grads.values()[0][-1]
    for parent in self.parents:
      parent.backward2(self.grad,self.id)
    if self.grad is not None:
      self.gradv+=self.grad
  def t(self):
    if self.requires_grad:
      self.grads.insert(0,(transpose_back,1))
    return tensor(self.item.T)
  def __div__(self, other):
    if not isinstance(other,tensor):
      return handle_op(self.item/other,self,grads=[(other, div_back)])
    return handle_op(self.item/other.item,self,other,grads=[(other, div_back),(self,rdiv_back)])
  def __sub__(self,other):
    if not isinstance(other,tensor):
      return handle_op(self.item-other,self,grads=[(other, div_back)])
    return handle_op(self.item-other.item,self,other,grads=[(other, div_back),(self,rdiv_back)])
  def __rsub__(self,other):
    if not isinstance(other,tensor):
      return handle_op(other-self.item,self,grads=[(other, add_back)])
    return handle_op(other.item-self.item,self,other,grads=[(other,add_back),(self,add_back)])
  def __radd__(self,other):
    if not isinstance(other,tensor):
      return handle_op(other+self.item,self,grads=[(other, add_back)])
    return handle_op(other.item+self.item,self,other,grads=[(other,add_back),(self,add_back)])
  def __matmul__(self,other):
    if not isinstance(other,tensor):
      return handle_op(mm(self.item,other),self,grads=[(other, matmul_back)])
    return handle_op(mm(self.item,other.item),self,other,grads=[(other, matmul_back),(self,rmatmul_back)])
  def __rmatmul__(self,other):
    return handle_op(mm(other, self.item),self,grads=[(other, rmatmul_back)])
  def __pow__(self, other):
    if not isinstance(other,tensor):
      return handle_op(self.item**other,self,grads=[(other, pow_back)])
    return handle_op(self.item**other.item,self,other,grads=[(other, pow_back),(self,rpow_back)])
  def __getitem__(self,i):
    return self.item[i]
  def wipe(self):
    self.grad=None
    self.tensors=[]
    self.gradv=0
    self.grads=[]
    self.parents=()
  def toggle_grad(self):
    self.requires_grad=not self.requires_grad
    self.grads=create_container(self.requires_grad, self.grads)
    return self.requires_grad
  def deactiveate(self):
    return tensor(self.item,requires_grad=False)
  def deactivate_(self):
    self.grads=placeholder_list()
    self.requires_grad=False
  def pause(self):
    self.grads=placeholder_list(self.grads)
    self.requires_grad=False
  def resume(self):
    self.grads=self.grads.data
    self.requires_grad=True
  def reshape(self,*new_shape):
    result=tensor(self.item.reshape(*new_shape),parents=(self),requries_grad=self.requires_grad)
    self.grads.update({result.id:(self.shape,reshape_back)})
  def __str__(self):
    return str(self.item)
  def __repr__(self):
    return repr(self.item)
  def __iter__(self):
    return self.item.__iter__()
  def __bool__(self):
    return self.item.__bool__()
  def __neg__(self):
    return self*-1
  def astype(self,t):
    self.item=self.item.astype(t)
  def resize(self,*shape):
    return tensor(np.resize(self.item,shape))
def randn(*shape,requires_grad=False):
  return tensor(np.random.randn(*shape),requires_grad=requires_grad)
class no_grad:
  def __init__(self):
    pass
  def __enter__(self):
    global_track=False
  def __exit__(self,*args):
    if args[0] is not None:
      print(args)
      raise
    global_track=True

if __name__ == '__main__':
  a=randn(3,3,requires_grad=True)
  b=randn(3,3,requires_grad=True)
  c=randn(6,6,6)
  print(c)
  print(sum_compress(c,dim=1,new_length=2).shape)
