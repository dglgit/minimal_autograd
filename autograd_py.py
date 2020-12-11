from operator import add
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
  ob.grad=ob.grad*(context*(abs((x.item<0).astype(int)-1)))
def relu_back(ob,context):
  ob.grad=ob.grad*(context.item<0).astype(int)
def lrelu_back(ob,context):
  inv=abs((context.item<0).astype(int)-1)*c
  negative=(context.item<0).astype(int)
  ob.grad=ob.grad*(inv+negative)
def mm(a,b,do_grad=False):
  result=tensor([[sum(a[i]*b[:,j]) for j in range(b.shape[1])] for i in range(a.shape[0])])
  if do_grad:
    print('y')
    result.parents=(a,b)
    a.grads.insert(0,(matmul_back,b))
    b.grads.insert(0,(rmatmul_back,a))
  return result
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
matmul_grads=[matmul_back, rmatmul_back]
div_grads=[div_back,matmul_grads]
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

def create_container(mutable=True,*data):
  return list(data) if mutable else placeholder_list()
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
    for i in range(len(grads)):
      others[i].grads.append(grads[i])
  else:
    parents=()
  return tensor(result,parents=parents,requires_grad=len(parents)>0)
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
    self.grads=create_container(requires_grad)
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
    if type(other) in native_python_types:
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
      self.grad=0
      self.grad+=incoming_grad
      for f,i in self.grads:
        f(self,i)
    elif len(self.grads)>0:
      self.grad=self.grads[0][1]
    for parent in self.parents:
      parent.backward2(self.grad)
    if self.grad is not None:
      self.gradv+=self.grad
  def t(self):
    if self.requires_grad:
      self.grads.insert(0,(transpose_back,1))
    return tensor(self.item.T)
  def flatten(self):
    return [[j for j in i] for i in self.item]
  def __div__(self, other):
    result=self.item/other
    self.grads.insert(0,(mul_back,1/other))
    return tensor(result,parents=(self,other))
  def __sub__(self,other):
    if self.requires_grad:
      result=tensor(self.item-other,parents=(self,other))
    else:
      result=tensor(self.item-other)
    return result
  def __rsub__(self,other):
    result=tensor(other-self.item,parents=(self,other))
    return result
  def __radd__(self,other):
    result=tensor(other+self.item)
    return result
  def __matmul__(self,other):
    result=mm(self,other)
    print('yeet')
    if self.requires_grad:
      self.grads.insert(0,(matmul_back, other))
    if other.requires_grad:
      other.grads.insert(0,(rmatmul_back, self))
      #self.tensors.insert(0,other)
    return tensor(result,parents=(self,other))
  def __rmatmul__(self,other):
    print('rmatmul')
    result=tensor(mm(other,self,self.requires_grad))
    if self.requires_grad:
      self.grads.insert((rmatmul_back,other))
    return result
  def __pow__(self, other):
    result=self.item**other
    self.grads.insert(0,(pow_back,other))
    return tensor(result,parents=(self,other))
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
  def reshape(self):
    self.grads.insert(0,(reshape_back,self.item.shape))
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
def randn(*shape,requires_grad=False):
  return tensor(np.random.randn(*shape),requires_grad=requires_grad)
class no_grad:
  def __init__(self):
    pass
  def __enter__(self):
    global_track=False
  def __exit__(self,*args):
    global_track=True
a=randn(2,3,requires_grad=True)
b=randn(2,3,requires_grad=True)
dud=placeholder_list([2,3,4])
print(list(dud),type(list(dud)))
print('a')
print(a)
print('b')
print(b)
c=a*b
print('c')
print(c)
print(b.grads)
print(type(b[0][0]))