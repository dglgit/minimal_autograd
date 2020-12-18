from autograd_py import *

class Base:
    def __init__(self,*args):
        self.parameters=[i for i in self.__dict__.values() if hasattr(i,'requires_grad')]
        self.grads=[]
    def zero(self):
        for param in self.parameters:
            param.zero()
    def log_op(self,fn,args):
        #self.grads.insert(0,(args,op_grads[fn]))
        return fn(args)
    
            
class Linear(Base):
    def __init__(self,in_shape,out,bias=True):
        self.w=randn(1,in_shape,out,requires_grad=True)
        if bias:
            self.b=randn(1,1,out,requires_grad=True)
        else:
            self.b=np.zeros((1,1,out))
        self.in_shape=in_shape
        self.out=out
        super().__init__()
    def __call__(self,x):
        return mm(x,self.w)+self.b

class static(Base):
    def __init__(self,*modules):
        self.train=True
        self.modules=modules
        super().__init__()
    def train(self):
        self.train=True
    def eval(self):
        self.train=False
    def __call__(self,x):
        for i in self.modules:
            x=i(x)
class make_constructor:
    def __init__(self,func,*args):
        self.func=func
        self.args=args
    def __call__(self,*x):
        return self.func(*x,*self.args)
        
class sgd:
    def __init__(self,params,lr):
        self.params=params
        self.lr=lr
    def step(self,zero=True):
        if zero:
            for p in self.params:
                p.item-=p.gradv*self.lr
                p.zero()
        else:
            for p in self.params:
                p.item-=p.gradv*self.lr