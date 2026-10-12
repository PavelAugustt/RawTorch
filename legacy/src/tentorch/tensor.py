import numpy as np
from numpy import dtype, float64, broadcast_to

class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        self.data = np.array(data, dtype=float64)
        self.shape = self.data.shape
        self.requires_grad= requires_grad
        self.grad = None
        self._backward = None
        self._prev = []
        self._name = name or f"Tensor_{id(self)}"

    def __repr__(self):
        return f"Tensor({self.data})"
    
    def backward(self):
        """Reverse-mode AD via a topological walk (micrograd-style).

        Build children-before-parents order, then apply local _backward
        closures from the output toward the leaves so every grad is ready
        before parents consume it.
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            for parent in tensor._prev:
                build_topo(parent)
            topo.append(tensor)

        build_topo(self)

        for tensor in reversed(topo):
            if tensor._backward:
                tensor._backward()

    def _visualize_graph(self):
        def recurse(tensor, visited=None, indent=0, prefix=""):
            if visited is None:
                visited = set()

            if id(tensor) in visited:
                print(f"{' ' * (indent * 2)}{prefix}└── {repr(tensor)} (already visited)")
                return
            visited.add(id(tensor))

            connector = f"{prefix}└── " if prefix else ""
            print(f"{' ' * (indent * 2)}{connector}{repr(tensor)}")

            for i, child in enumerate(tensor._prev):
                next_prefix = "│   " if i < len(tensor._prev) - 1 else "    "
                recurse(child, visited=visited, indent=indent + 1, prefix=next_prefix)

        print("\nComputation Graph:")
        recurse(self)
         
    
    def broadcast(self, other):
        if isinstance(other, Tensor):
            if other.requires_grad:
                other_data = other.data
                broadcasted_data = (other_data, self.shape)
                requires_grad = True
            else:
                broadcasted_data = other.data
                requires_grad = self.requires_grad
        else:
            broadcasted_data = other
            requires_grad = self.requires_grad
    
        return Tensor(broadcasted_data, requires_grad=requires_grad)

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = self.broadcast(other) if other.shape != self.shape else other
        else:
            other = Tensor(other)  # Convert to Tensor for compatibility
            
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            self.grad += np.ones_like(self.data) * out.grad
            other.grad += np.ones_like(other.data) * out.grad

        out._backward = _backward
        out._prev = [self, other]

        return out

    def __sub__(self, other):
        # to ensure other.data is a tensor
        if isinstance(other, Tensor):
            other = self.broadcast(other) if other.shape != self.shape else other
        else:
            other = Tensor(other)

        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            self.grad += np.ones_like(self.data) * out.grad
            other.grad += -np.ones_like(other.data) * out.grad

        out._backward = _backward

        out._prev = [self, other]

        return out                
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = self.broadcast(other) if other.shape != self.shape else other
        else:
            other = Tensor(other)

        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)


        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            self.grad += other.data * out.grad # in respect to self
            other.grad += self.data * out.grad # in respect to out

        out._backward = _backward
        out._prev = [self, other]

        return out             
    
    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        if len(self.shape) == 1 and len(other.shape) == 1:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for dot product: {self.shape} and {other.shape}")
            out_shape = ()
        elif len(self.shape) == 1 and len(other.shape) == 2:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for vector-matrix product: {self.shape} and {other.shape}")
            out_shape = (other.shape[1],)
        elif len(self.shape) == 2 and len(other.shape) == 1:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix-vector product: {self.shape} and {other.shape}")
            out_shape = (self.shape[0],)
        else:
            if self.shape[-1] != other.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.shape} and {other.shape}")
            out_shape = self.shape[:-1] + other.shape[1:]

        out = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                if len(self.shape) == 1 and len(other.shape) == 1:
                    self.grad = self.grad + (other.data * out.grad) if self.grad is not None else other.data * out.grad
                else:
                    grad_shape = np.matmul(out.grad.reshape(*out_shape), other.data.T.reshape(other.shape[-1], -1))
                    self.grad = self.grad + grad_shape if self.grad is not None else grad_shape

            if other.requires_grad:
                if len(self.shape) == 1 and len(other.shape) == 1:
                    other.grad = other.grad + (self.data * out.grad) if other.grad is not None else self.data * out.grad
                else:
                    grad_shape = np.matmul(self.data.T.reshape(-1, self.shape[-1]), out.grad.reshape(*out_shape))
                    other.grad = other.grad + grad_shape if other.grad is not None else grad_shape

        out._backward = _backward
        out._prev = [self, other]
        return out

    
    def randn(shape):
        return Tensor(np.random.randn(shape))