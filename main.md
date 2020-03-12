# Compilation Principal of Pytorch and TensorFlow

## Introduction

(... refer to pre.pdf)

## Organization

### Modules

As memtioned before, neural networks consists of smaller neural networks aka layers. Both Pytorch and Tensorflow manage these layers with "modules". Each module represents a layer, and manages its own parameters and child modules. It defines how data flows between child modules and is transformed to outputs.

### Code

```python
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
```

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(32, 1),
)
```

## Why need compilation

Both Pytorch and tensorflow can run directly in Python without compilation. So why is compilation so important?

1. **Performance**. Although the bottleneck of deep learning is mostly with GPU and the performance of particular language
is not that important, you still probably want to save time especially in production environment.
2. **Language Irrelevance**. Training in a model in Python is common and efficient. But deploying the model may involing other
languages. It's convenient to translate the model into a language-irrelevant form that can be executed by other languages.
3. **Optimization**. There are plenty of potential operations run by compilers to optimize the computation graph. For example, transforming `f(x) + f(x)` into `f(x) * 2` saves a lot of calculation when `f()` is computationally intensive. There is also a technique called "fushion" on GPU that combines neighbor operations into groups instead of running them one-by-one, that promotes efficiency.

## Problem Feature

### Pureness

The target function is a **pure** function. It doesn’t rely on anything except the input data. Nor does it affect anything except the output.

All variables and parameters are constant regarding the graph.

Look at the following code:

```python
a = f(x)
...  # lines of other operations
b = f(x)
```

If `f()` is pure, we know `a == b` as long as x is not changed. Optimization can be done so that the value of `b` can be
set to `a` instead of evaluating `f(x)` again.

However, if `f()` is not pure, `b` can be different from `a` even when `x` is unchanged. And we always have to recalculate `f(x)`.

### Conditioning Is Rare

In the world of deep learning, almost all operations are data-independent. No matter what data you feed, they always do
the same thing. It is rare you need to write code like "if ..." or "for ...". It makes sure the syntax tree is simple
and nothing rely on run-time information.

For example:

```python
def f_dynamic(x):
    while x > 1:
        x = g(x)
    return x
```

```python
def f_static(x):
    x = g(x)
    x = g(x)
    x = g(x)
    return x
```

`f_dynamic` is hard to optimize because at compile-time you never know how many times does it call `g()` exactly. Thus you can't trace the computation path of x. On the other hande, `f_static` is quite straight-forward.

## Solutions

### Trace

As mentioned before, deep learning operations are usually data-independent. Pytorch takes this advantage and develops a method called "trace". The basic idea is to get a sample data, put it through the computation graph, and trace what operations it's been through.

Pros:

- This is easy to accomplish
- Generated expression tree is simple to execute
- Functions are inlined automatically
- Unused operations are dropped automatically

Cons:

- Unable to express data-dependent operations

#### Example

We have a function `f`:

```python
def f(x):
    y = (x[0] * theta[0]) + (x[1] * theta[1])
    return exp(y)
```

Then we construct a sample data:

```python
x = Tensor([0.1, 0.1])
```

The `Tensor` class is written in a way that all its operators are overriden so that when you call `x[0] * theta[0]` it generates something like: `Mul(Index(x, 0), Index(theta, 0))`.

Then `f(x)` is compiled into something like:

```python
Exp(Add(
    Mul(Index(x, 0), Index(theta, 0)),
    Mul(Index(x, 1), Index(theta, 1)),
))
```

However, if we have a more complex function:

```python
def g(x):
    if x[0] > 0:
        return x / 2
    else:
        return x + 1
```

Then the behavior of the compiled graph is undefined. It depends on the sample data you put into the function.

If you feed `x = [0.1, 0.1]`, then it goes into the first branch and returns `Div(x, 2)`.
If you feed `x = [-0.1, 0.1]`, then it goes into the second branch and returns `Add(x, 1)`.

### Script

Too solve the data-dependency problem, Pytorch comes with another compilation solution called "script". It provides
a new language called `TorchScript`, which is a subset of Python language.

The "script" method first translates your
python function into a TorchScript "abstract syntax tree"(AST). The variables are static-typed. Their types are decided
according to initial value, type hint or comment. An AST is a tree of blocks, statments and expressions as nodes.

Then each node in the tree is replaced (re-interpreted) to a TorchScript intermediate representation(IR).

An IR is a representation of code between high-level programming language and low-level assembly. It's high-level enough not to care about the details of target computer, while low-level enough that it inherits no character of the original language. It's commonly used in modern compilation toolchains like llvm.

![IR struct](./images/ir-struct.png)

At this stage, the computation graph is constructed. Pytorch will run some optimizations to the graph, including "UnrollLoops", "EliminateDeadCode", "EliminateCommonSubexpression", "FuseGraph".

Finally a graph executer translates the IR into more low-level instructions, which can be executed by a pytorch virtual machine.

#### Example

TorchScript code:

```python
def func(x):
    if x[0] > 0:
        return x * 2
    else:
        return x + 1
```

Python AST:

```python
FunctionDef(name="func", params="x", body=[
    IfBlock(test=GT(GetItem(Name("x"), Number(0)), Number(0)), body=[
        Return(Mul(Name("x"), Number(2)))
    ], else_=[
        Return(Add(Name("x"), Number(1)))
    ])
])
```

TorchScript IR:

```
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=0]()
  %8 : int = prim::Constant[value=2]()
  %11 : int = prim::Constant[value=1]()
  %4 : Tensor = aten::select(%x.1, %2, %2)
  %5 : Tensor = aten::gt(%4, %2)
  %6 : bool = aten::Bool(%5)
  %22 : Tensor = prim::If(%6)
   block0():
      %9 : Tensor = aten::mul(%x.1, %8)
    block1():
      %13 : Tensor = aten::add(%x.1, %11, %11)
  return (%22)
```

Instructions:

```asm
MOV x, y
ADD x, 1
PUSH x
TEST x, x
JGT ...
ADD x, 1
JMP ...
DIV x, 2
RET x
```
