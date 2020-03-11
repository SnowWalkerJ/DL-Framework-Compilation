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

### Direct Construction (Tensorflow)

This is like our Numflow. It provides basic Expression classes and user 

### Trace (Pytorch)

### Script (Pytorch)

