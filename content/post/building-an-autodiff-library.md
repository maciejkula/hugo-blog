+++
title = "Building an autodifferentiation library"
author = ["maciej"]
date = 2018-07-18T17:38:00+01:00
lastmod = 2018-07-18T17:38:23+01:00
categories = ["engineering"]
draft = false
weight = 2001
+++

_This blog post originally appeared on [Medium](https://medium.com/@maciejkula/building-an-autodifferentiation-library-9ccf32c7a658)_

Popular general-purpose [auto-differentiation](https://en.wikipedia.org/wiki/Automatic%5Fdifferentiation) frameworks like PyTorch or TensorFlow are very capable, and, for the most part, there is little need for writing something more specialized.

Nevertheless, I have recently started writing my own autodiff package. This blog post describes what I’ve learned along the way. Think of this as a poor-man’s version of a [Julia Evans](https://jvns.ca/) blog post.

Note that there are many blog posts describing the mechanics of autodifferentiation much better than I could, so I skip the explanations here. Additionally, there are several other [interesting](http://colah.github.io/posts/2015-09-NN-Types-FP/) [posts](https://jeremyrsmith.github.io/scala-math-slides/#23) [and](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html) [articles](https://arxiv.org/abs/1710.06892) on building type-safe neural networks constructs, so while my library follows very similar patterns (statically-typed graphs and dependent types), I don’t dwell on the type system angle too much.

Finally, In case you’d like to jump straight to the code, the end result is [here](https://github.com/maciejkula/wyrm), together with an obligatory neural-network based [FizzBuzz solution](https://github.com/maciejkula/fizzbuzz).


## Motivation {#motivation}

There are a couple of reasons why I wanted to have my own autodiff/backprop framework, rather than use PyTorch or TensorFlow.

-   PyTorch and TF are quite slow when fitting models that require little computation per minibatch. In computer vision problems so much computation is done per minibatch that framework overhead is mostly a non-issue. This isn’t true of fitting matrix-factorization-style models, useful in the recommender systems community. Even on a GPU, fitting these models is very slow.
-   I want to be able to use my autodiff library to write and distribute models as Python packages with minimal dependencies. Being able to produce a fairly small and spelf-contained binary is an advantage over the rather heavy TF and PyTorch dependencies.
-   It was a fun learning experience, and allowed me to understand the inner workings of mature neural network libraries in a little bit more detail.

Motivated by the desire for a lightweight solution that works well for recommender (and possibly NLP) models, I wrote down a list of design constraints.

-   I want the framework to naturally support sparse gradients: cases where the vast majority of gradients are zero. This is very common in NLP and recommender models that use large embedding layers. In any given minibatch, only a very small proportion of the embedding layer is used, and the gradients of the remaining entries are zero. Being able to skip the zeros when performing a gradient update is essential in making these models fast.
-   I want the framework to have minimal overhead on top of the actual computation. Since I mainly want to fit small, sparse models, overhead is key. In PyTorch, the run time of such models is dominated by the overhead of looping in Python. To avoid this, my library has to forego Python in its fitting loop, and be written entirely in a compiled language to take advantage of compiler optimizations.
-   The models graphs have to be define-by-run, much like Chainer or PyTorch. The usability and debuggability of this approach is too valuable for me to even contemplate going back to the TensorFlow way of doing things. At the same time, I’m happy for the graph to be static once defined. This helps in keeping the overhead small: I can allocate intermediate computation buffers once and keep re-using them, instead of writing a complex buffer pool system (or, worse yet, repeatedly allocating and freeing memory on every pass).
-   I want performance to scale approximately linearly with the number of available CPU cores. This means parallelizing at the level of the entire graph rather than individual operations. Each computation thread will have its own copy of the graph, but write to shared parameter buffers on update. This is effectively the Hogwild! approach, where multiple threads of computation update shared parameter buffers concurrently, without any locking. This allows near-linear scaling with little degradation in model quality as long as gradients are relatively sparse.

There is also a short list of things I don’t want, or don’t care enough about to add for now:

-   GPU support. I mostly want to fit tiny models (or at least models with lots of parameters but little computation per minibatch).
-   CNNs, or, indeed, tensors with more than two dimensions.

Given the list of requirements (and non-requirements), some design decisions follow naturally.

-   The whole thing is going to be written in a compiled language that is capable of producing native shared objects with no runtime. Models will also be defined in the same language.
-   That language is going to be [Rust](https://www.rust-lang.org/). It’s an amazing language, and a perfect fit for this sort of task. For this reason, a lot of what follows has a Rust flavour. However, the design trade-offs I describe will (I believe) be the same in C++ and other statically typed and AOT compiled programming languages.
-   I’m going to use [reverse-mode autodifferentiation](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation). That way, I can easily backpropagate through arbitrary (static) computation graphs with multiple inputs.

When writing libraries, I often think of the API I want to be able to expose and work back from there. In this case, I want to write something like the following:

```rust
let slope = Parameter::new(1.0);
let intercept = Parameter::new(0.0);
let x = Input::new(3.0);
let y = Input::new(2.0 * 3.0 + 1.0);
let loss = (y — (slope * x + intercept)).square();
loss.backward();
```

and have it just work.

Preliminaries done, we can move on to the fun part: figuring out how to implement the graph.


## Representing the graph {#representing-the-graph}

What sort of data structure do we choose to represent the graph? I looked at two alternatives.

-   Vector-based: all the computation nodes are stored contiguously in a vector, and use indices into that vector to address their parent nodes. For example, when creating an input node, an InputNode object is pushed onto the vector with index 0. If you then square that node, SquareNode is pushed onto the tape with index 1, knowing that its parent is an index 0. During a forward pass, the square node will use that index to get the value of its input.
-   Graph-based. Nodes are placed at arbitrary locations in memory, and use references to their parents to maintain the graph structure. (The vector representation can be seen as a linearization of the graph-based model.)

```nil
     Vector-based                              Graph-based

   +---------------+                       +-----------------+
   |               |                       |                 |
+-->     A * B     <--+                +--->      A * B      <--+
|  |               |  |                |   |                 |  |
|  +---------------+  |                |   +-----------------+  |
|  |               |  |                |                        |
|  |       B       +--+                |                        |
|  |               |                   |                        |
|  +---------------+            +------+---------+    +---------+-------+
|  |               |            |                |    |                 |
+--+       A       |            |       A        |    |        B        |
   |               |            |                |    |                 |
   +---------------+            +----------------+    +-----------------+
```

There are a couple of advantages to the vector-based approach.

-   All the nodes are in the same place. They are stored contiguously in memory, potentially reducing memory locality problems.
-   It’s easy to reason about their ownership. This makes cloning the graph very easy: you just clone the node vector. This is important because I rely on having multiple copies of the graph for my parallelization approach.
-   The nodes are arranged in topological order. We can correctly perform a forward pass with no duplicate work by simply iterating forward along the vector.

But there are also disadvantages.

It’s not clear what sort of object we are storing in the node vector. All of the nodes are different types (of different sizes), and vectors are homogeneously typed. Rust offers two solutions to this problem, but neither is fully satisfactory.

The first is [enums](https://doc.rust-lang.org/book/first-edition/enums.html) (sum types; ADTs; tagged unions). We define a `Node` type to be the union of all possible node types, and store that in the node vector. This way, everything has the same type. We still need to dispatch the node’s methods from the enclosing `Node` type to the contained inner node. This can be done via [pattern matching](https://doc.rust-lang.org/book/first-edition/match.html) (a switch statement on the tags of the union type); with Rust’s support for pattern matching and macros, writing the necessary code is a breeze.

However, this imposes a runtime cost. Every time we use a node, we need to go through the switch statement to resolve the inner type. In principle, optimizing compilers will compile such code to jump tables. In practice, the assembly generated for the dispatch code in my experiments was simply a linear scan over all the possibilities, imposing a dispatch cost that is linear in the number of concrete node types the framework supports. Worse still, the compiler is reluctant to inline both the switch itself and the called functions. The former is bad because it increases branch prediction misses, the latter increases function call overhead. (This problem is exacerbated by the recent branch-prediction attacks: it’s likely that [compiler mitigations](http://archive.is/s831k) will make indirect instructions like these substantially more expensive.)

The final disadvantage of using sum types for the node vector is that it results in a closed system (akin to Scala’s [sealed traits](https://underscore.io/blog/posts/2015/06/02/everything-about-sealed.html)): downstream users of the library cannot add new node types.

The alternative is to use Rust’s runtime polymorphism mechanism, [trait objects](https://doc.rust-lang.org/book/first-edition/trait-objects.html). Trait objects are a way of abstracting over the concrete type of an object: instead of storing structs inline, we hide them behind a pointer to their data and a table of their methods. When calling a method, we jump to the vtable, find the function, and execute it. Using trait objects, we put these fat pointers into the node vector instead of nodes themselves.

This solution, however, introduces exactly the kind of indirection we set out to avoid in the first place. Additionally, it completely defeats the compiler’s efforts at inlinining: the function to be called is not known until runtime.

What about the graph-based design? Here, each node is placed in its own location in memory, and can refer to its ancestors via references. Because each node can be re-used an arbitrary number of times, I use Rust’s equivalent of a `shared_ptr` from C++, [`the Rc<T>`](https://doc.rust-lang.org/std/rc/struct.Rc.html).

One immediate disadvantage of this approach is that it blurs the ownership structure of the graph, making cloning and serialization/deserialization difficult: because nodes can be re-used, naive cloning/deserialization will result in multiple copies of the same nodes being created.

The second disadvantage is the lack of a readily-available topological ordering: both forward and backward passes have to be done recursively, and care has to be taken to avoid re-computing the values of shared subgraphs.

The advantage of using the graph representation is the types of any node’s parents are known at compile time. Every node is (recursively) generic over the types of its parents: adding two InputNodes will produce an `AddNode<InputNode, InputNode>`. Adding that to another input node will produce an `AddNode<AddNode<InputNode, InputNode>, InputNode>` and so on. This gives me static method dispatch and the potential for inlining, in addition to a design that plays much more nicely with the type system.


## Results {#results}

Using some informal benchmarks, the graph-based approach is approximately 30% faster than the vector-based approach. The end result can run a full epoch of a BPR learning-to-rank factorization model on the Movielens 100K dataset ([code](https://github.com/maciejkula/wheedle/blob/master/src/lib.rs#L422%2529)) in under 20 milliseconds on my puny dual-core laptop, and should scale linearly with more cores.

This takes advantage of a number of optimizations in addition to the underlying graph structure.

-   I use Rust’s [SIMD intrinsics](https://rust-lang-nursery.github.io/stdsimd/x86%5F64/stdsimd/) for a number of operations, like vector dot products and scaled addition.
-   For most operations, I assume C-contiguous matrices and iterate directly over the underlying data rather than use `ndarrays` [iterator methods](https://docs.rs/ndarray/0.11.0/ndarray/iter/struct.Iter.html). This turns out to be much faster, presumably because it allows LLVM to autovectorize the loops.
-   It turns out that LLVM is smart enough to autovectorize most numerical loops that don’t involve a reduction step (mostly assignments). Combined with (2), this makes a lot of numerical loops efficient with minimal optimization effort.

There are a number of ways to make the computation faster still.

1.  At the moment, the code doesn’t do any subgraph result caching in the forward pass: if a node is used twice in the forward pass, all of the computations it depends on will be done twice. This can easily be solved via a simple topological sort algorithm, marking the nodes as evaluated once they have evaluated their value. (_Addendum: this turns out to be incredibly important for recurrent neural networks, so is now implemented._)
2.  Similarly, gradients are passed straight to parameter nodes in the backward pass. If a node is used more than once, this means that unnecessary work is done in passing its gradients down one at a time. Accumulating all the gradients and only recursing once will save on that work. (_Addendum: as above._)
3.  There is some unnecessary copying of inputs; making better use of references when possible should yield some small performance gains.


## What’s next {#what-s-next}

I have written (and continue to maintain) a number of open-source Python ML packages. The models are written by hand in Cython, and while they perform well, extending them is tricky. This is due partly to Cython’s limitations, and partly due to the effort required for manual derivation of update rules.

I hope that this library (or some variation thereof) will make that task easier, and allow me to more easily implement complex models and release them as standalone Python packages. I’ll report back on how I fare.


## Addendum {#addendum}

Turns out that the graph representation is a little bit problematic when applied to recurrent neural networks: at every step of the recurrence, the complexity of the resulting types increases, leading to rather baroque types:

```rust
Variable<nodes::LogNode<nodes::SoftmaxNode<nodes::DotNode<layers::recurrent::LSTMCellHidden<layers::recurrent::LSTMCellState<layers::recurrent::LSTMCellSt
ate<layers::recurrent::LSTMCellState<nodes::InputNode, nodes::InputNode, nodes::IndexNode<nodes::ParameterNode>>, layers::recurrent::LSTMCellHidden<nodes::InputNode, nodes::InputNode, nodes::IndexNode<nodes::Par
ameterNode>>, nodes::IndexNode<nodes::ParameterNode>>, layers::recurrent::LSTMCellHidden<layers::recurrent::LSTMCellState<nodes::InputNode, nodes::InputNode, nodes::IndexNode<nodes::ParameterNode>>, layers::recu
rrent::LSTMCellHidden<nodes::InputNode, nodes::InputNode, nodes::IndexNode<nodes::ParameterNode>>, nodes::IndexNode<nodes::ParameterNode>>, nodes::IndexNode<nodes::ParameterNode>>, layers::recurrent::LSTMCellHid
den<layers::recurrent::LSTMCellState<layers::recurrent::LSTMCellState<nodes::InputNode, nodes::InputNode, nodes::IndexNode<nodes::ParameterNode>>, layers::recurrent::LSTMCellHidden<nodes::InputNode, nodes::Input
Node, nodes::IndexNode<nodes::ParameterNode>>, nodes::IndexNode<nodes::ParameterNode>>, layers::recurrent::LSTMCellHidden<layers::recurrent::LSTMCellState<nodes::InputNode, nodes::InputNode, nodes::IndexNode<nod
es::ParameterNode>>, layers::recurrent::LSTMCellHidden<nodes::InputNode, nodes::InputNode, nodes::IndexNode<nodes::ParameterNode>>, nodes::IndexNode<nodes::ParameterNode>>, nodes::IndexNode<nodes::ParameterNode>
>, nodes::IndexNode<nodes::ParameterNode>>, nodes::ParameterNode>>>>
```

Needless to say, after a couple of recurrent steps the compiler gives up. This can be resolved by implementing a fused LSTM cell, rather than assembling it from simpler operations, or opting for selective type erasure via trait objects. So far, I’ve used the second solution: the output values of each LSTM cell have their concrete types erased by boxing them up in a trait object. Still, it illustrates the dangers of relying on complex type system constructs.
