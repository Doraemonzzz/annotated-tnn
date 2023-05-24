<center><h1> The Annotated Tnn </h1></center>


<center>
<p><a href="https://openreview.net/pdf?id=IxmWsm4xrua">Toeplitz Neural Network for Sequence Modeling</a></p>
</center>
<img src="../images/network.png" width="100%"/>

*博客由[Doreamonzzz](https://github.com/Doraemonzzz)撰写。*



更新日志：
- 20230313，开始撰写博客；
- 20230320，完成动机以及各个部件的实现部分；
- 20230524，完成校阅以及引用；

Toeplitz Neural Network(TNN)是一种全新的网络结构，以一种完全不同的方式进行序列建模，在单向/双向语言模型，图像分类任务上和Transformer性能相近，并且在长序列建模[LRA](https://arxiv.org/abs/2011.04006)任务上取得和[S4](https://arxiv.org/abs/2111.00396)相当的性能。这篇博客的主要目的就是以[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)和[The Annotated S4](https://srush.github.io/annotated-s4/)风格介绍TNN，在阅读完这篇博客后，您将得到如下收获：
1. 了解TNN的动机和设计理念；
2. 掌握TNN各个部件的实现；
<!-- 3. 学习如何将TNN应用在$n$维序列建模任务；
1. 了解TNN的优缺点；
2. 了解TNN和S4, RWKV等方法的联系； -->

总而言之，在阅读完本博客之后，您将成为TNN的专家，并且可以将TNN应用到您的项目中，让我们开始吧。

<!-- # 目录

- [目录](#目录)
  - [预备知识](#预备知识)
    - [Token mixing and channel mixing](#token-mixing-and-channel-mixing)
    - [相对位置编码](#相对位置编码)
  - [TNN的动机](#tnn的动机)
  - [TNN的实现](#tnn的实现)
    - [准备工作](#准备工作)
    - [Tno的实现](#tno的实现)
      - [Naive实现](#naive实现)
      - [Matrix production实现](#matrix-production实现)
      - [FFT实现](#fft实现)
      - [Circulant matrix](#circulant-matrix)
        - [定义](#定义)
        - [快速矩阵乘法](#快速矩阵乘法)
        - [实现](#实现)
        - [小结](#小结)
    - [Toeplitz matrix](#toeplitz-matrix)
      - [定义](#定义-1)
        - [快速矩阵乘法](#快速矩阵乘法-1)
        - [实现](#实现-1)
      - [验证实现](#验证实现)
      - [补充](#补充)
      - [小结](#小结-1)
    - [Rpe的实现](#rpe的实现)
      - [Naive实现](#naive实现-1)
      - [Relative Position Encoder](#relative-position-encoder)
      - [实现Relative Position Encoder](#实现relative-position-encoder)
      - [将Tno和Rpe合并](#将tno和rpe合并)
    - [Tnn layer的实现](#tnn-layer的实现)
      - [GLU](#glu)
      - [GTU](#gtu)
      - [TnnLayer](#tnnlayer)
      - [小结](#小结-2)
  - [全文总结](#全文总结) -->

## 预备知识

### Token mixing and channel mixing
让我们首先从Transformer开始。Transformer作为一个网络结构已经席卷了各个领域，其核心部分主要可以由如下两个计算公式描述：
$$
\begin{aligned}
\mathbf X_1 &=\mathrm{Norm}(\mathbf X + \mathrm{MHA}(\mathbf X)),\\
\mathbf O &= \mathrm{Norm}(\mathbf X_1 + \mathrm{FFN}(\mathbf X_1)).
\end{aligned}
$$
其中$\mathbf X \in \mathbb R^{n\times d}$是输入（也可以称为token matrix，其中矩阵的每一行为一个token的向量表示），$n$是序列长度，$d$是特征维度。

既然现在有两个主要模块——$\mathrm {MHA}$和$\mathrm {FFN}$，那么他们的作用是否有所不同呢？在[Metaformer](https://arxiv.org/abs/2111.11418)一文中，研究者指出，$\mathrm {MHA}$的主要作用是Token mixing，而$\mathrm {FFN}$的主要作用是Channel mixing。

这是什么意思呢？我们可以从矩阵乘法的角度清晰的理解这点：给定输入（token matrix）$\mathbf X \in \mathbb R^{n\times d}$，考虑矩阵乘法$\mathbf A \mathbf X$和$\mathbf X \mathbf B$，那么：
- $\mathbf A \mathbf X$表示矩阵$\mathbf X$行的线性组合，而每一行表示一个token，即token的线性组合，所以称为token mixing；
- $\mathbf X  \mathbf B$表示矩阵$\mathbf X$列的线性组合，而每一列表示一个channel，即channel的线性组合，所以称为channel mixing；

在Transformer中，矩阵$\mathbf A$即为$\mathrm{Softmax}(\mathbf Q \mathbf K^{\top} /\sqrt{d})$，矩阵$\mathbf B$即为$\mathrm {FFN}$中的全连接层。

大多数对Transformer的改进都是集中在token mixing:$\mathbf A \mathbf X$的计算上，以各种各样的方式降低其运算复杂度，TNN也是使用了类似的思路，最核心的一点就是利用了相对位置编码，或者说，Toeplitz矩阵。

### 相对位置编码

位置编码是Transformer中的重要组成部分，一开始广为使用的是[绝对位置编码(APE)](https://arxiv.org/abs/1706.03762)，这种编码的方式可以用如下计算方式概括：
$$
\mathbf x_i =\mathbf w_i + \mathbf p_i.
$$
其中$\mathbf w_i$表示第$i$个词的word embedding，$\mathbf p_i$表示第$i$个位置的position embedding。

后来，有研究人员发现，在序列建模中，词的相对位置信息，可能比词的绝位置信息更加重要。
> 例如"我年纪比你大"的语意和"我年纪比你大"完全不同，但是这两句话只是交换了"你"和"我"的位置。
>

于是研究人员开始将相对位置编码引入，相对位置编码的使用和绝对位置编码有所不同，其作用在Attention计算的位置：
$$
\mathbf s_{ij} = \mathbf q_i^{\top} \mathbf k_j/\sqrt{d} + t_{i-j}.
$$
如果写成矩阵的形式则更加直观：
$$
\begin{aligned}
\mathbf S & = \mathbf Q \mathbf K^{\top} / \sqrt {d} + \mathbf T,\\
\mathbf T & =\left[\begin{matrix}
t_0 & t_{-1} & \cdots  & t_{-n+1} \\
t_1 & t_0  &  &  \vdots \\
\vdots &   &   t_0 & t_{-1} \\
t_{n-1} & \ldots  & t_1 & t_0
\end{matrix}\right] \in \mathbb R^{n\times n}.
\end{aligned}
$$

这里，矩阵$\mathbf T$有一个数学名称——[Toeplitz矩阵](https://en.wikipedia.org/wiki/Toeplitz_matrix)，不难看出该矩阵有$2n-1$个独立元素。

## TNN的动机

有了之前的准备工作，可以引入我们工作的两个动机：
1. 既然相对位置信息如此重要，那么有没有可能只依赖于相对位置信息（Toeplitz matrix）进行token mixing呢？
   1. 直观上来说，就是将Attention Matrix替换为Toeplitz matrix。
2. 假设(1)成立，那么我们需要进行的主要操作是$\mathbf T \mathbf X$，既然矩阵$\mathbf T$是一个特殊结构的矩阵，那么有没有可能加速运算呢？

我们对两个问题都进行了肯定的答复：
1. 完全可以只依赖于相对位置信息进行token mixing；
2. 由于矩阵的特殊性，可以将运算复杂度由$O(n^2 d)$降低为$O(nd\log n)$；

可以看到，我们的动机极其简单和优雅，最核心的思路就是将$\mathrm{Softmax}(\mathbf Q \mathbf K^{\top} / \sqrt {d})$替换为$\mathbf T$，但是，这种简单的替换就可以拥有比各种花哨更改更好的性能，这就更加验证了相对位置信息在序列建模中的重要性。


<!-- 因此，在后续的讨论快速矩阵乘法时，我们指的是及$\mathbf T\mathbf x$，其中$\mathbf T\in \mathbb R^{n\times n}, \mathbf x \in \mathbb R^{n\times 1}$。 -->

## TNN的实现


### 准备工作

接下来的问题就是如何实现TNN，在此之前，我们对之前的公式做一定的调整。

在之前的讨论中，我们提到了$\mathbf T \mathbf X$可以高效实现，其中$\mathbf T\in \mathbb R^{n\times n}, \mathbf X \in \mathbb R^{n\times d}$，这种情况相当于每个channel共享同一个Toeplitz matrix，但是注意到我们可以让不同的channel使用不同的Toeplitz matrix，我们经验上发现，这样一定程度上可以增大模型的表达性，所以在TNN中，**每个channel**使用了不同的Toeplitz matrix。注意到形状为$n\times n$的Toeplitz matrix实际上只有$2n-1$个独立元素，为了方便后续讨论，我们定义如下映射：$f: \mathbb R^{(2n-1)\times 1} \to \mathbb R^{n\times n}$：
$$
f(\mathbf t)=f(t_{-n+1},\ldots, t_{n-1}) =\left[\begin{matrix}
t_0 & t_{-1} & \cdots  & t_{-n+1} \\
t_1 & t_0  &  &  \vdots \\
\vdots &   &   t_0 & t_{-1} \\
t_{n-1} & \ldots  & t_1 & t_0
\end{matrix}\right] \in \mathbb R^{n\times n}.
$$
该映射的作用是将维度为$(2n-1)\times 1$的向量填充为$n\times n$的Toeplitz matrix。

结合之前的记号，我们定义为Tno算子(Toeplitz neural operator)为：
$$
\mathrm{Tno}: \mathbb R^{(2n-1)\times d}\times \mathbb R^{n\times d} \to \mathbb R^{n\times d},\\
\mathbf O= \mathrm{Tno}(\mathbf T, \mathbf X), \\
\mathbf O[:, i]= f(\mathbf T[:, i]) \mathbf X[:, i].
$$

备注：这里的记号$\mathbf T\in \mathbb R^{(2n-1)\times d}$和一开始含义有所不同，注意不要搞混。

在开始正式的实现之前，我们先引入一些必要的依赖库以及一些辅助函数：

<!-- ，写成计算公式即为：
$$
\mathbf O[:, i]= \mathbf T[:, i] \mathbf X[:, i], \\
\mathbf O[:, i]\in \mathbb R^{n\times 1}, \mathbf T[:, i]\in \mathbb R^{n\times n},  \mathbf X[:, i]\in \mathbb R^{n\times 1}.
$$ -->

<!-- 备注：这里记号有点太严谨，$\mathbf T[:, i]$的形状应该是$(2n-1) \times 1$，上述符号是指 -->


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def get_activation_fn(activation):
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            return 1 + F.elu(x)
        return f
    elif activation == "2+elu":
            def f(x):
                return 2 + F.elu(x)
            return f
    elif activation == "silu":
        return F.silu
    else:
        return lambda x: x
```

### Tno的实现

#### Naive实现

最朴素的实现自然是利用定义进行实现，例如如下代码中，我们使用4重循环，外面两重循环遍历batch, channel维度，第三重循环遍历输出位置，最后一重循环遍历求和项，注意到我们的$\mathbf T[:, i]$输入形式为$t_{-n+1}, ... , t_{-1}, t_0, t_1, ... , t_{n - 1}$，第三重循环遍历到$i$时，涉及的$t$为$t_{i}, t_{i-1},\ldots, t_{i-n+1}$，而$n - 1 + i$是$t_{i}$在$\mathbf T[:, i]$的实际索引：


```python
def tno_naive(x, t):
    # x: (b, n, d)
    # t: (2n - 1, d), t_(-(n - 1)), ... , t_(-1), t_0, t_1, ... , t_(n - 1) 
    b, n, d = x.shape
    o = torch.zeros_like(x).to(x)
    for b_ in range(b):
        for d_ in range(d):
            for i in range(n):
                for j in range(n):
                    o[b_][i][d_] += t[n - 1 + i - j][d_] * x[b_][j][d_]

    return o
```

这种实现显然太低效，但是至少我们有了一个正确的版本，这对我们后续改进算法也是有帮助的，不难看出这样计算的时间复杂度为$O(n^2d)$，空间复杂度为$O(nd)$（忽略batch维度）。                             

#### Matrix production实现
第二种实现是并行版本，其思路就是先构造Toeplitz matrix，然后利用矩阵乘法进行计算。最主要的部分是将映射$f$实现出来，代码基于[此处](https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch)，主要思路是先将输入改写为$t_0, t_{-1}, ... , t_{1-n}, t_{n - 1}, ... , t_1$，然后构造index $0, 1, \ldots,n -1, -(n - 1), ..., -1$，将输入映射到Toeplitz matrix，最后得到Toeplitz matrix进行矩阵乘法：


```python
def tno_matrix(x, t):
    # x: (b, n, d)
    # t: (2n - 1, d), t_(-(n - 1)), ... , t_(-1), t_0, t_1, ... , t_(n - 1) 
    n = x.shape[1]
    t = t.unsqueeze(0)
    # c: t_0, t_1, ... , t_(n - 1)
    c = t[:, n - 1:]
    # r: t_0, t_(-1), ... , t_(-(n - 1))
    r = t[:, :n].flip(1)
    # vals: [t_0, t_(-1), ... , t_(-(n - 1)), t_(n - 1), ... , t_1]
    vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
    i, j = torch.ones(n, n).nonzero().T
    t_matrix = vals[:, j - i].reshape(n, n, -1)
    o = torch.einsum("n m d, b m d -> b n d", t_matrix, x)

    return o
```

这种实现的好处是可以利用矩阵乘法，尽管复杂度依然为$O(n^2d)$，但实际效率会快很多；但是由于要构造Toeplitz matrix，所以空间复杂度为$O(n^2d)$，并且这部分还是一个很大的IO开销，所以实际中的速度并不会很快。

#### FFT实现
有了之前的铺垫，可以看出前两种方法无论是时间复杂度和空间复杂度相比Attention并没有什么优势，那么有没有办法解决这点呢？回答是肯定的，这就需要[FFT]()这把利刃。后续的讨论涉及到一些数学知识，这里先高度概括一下思路：
1. 给出Circulant matrix的快速矩阵乘法算法；
2. 建立Toeplitz marix和Circulant matrix的关系；

#### Circulant matrix

##### 定义
矩阵$\mathbf C\in \mathbb R^{n\times n}$是一个[Circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix)当且仅当$\mathbf C_{ij}= c_{(i-j + n )\bmod n}$ ,即：
$$
\mathbf C=\left[\begin{matrix}
c_0 & c_{n-1} &c_{n-2} & \cdots & \cdots & c_{1} \\
c_1 & c_0 & c_{n-1} & \ddots & & \vdots \\
c_2 & c_1 & \ddots & \ddots & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & c_{n-1} & c_{n-2} \\
\vdots & & \ddots & c_1 & c_0 & c_{n-1} \\
c_{n-1} & \ldots & \ldots & c_2 & c_1 & c_0
\end{matrix}\right] \in \mathbb R^{n\times n}.
$$
关于Circulant matrix，有如下重要性质：

Circulant matrix $\mathbf C\in \mathbb R^{n\times n}$正交相似于对角阵$\mathbf \Lambda$，特别地，相似矩阵$\mathbf F$是$n\times n$ DFT矩阵:
$$
\mathbf C = \mathbf F^{\top} \Lambda \mathbf F, \\
\Lambda = \mathrm{diag}\{\mathbf F[c_0,c_1,\ldots, c_{n-1}]^\top\} \in \mathbb R^{n\times n}, 
{\mathbf F}_{st}= \exp\left(\frac{2\pi st i}{n}\right),i^2=-1.
$$
证明可以参考[这里](https://ee.stanford.edu/~gray/toeplitz.pdf)。

##### 快速矩阵乘法

现在考虑matrix-vector production操作$\mathbf M \mathbf x, \mathbf M\in \mathbb R^{n\times n}, \mathbf x\in \mathbb R^{n\times 1}$，那么：

- 如果$\mathbf M$为一般的矩阵，那么该计算的时间复杂度为$O(n^2)$;
- 如果$\mathbf M$为DFT矩阵，那么该计算的时间复杂度为$O(n \log n)$;


基于上述事实，考虑$\mathbf M=\mathbf C$为Circulant matrix的情形，那么：
$$
\mathbf C \mathbf x = \mathbf F^{\top} \Lambda \mathbf F \mathbf x.
$$
该计算可以分解为几个步骤：

- $\mathbf x_{\mathrm{fft}}=\mathbf{Fx}$；
- $\mathbf c_{\mathrm{fft}}=\mathbf F[c_0,c_1,\ldots, c_{n-1}]^\top$；
- $\mathbf o_{\mathrm{fft}}=\mathbf x_{\mathrm{fft}}\odot \mathbf c_{\mathrm{fft}}$；
- $\mathbf o= \mathbf F^{\top} \mathbf o_{\mathrm{fft}}$；

其中$\odot$表示element-wise production，可以看出，算法的总时间复杂度为$O(n\log n)$，空间复杂度为$O(n)$，所以Circulant matrix对应的矩阵乘法是高效的。

##### 实现
有了之前的说明，不难利用`fft`实现上述计算：


```python
def circulant_fft(x, c):
    # x: (b, n, d)
    # c: (n, d), c_0, c_1, ... , c_(n - 1) 
    n = x.shape[1]
    c = c.unsqueeze(0)
    x_fft = torch.fft.rfft(x, n, dim=-2)
    c_fft = torch.fft.rfft(c, n, dim=-2)
    o_fft = x_fft * c_fft
    o = torch.fft.irfft(o_fft, n, dim=-2)

    return o
```

##### 小结

现在我们已经有了一个关于Circulant matrix的高效矩阵乘法，那么下一个问题就是建立Toeplitz matrix和Circulant matrix的关系。

### Toeplitz matrix
#### 定义
矩阵$\mathbf T\in \mathbb R^{n\times n}$是一个Toeplitz matrix当且仅当$\mathbf T_{ij}= t_{i-j}$，即
$$
\mathbf T=\left[\begin{matrix}
t_0 & t_{-1} &t_{-2} & \cdots & \cdots & t_{-n+1} \\
t_1 & t_0 & t_{-1} & \ddots & & \vdots \\
t_2 & t_1 & \ddots & \ddots & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & t_{-1} & t_{n-2} \\
\vdots & & \ddots & t_1 & t_0 & t_{-1} \\
t_{n-1} & \ldots & \ldots & t_2 & t_1 & t_0
\end{matrix}\right] \in \mathbb R^{n\times n}.
$$
从形式上来看，Toeplitz matrix和Circulant matrix非常像，唯一的区别在于前者的独立元素数量为$2n-1$，后者的独立元素数量为$n$，那么一个简单的思路就是将Toeplitz matrix嵌入到一个阶数大于等于$2n-1$矩阵中，而这个矩阵本生是一个Circulant matrix，下面来看下这是如何具体操作的。

可以将Toeplitz matrix $\mathbf T\in \mathbb R^{n\times }$嵌入到Circulant matrix $\mathbf C \in \mathbb R^{2n\times 2n}$中:
$$
c_{k} =\begin{cases}
t_k , 0 \le k \le n - 1\\
t_0 , k=n\\
t_{k -2n},   n+1\le k \le 2n-1
\end{cases} ,
$$
即，
$$
\mathbf C=\left[\begin{array}{ccccc|ccccc}
t_0 & t_{-1} & \ldots & \ldots & t_{-n+1} & t_0 & t_{n-1} & \ldots & t_2 & t_1 \\
t_1 & t_0 & \ddots & & \vdots & t_{-n+1} & \ddots & \ddots & & t_2 \\
t_2 & \ddots & \ddots & \ddots & \vdots & \vdots & \ddots & & \ddots & \vdots \\
\vdots & & \ddots & t_0 & t_{-1} & t_{-2} & & \ddots & \ddots & t_{n-1} \\
t_{n-1} & \ldots & \ldots & t_1 & t_0 & t_{-1} & t_{-2} & \ldots & t_{-n+1} & t_0 \\
\hline t_0 & t_{n-1} & \ldots & \ldots & t_1 & t_0 & t_{-1} & \ldots & \ldots & t_{-n+1} \\
t_{-n+1} & \ddots & \ddots & & t_2 & t_1 & t_0 & \ddots & & \vdots \\
\vdots & \ddots & & \ddots & \vdots & t_2 & \ddots & \ddots & \ddots & \vdots \\
t_{-2} & & \ddots & \ddots & t_{n-1} & \vdots & & \ddots & t_0 & t_{-1} \\
t_{-1} & t_{-2} & \ldots & \ldots & t_0 & t_{n-1} & \ldots & \ldots & t_1 & t_0
\end{array}\right] \in \mathbb R^{2n\times 2n}.
$$
使用分块矩阵的符号，我们可以定义：
$$
\begin{gathered}
 \mathbf C = \left[\begin{matrix}
\mathbf C_1 & \mathbf C_2\\
\mathbf C_3 & \mathbf C_4\\
\end{matrix}\right] \in \mathbb R^{2n\times 2n},\mathbf C_s \in \mathbb R^{n \times n}, s=1,2,3,4,
\mathbf C_1 = \mathbf T 
\end{gathered}.
$$
有了上述准备工作，可以得到Toeplitz matrix-vector production的快速算法。

##### 快速矩阵乘法

对于向量$\mathbf x\in \mathbb R^{n}$, 定义:
$$
\mathbf x_1 = \left[\begin{matrix}
\mathbf x\\
\mathbf 0_n
\end{matrix}\right] \in \mathbb R^{2n},
$$
所以，
$$
\mathbf C \mathbf x_1 =\left[\begin{matrix}
\mathbf C_1 & \mathbf C_2\\
\mathbf C_3 & \mathbf C_4\\
\end{matrix}\right]\left[\begin{matrix}
\mathbf x\\
\mathbf 0_n
\end{matrix}\right]=\left[\begin{matrix}
\mathbf C_1 \mathbf x\\
\mathbf C_3 \mathbf x
\end{matrix}\right]=\left[\begin{matrix}
\mathbf T \mathbf x\\
\mathbf C_3 \mathbf x
\end{matrix}\right] \in \mathbb R^{2n},
$$
因此:
$$
\left[\begin{matrix}
{\mathbf I}_n &
{\mathbf 0}_{n\times n}
\end{matrix}\right]\mathbf C \mathbf x_1  =
\left[\begin{matrix}
\mathbf I_n &
\mathbf 0_{n\times n}
\end{matrix}\right]\left[\begin{array}{c}
\mathbf T \mathbf x\\
\mathbf C_3 \mathbf x
\end{array}\right]=\mathbf T \mathbf x.
$$
关于时间复杂度，注意到我们是将$n\times n$的Toeplitz matrix嵌入到一个$2n\times 2n$的Circulant matrix中，所以时间复杂度仍然为$O(n\log n)$。

##### 实现
和Circulant matrix的情形类似，可以利用`fft`实现上述计算：


```python
def tno_fft(x, t):
    # x: (b, n, d)
    # t: (2 * n, d), t0, t1, ..., t(n-1), t0, t_(-(n-1)), ... , t_(-1)
    n = x.shape[1]
    t = t.unsqueeze(0)
    x_fft = torch.fft.rfft(x, 2 * n, dim=-2)
    t_fft = torch.fft.rfft(t, 2 * n, dim=-2)
    o_fft = x_fft * t_fft
    o = torch.fft.irfft(o_fft, 2 * n, dim=-2)[:, :n]

    return o
```

#### 验证实现
在之前的讨论中，我们给出了Tno的三种实现方式，在本节中，我们将验证这些实现的正确性。


```python
b = 2
n = 16
d = 128

t_zero = torch.randn(1, d)
# t1, ..., t(n-1)
t_pos = torch.randn(n - 1, d)
# t-(n-1), ... , t-1
t_neg = torch.randn(n - 1, d)
t1 = torch.cat([t_neg, t_zero, t_pos], dim=0).cuda()
t2 = torch.cat([t_zero, t_pos, t_zero, t_neg], dim=0).cuda()
x = torch.randn(b, n, d).cuda()

o1 = tno_naive(x, t1)
o2 = tno_matrix(x, t1)
o3 = tno_fft(x, t2)

print(f"The output error between tno_naive and tno_matrix is {torch.norm(o1 - o2)}")
print(f"The output error between tno_naive and tno_matrix is {torch.norm(o1 - o3)}")
```

    The output error between tno_naive and tno_matrix is 2.414959999441635e-05
    The output error between tno_naive and tno_matrix is 5.38119456905406e-05


#### 补充
现在我们已经完成了大部分内容，这里最后补充如何将Tno适配到Autoregressive Language Model(causal)的情形。和Attention类似，只要保证Toeplitz matrix的上三角部分为$0$即可，即：
$$
\mathbf T=\left[\begin{matrix}
t_0 & 0 & 0 & \cdots & \cdots & 0 \\
t_1 & t_0 & 0 & \ddots & & \vdots \\
t_2 & t_1 & \ddots & \ddots & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & 0 & 0 \\
\vdots & & \ddots & t_1 & t_0 &0 \\
t_{n-1} & \ldots & \ldots & t_2 & t_1 & t_0
\end{matrix}\right] \in \mathbb R^{n\times n}.
$$
在实现时，注意到`fft`是zero padding，所以只需要将输入：
```python
t2 = torch.cat([t_zero, t_pos, t_zero, t_neg], dim=0).cuda()
```
修改为下式即可：
```python
t2 = torch.cat([t_zero, t_pos, t_zero], dim=0).cuda()
```

#### 小结
在本节中，我们从naive的算法开始，最终得到了一个基于FFT算法的高效实现，并且给出处理单向情形的方案。

### Rpe的实现
注意到Tno的计算涉及到$x,t$，$x$是输入，$t$是相对位置系数，所以下一步就是如何计算$t$。对于序列长度为$n$，特征维度为$d$的模型，我们一共有$(2n-1)\times d$个系数，所以接下来的问题就是如何得到这些系数。


#### Naive实现
最简单的思路就是直接给模型增加$(2n-1)\times d$个参数，但是这样做有几个问题：
1. 当序列长度$n$比较大的时候，模型参数量会非常多；
2. 尽管我们有$(2n-1)\times d$个系数，但是对于每个channel的$2n-1$个系数，不能完全假设他们是独立的，例如$t_1$和$t_{-1}$必然有内在联系；
3. 无法处理任意长的序列；
   1. 这点可以理解为，当超过最大序列长度时，没有对应的系数，所以模型也没有[外推性](https://arxiv.org/abs/2108.12409)；

那么是否有办法解决这些问题呢？回答是肯定的。


#### Relative Position Encoder

对于问题１，２，我们利用某种方式参数化这$(2n-1)\times d$个参数即可，最简单方式就是使用神经网络，特别的，我们使用的是一个名为Relative Position Encoder(RPE)的网络，网络的输入是1维实数$-(n-1), \ldots, (n-1)$，输出是$d$维特征。在使用时，我们会输入$[-(n-1),\ldots, (n-1)]^{\top} \in \mathbb R^{2n-1}$，输出的形状是$(2n-1)\times d$。

对于问题３，我们现在可以一定程度上解决这个问题，现在只要将相对位置（超出训练时的最大训练长度也可）输入到RPE中，即可得到对应系数。但是这样还远远不够，因为这种方式只是让模型“强行”计算了一个值，为了使得性能正常，我们参考了[Alibi](https://arxiv.org/abs/2108.12409)的方案，使用了指数衰减的形式，即：
$$
    \bar t_{i-j}=\lambda^{|i-j|} t_{i-j}, 0< \lambda < 1.
$$
其中$\lambda$是一个超参，我们在$n=512$时选择$\lambda=0.99$。

#### 实现Relative Position Encoder

有了之前的讨论，我们给出Relative Position Encoder的实现，本质是就是一个全连接网络，加上归一化和激活函数：


```python
class Rpe(nn.Module):
    def __init__(
        self, 
        dim, 
        outdim, 
        residual, 
        act="relu", 
        bias=True, 
        layers=3, 
    ):
        super().__init__()
        
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.pos_proj = nn.Linear(1, self.pos_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for i in range(layers):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(self.pos_dim),
                    self.get_act(),
                    nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim, bias=bias),
        )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, biases):
        x = self.pos_proj(biases)
        if self.residual:
            for m in self.layers:
                x = m(x) + x
        else:
            for m in self.layers:
                x = m(x)
        x = self.out(x)

        return x
```

#### 将Tno和Rpe合并

在我们的原始实现中，Rpe是和Tno合并在一起的，完整的实现如下：


```python
class Tno(nn.Module):
    def __init__(
        self, 
        h, 
        dim, 
        rpe_dim, 
        causal=False, 
        use_decay=False, 
        residual=False, 
        act="relu", 
        par_type=1, 
        gamma=0.99,
        bias=True,
        layers=3,
    ):
        super().__init__()
        
        self.h = h
        self.dim = dim
        self.causal = causal
        self.par_type = par_type
        self.zero_value = 0
        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.ones(h, 1, dim) * gamma, requires_grad=False)

        self.rpe = Rpe(
            dim=rpe_dim, 
            outdim=h * dim, 
            residual=residual,
            act=act,
            bias=bias, 
            layers=layers,
        )
        
        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0
        elif self.par_type == 2:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        elif self.par_type == 3:
            index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)
        
        return index
        
    def get_zero(self):
        index = torch.zeros(1).reshape(1, -1) * 1.0
        if self.par_type == 3:
            index = torch.exp(index)
            
        return index

    def get_neg(self, n):
        if self.causal:
            index = torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim) * self.zero_value
        else:
            if self.par_type == 1:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0
            elif self.par_type == 2:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0 / n

        return index
    
    def rpe_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.rpe(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, 'n (h d) -> h n d', h=self.h)

        return res
    
    def forward_causal(self, x, dim=-2):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))

        if self.use_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            gamma = self.gamma
            gamma = gamma ** coef
            pos = gamma * pos
        a = torch.cat([zero, pos, zero], dim=1)
        a = self.act_fun(a)

        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)

        return output
        
    def forward_non_causal(self, x, dim=-2):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))
        neg_index = self.get_neg(n - 1).to(x)
        if self.causal:
            neg = neg_index
        else:
            neg = self.rpe_transform(neg_index)

        if self.use_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            gamma = self.gamma
            gamma = gamma ** coef
            pos = gamma * pos
            neg = torch.flip(gamma, dims=[1]) * neg
        a = torch.cat([zero, pos, zero, neg], dim=1)
        a = self.act_fun(a)
        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)

        return output
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, d
        # a: h, n, d
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output

```

### Tnn layer的实现

有了之前的铺垫，我们可以介绍Tnn Layer，该模块包含一个Token mixer(GTU)以及一个Channel mixer(GLU)，由于GLU和GTU非常相似，所以我们从GLU开始介绍。

#### GLU
[GLU](https://arxiv.org/abs/2002.05202)是利用Gate的形式达到Channel mixing的作用，写成数学公式为：
$$
\mathbf O = [f({\mathbf X} {\mathbf W_1}) \odot ({\mathbf X} {\mathbf W_2})] {\mathbf W_3}.
$$
实现如下：


```python
class GLU(nn.Module):
    def __init__(self, d1, d2, act_fun, fina_act="None", dropout=0.0, bias=True):
        super().__init__()
        
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = get_activation_fn(act_fun)
        self.p = dropout
        if self.p > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.fina_act = get_activation_fn(fina_act)

    def forward(self, x):
        o1 = self.l1(x)
        weight = self.act_fun(o1)
        if self.p > 0.0:
            weight = self.dropout(weight)
        o2 = self.l2(x)
        output = weight * o2
        output = self.l3(output)
        output = self.fina_act(output)

        return output
```

#### GTU

GTU参考了GLU的思路，唯一的不同是在其中一个分支上使用了`Tno`，并且增加一个激活函数，写成数学公式即为：
$$
\mathbf O = [f({\mathbf X} {\mathbf W_1}) \odot (\mathbf T f({\mathbf X} {\mathbf W_2}))] {\mathbf W_3}.
$$
实现如下：


```python
class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
        use_norm=False,
        norm_type="layernorm",
        use_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        gamma=0.99,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.num_heads = num_heads
        self.normalize = normalize

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        self.head_dim = d1 // num_heads
        # linear projection
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        # tno
        self.toep = Tno(
            h=num_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding, 
            causal=causal, 
            use_decay=use_decay, 
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            layers=rpe_layers,
        )
        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
    
    def forward(self, x):
        # x: b, n, d
        num_heads = self.num_heads

        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = u * output
        output = self.o(output)
        
        return output
```

#### TnnLayer

有了之前的准备工作，我们很容易实现出TnnLayer，因为这只不过是GTU和GLU的堆叠：


```python
class TnnLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads,
        rpe_embedding,
        glu_dim,
        # model params
        prenorm=True,
        norm_type="layernorm",
        # gtu params
        causal=False,
        gtu_act="silu",
        expand_ratio=3,
        use_decay=False,
        gamma=0.999,
        # rpe params
        rpe_act="relu",
        rpe_layers=3,
        # glu params
        glu_act="silu",
    ):
        super().__init__()
        self.token_mixer = Gtu(
            # gtu params
            embed_dim=dim,
            num_heads=num_heads,
            act_fun=gtu_act,
            norm_type=norm_type,
            causal=causal,
            expand_ratio=expand_ratio,
            use_decay=use_decay,
            gamma=gamma,
            # rpe params
            rpe_embedding=rpe_embedding,
            rpe_act=rpe_act,
            rpe_layers=rpe_layers,
        )

        self.token_norm = nn.LayerNorm(dim)
        self.feature_norm = nn.LayerNorm(dim)
        
        self.feature_mixer = GLU(
            d1=dim, 
            d2=glu_dim,
            act_fun=glu_act,
        )
    
    def forward(self, x):
        x = x + self.token_mixer(self.token_norm(x))
        x = x + self.feature_mixer(self.feature_norm(x))

        return x

```

在使用时，您只需要将TransformerLayer替换成TnnLayer即可。

#### 小结
在本节中，我们完成了TnnLayer的实现，有了之前的铺垫工作，这一切并不困难。现在，您已经可以将Tnn应用到您的项目中了。


## 全文总结

通过之前的内容，您应该对TNN有所了解，这里，让我们对全文的核心进行总结：

- Transformer可以分为Token mixing和Channel mixing；
- Attention的作用是Token mixing，而相对位置信息对Attention很重要，我们提出使用相对位置信息(Toepltiz matrix)来代替Attention Matrix；
- 使用Toeplitz matrix进行矩阵乘法可以加速，所以我们的方法理论上速度很快；
- Toeplitz matrix的系数可以使用Rpe进行参数化，从而减少参数，结合指数衰减可以得到外推性；

当然，TNN还有很多问题存在，例如：

- 为什么相对位置信息就足够进行序列建模？
- TNN真的只使用了相对位置信息吗？
- TNN能达到理论速度上界吗？
- TNN不能做哪些任务？
- TNN有哪些先验假设？

关于这些问题，我们将在后续的博客中回答，期待您的再次阅读。
