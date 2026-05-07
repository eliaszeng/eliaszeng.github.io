# 第一周教材：Python核心工程能力 + PyTorch深度学习基础

> **阶段**: P1 基础原理 | **日期**: 2026/04/07 – 04/13 | **总学时**: ~23.5h
>
> **本周定位**: 你已有 Python/PyTorch 实战经验，本周不是"从零学起"，而是**把你用过但没深挖过的底层机制搞透**，达到"面试时被追问五层 why 不慌"的程度。

---

## 学习导航

| 日期 | 主题 | 核心产出 | 时长 |
|------|------|----------|------|
| 周一 04/07 | Python核心+并发 | 并发对比notebook | 3h |
| 周二 04/08 | Python元编程+设计模式 | 装饰器库+设计模式 | 3h |
| 周三 04/09 | PyTorch: Tensor+Autograd | 反向传播实验 | 3h |
| 周四 04/10 | PyTorch: Module+优化器+数据 | 优化器+AMP实验 | 3h |
| 周五 04/11 | 【实战】micrograd引擎 | micrograd (GitHub) | 4h |
| 周六 04/12 | CNN+ResNet+ViT | ResNet+ViT实现 | 5h |
| 周日 04/13 | 复盘+工程深挖① | QA笔记+ConSE工程复盘 | 2.5h |

---

# Day 1 (周一): Python核心机制与并发编程

## 学习目标

面试验收标准：能脱稿回答以下问题——

1. Python的引用计数+分代GC是怎么工作的？循环引用怎么处理？
2. GIL是什么？为什么有了GIL多线程还有用？什么时候用多进程？什么时候用asyncio？
3. `__slots__`的原理和适用场景？

---

## 1.1 Python对象模型

### 1.1.1 一切皆对象

Python中一切都是对象——整数、函数、类本身都是对象，都继承自`object`，都有`id`、`type`和`value`三要素。

```python
# 验证"一切皆对象"
x = 42
print(type(x))          # <class 'int'>
print(isinstance(x, object))  # True

def foo(): pass
print(type(foo))         # <class 'function'>
print(isinstance(foo, object))  # True

print(type(int))         # <class 'type'>
print(isinstance(int, object)) # True
```

**面试关键点**: Python变量不是"盒子"（存放值），而是"标签"（指向对象的引用）。这解释了为什么 `a = [1,2,3]; b = a; b.append(4)` 会同时修改 `a`。

### 1.1.2 可变 vs 不可变

| 不可变 (immutable) | 可变 (mutable) |
|---|---|
| int, float, str, tuple, frozenset, bytes | list, dict, set, bytearray |

不可变对象的"修改"实际是创建新对象：

```python
a = "hello"
print(id(a))  # 比如 140234567890
a += " world"
print(id(a))  # 变了！140234567999 —— 新对象
```

**工程意义**: 不可变对象可以作为dict的key、可以被hash、线程安全。这就是为什么PyTorch的`torch.Size`是tuple。

### 1.1.3 小整数缓存与字符串驻留 (intern)

CPython缓存 [-5, 256] 的整数对象，这是面试中常见的"坑"：

```python
a = 256; b = 256
print(a is b)  # True —— 缓存范围内，同一对象

a = 257; b = 257
print(a is b)  # False（在交互式环境中）—— 超出缓存
# 注意：在.py文件中可能为True（编译器优化）
```

**面试提醒**: `is` 比较的是 id（内存地址），`==` 比较的是值。永远用 `==` 比较值，用 `is` 只比较 `None`。

---

## 1.2 内存管理与垃圾回收

### 1.2.1 引用计数 (Reference Counting)

CPython的主要GC机制。每个对象有一个引用计数器`ob_refcnt`：

```python
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (a本身 + getrefcount的参数)

b = a
print(sys.getrefcount(a))  # 3

del b
print(sys.getrefcount(a))  # 2
```

引用计数 +1 的场景：赋值、函数参数传递、放入容器。
引用计数 -1 的场景：`del`、变量被重新赋值、离开作用域、从容器中移除。

**引用计数归零 → 立即释放内存。** 这是Python GC的第一道防线，大部分对象都通过这种方式回收。

### 1.2.2 循环引用与分代GC

引用计数无法处理循环引用：

```python
a = []
b = []
a.append(b)  # a -> b
b.append(a)  # b -> a
del a, del b
# 此时两个list对象的引用计数都是1（互相引用），永远不会归零
```

CPython的分代垃圾回收器解决这个问题：

| 代 | 含义 | 触发条件 |
|---|---|---|
| Gen 0 | 新创建的对象 | 分配数 - 释放数 > 阈值(700) |
| Gen 1 | 经历过一次Gen0回收存活的对象 | Gen0回收10次后触发 |
| Gen 2 | 经历过一次Gen1回收存活的对象 | Gen1回收10次后触发 |

```python
import gc

# 查看阈值
print(gc.get_threshold())  # (700, 10, 10)

# 查看各代对象数
print(gc.get_count())  # (当前Gen0计数, Gen1计数, Gen2计数)

# 手动触发全量GC
gc.collect()

# 查看GC统计
print(gc.get_stats())
```

**分代GC算法核心**：标记-清除。从根对象（全局变量、栈上变量）出发，标记所有可达对象。不可达的对象被清除。

**面试高频追问**：

Q: 为什么不只用标记-清除？
A: 标记-清除需要遍历所有对象，开销大。引用计数能立即回收大部分对象，分代GC只需要处理循环引用。

Q: `gc.disable()` 有什么影响？
A: 只禁用分代GC，引用计数仍在工作。适用于已知不产生循环引用的性能敏感场景（如PyTorch训练循环中某些场景）。

Q: `__del__` 有什么问题？
A: 循环引用中的对象如果定义了`__del__`，GC无法确定析构顺序，在Python 3.4之前会放入`gc.garbage`不回收。3.4+通过PEP 442改进，但仍应避免依赖`__del__`。

### 1.2.3 `weakref` 弱引用

不增加引用计数的引用方式，常用于缓存和观察者模式：

```python
import weakref

class Node:
    def __init__(self, value):
        self.value = value

obj = Node(42)
ref = weakref.ref(obj)

print(ref())  # <Node object>
del obj
print(ref())  # None —— 对象已被回收
```

**工程应用**: PyTorch中 `torch.utils.hooks` 使用弱引用避免循环引用导致模型无法释放。

### 1.2.4 `__slots__`

默认情况下，Python对象使用 `__dict__` 存储属性，每个实例都有一个字典。`__slots__` 改为固定的属性槽：

```python
class PointDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PointSlots:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys
pd = PointDict(1, 2)
ps = PointSlots(1, 2)
print(sys.getsizeof(pd.__dict__))  # ~104 bytes (字典的开销)
# ps没有__dict__
```

**使用场景**: 创建大量小对象时（如数据加载器中的数据记录），`__slots__` 能显著减少内存。注意：使用 `__slots__` 后无法动态添加属性，子类也需要声明 `__slots__`。

---

## 1.3 GIL：全局解释器锁

### 1.3.1 GIL是什么

GIL (Global Interpreter Lock) 是CPython中的一把全局互斥锁，保证同一时刻只有一个线程执行Python字节码。

**为什么需要GIL**: CPython的引用计数不是线程安全的。如果两个线程同时修改引用计数，可能导致内存泄漏或提前释放。GIL是最简单的保护方案。

**GIL的释放时机**：
1. 每执行100条字节码(Python 2) / 每5ms(Python 3.2+) 主动释放
2. I/O操作时释放（文件读写、网络请求、`time.sleep`）
3. 调用C扩展时可以选择释放（NumPy/PyTorch的底层运算会释放GIL）

### 1.3.2 GIL对并发的影响

```
CPU密集型任务:
  多线程 → 实际串行执行（GIL的限制），甚至因为线程切换更慢
  多进程 → 真正并行（每个进程有自己的GIL）

I/O密集型任务:
  多线程 → 有效！线程在等I/O时释放GIL
  asyncio  → 更高效！单线程内协程切换，无线程创建开销
```

### 1.3.3 threading模块

```python
import threading
import time

counter = 0
lock = threading.Lock()

def increment(n):
    global counter
    for _ in range(n):
        with lock:  # 必须加锁保护共享数据
            counter += 1

threads = [threading.Thread(target=increment, args=(100000,)) for _ in range(4)]
start = time.time()
for t in threads: t.start()
for t in threads: t.join()
print(f"Result: {counter}, Time: {time.time() - start:.3f}s")
```

**面试关键点**: `threading.Lock` 只保证Python层面的互斥，不是替代GIL的方案。GIL保护的是CPython内部数据结构（引用计数），Lock保护的是你的业务数据。

### 1.3.4 multiprocessing模块

```python
import multiprocessing as mp
import time

def cpu_heavy(n):
    """CPU密集型：计算质数"""
    count = 0
    for i in range(2, n):
        if all(i % j != 0 for j in range(2, int(i**0.5)+1)):
            count += 1
    return count

if __name__ == '__main__':
    n = 100000

    # 串行
    start = time.time()
    results = [cpu_heavy(n // 4) for _ in range(4)]
    serial_time = time.time() - start

    # 多进程
    start = time.time()
    with mp.Pool(4) as pool:
        results = pool.map(cpu_heavy, [n // 4] * 4)
    parallel_time = time.time() - start

    print(f"Serial: {serial_time:.2f}s")
    print(f"Parallel (4 workers): {parallel_time:.2f}s")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
```

**工程注意**: 多进程的进程间通信(IPC)有开销。小任务不值得用多进程。PyTorch的`DataLoader(num_workers=N)` 本质就是多进程加载数据。

### 1.3.5 asyncio异步编程

asyncio是单线程、事件循环驱动的并发模型，特别适合I/O密集型场景：

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [f"https://httpbin.org/delay/1" for _ in range(5)]

    async with aiohttp.ClientSession() as session:
        # 并发发起所有请求
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        print(f"Fetched {len(results)} pages")

start = time.time()
asyncio.run(main())
print(f"Time: {time.time() - start:.2f}s")  # ~1s, 不是5s
```

**核心概念**:
- `async def` 定义协程函数，调用返回协程对象
- `await` 挂起当前协程，让出控制权给事件循环
- `asyncio.gather()` 并发运行多个协程
- 本质是协作式多任务：协程在 `await` 处主动让出CPU

**面试追问**: asyncio vs 多线程？

| 维度 | asyncio | threading |
|------|---------|-----------|
| 并发模型 | 协作式 (cooperative) | 抢占式 (preemptive) |
| 线程数 | 1个 | N个 |
| 切换开销 | 极低（函数调用级） | 较高（OS上下文切换） |
| 共享数据 | 天然安全（单线程） | 需要加锁 |
| 适用场景 | 大量I/O并发（>100个连接） | 少量I/O并发、需要使用阻塞库 |

---

## 1.4 动手实践：并发对比Benchmark

### 任务描述

创建一个notebook，对比三种并发模型在CPU密集和I/O密集场景下的性能差异。

### 代码模板

```python
"""
Day1 产出物: 并发对比 Benchmark Notebook
验收标准:
1. CPU密集场景：多进程 > 串行 > 多线程（证明GIL的影响）
2. I/O密集场景：asyncio ≈ 多线程 >> 串行（证明GIL在I/O时释放）
3. 包含清晰的性能对比表格和结论
"""

import time
import threading
import multiprocessing as mp
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt

# ============ CPU 密集型任务 ============

def cpu_task(n=5_000_000):
    """纯CPU计算：求和"""
    total = 0
    for i in range(n):
        total += i * i
    return total

def benchmark_cpu():
    n_tasks = 4
    results = {}

    # 串行
    start = time.perf_counter()
    for _ in range(n_tasks):
        cpu_task()
    results['串行'] = time.perf_counter() - start

    # 多线程
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as executor:
        list(executor.map(lambda _: cpu_task(), range(n_tasks)))
    results['多线程'] = time.perf_counter() - start

    # 多进程
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_tasks) as executor:
        list(executor.map(lambda _: cpu_task(), range(n_tasks)))
    results['多进程'] = time.perf_counter() - start

    return results

# ============ I/O 密集型任务 ============

def io_task(duration=0.5):
    """模拟I/O等待"""
    time.sleep(duration)
    return True

async def async_io_task(duration=0.5):
    """异步版I/O等待"""
    await asyncio.sleep(duration)
    return True

def benchmark_io():
    n_tasks = 10
    results = {}

    # 串行
    start = time.perf_counter()
    for _ in range(n_tasks):
        io_task()
    results['串行'] = time.perf_counter() - start

    # 多线程
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_tasks) as executor:
        list(executor.map(lambda _: io_task(), range(n_tasks)))
    results['多线程'] = time.perf_counter() - start

    # asyncio
    async def run_async():
        tasks = [async_io_task() for _ in range(n_tasks)]
        await asyncio.gather(*tasks)

    start = time.perf_counter()
    asyncio.run(run_async())
    results['asyncio'] = time.perf_counter() - start

    return results

# ============ 运行并可视化 ============

if __name__ == '__main__':
    print("=" * 50)
    print("CPU 密集型 Benchmark")
    print("=" * 50)
    cpu_results = benchmark_cpu()
    for method, t in cpu_results.items():
        print(f"  {method}: {t:.3f}s")

    print("\n" + "=" * 50)
    print("I/O 密集型 Benchmark")
    print("=" * 50)
    io_results = benchmark_io()
    for method, t in io_results.items():
        print(f"  {method}: {t:.3f}s")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(cpu_results.keys(), cpu_results.values(), color=['#2196F3', '#FF9800', '#4CAF50'])
    axes[0].set_title('CPU密集型任务', fontsize=14)
    axes[0].set_ylabel('耗时(秒)')

    axes[1].bar(io_results.keys(), io_results.values(), color=['#2196F3', '#FF9800', '#9C27B0'])
    axes[1].set_title('I/O密集型任务', fontsize=14)
    axes[1].set_ylabel('耗时(秒)')

    plt.tight_layout()
    plt.savefig('concurrency_benchmark.png', dpi=150)
    print("\n图表已保存: concurrency_benchmark.png")
```

### 预期结果与分析

运行后你应该看到类似的结果：

```
CPU 密集型:
  串行:   ~4.0s
  多线程: ~4.5s  ← 比串行还慢！GIL + 线程切换开销
  多进程: ~1.2s  ← 接近线性加速

I/O 密集型:
  串行:   ~5.0s
  多线程: ~0.5s  ← GIL在sleep时释放
  asyncio: ~0.5s ← 同样并发
```

### 验收检查

- [ ] notebook能运行，输出CPU密集和I/O密集两组结果
- [ ] CPU密集型中多进程明显快于多线程和串行
- [ ] I/O密集型中多线程和asyncio明显快于串行
- [ ] 能口头解释每组结果为什么是这样

---

# Day 2 (周二): Python元编程与设计模式

## 学习目标

面试验收标准：

1. 装饰器的三层结构（带参数）是怎样的？`functools.wraps` 有什么用？
2. 描述符协议是什么？`@property` 的底层原理？
3. 元类有什么实际应用场景？
4. 工厂/策略/观察者模式在ML工程中的应用？

---

## 2.1 装饰器深入

### 2.1.1 装饰器本质

装饰器本质是一个接受函数作为参数、返回新函数的高阶函数：

```python
# @decorator 语法等价于:
# func = decorator(func)

def timer(func):
    import time
    import functools

    @functools.wraps(func)  # 保留原函数的__name__, __doc__等
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    """This is a slow function"""
    import time; time.sleep(1)

slow_function()
print(slow_function.__name__)  # "slow_function"（而不是"wrapper"，感谢wraps）
print(slow_function.__doc__)   # "This is a slow function"
```

**`functools.wraps` 的重要性**: 没有它，被装饰的函数会丢失 `__name__`、`__doc__`、`__module__` 等元信息。在调试和日志中这很关键。

### 2.1.2 三层装饰器（带参数）

```python
import functools
import time
import logging

def retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):
    """带参数的重试装饰器 — 三层结构"""
    def decorator(func):  # 第二层：接收被装饰的函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # 第三层：实际执行逻辑
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logging.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {e}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator  # 第一层返回真正的装饰器

@retry(max_attempts=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
def fetch_data(url):
    """从API获取数据"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("Connection failed")
    return {"data": "success"}
```

**记忆方法**: 带参数装饰器 = 装饰器工厂。`retry(max_attempts=3)` 返回一个装饰器，这个装饰器再装饰 `fetch_data`。

### 2.1.3 实用装饰器库

```python
import functools
import time
import logging
from typing import Any, Callable
from collections import OrderedDict

# ---- 1. @timer: 计时装饰器 ----
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# ---- 2. @cache_with_ttl: 带过期时间的缓存 ----
def cache_with_ttl(ttl_seconds=300, maxsize=128):
    """比functools.lru_cache多了TTL过期功能"""
    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args):
            now = time.time()
            # 检查缓存
            if args in cache:
                result, timestamp = cache[args]
                if now - timestamp < ttl_seconds:
                    cache.move_to_end(args)  # LRU更新
                    return result
                else:
                    del cache[args]  # 过期删除

            # 计算并缓存
            result = func(*args)
            cache[args] = (result, now)

            # 淘汰最旧的
            while len(cache) > maxsize:
                cache.popitem(last=False)

            return result

        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache), 'maxsize': maxsize, 'ttl': ttl_seconds
        }
        return wrapper
    return decorator

# ---- 3. @retry (见上面的实现) ----

# ---- 4. @validate_types: 简易类型检查 ----
def validate_types(**expected_types):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, expected_type in expected_types.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' expected "
                            f"{expected_type.__name__}, got "
                            f"{type(value).__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ---- 使用示例 ----
@timer
@cache_with_ttl(ttl_seconds=60)
def expensive_computation(n):
    """模拟耗时计算"""
    time.sleep(0.1)
    return sum(i**2 for i in range(n))

@validate_types(learning_rate=float, epochs=int)
def train(learning_rate, epochs):
    print(f"Training with lr={learning_rate}, epochs={epochs}")
```

---

## 2.2 描述符协议

描述符是实现了 `__get__`、`__set__`、`__delete__` 中至少一个的对象。`@property`、`@classmethod`、`@staticmethod` 底层都是描述符。

```python
class TypedAttribute:
    """描述符：带类型检查的属性"""
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __set_name__(self, owner, name):
        self.storage_name = f'_typed_{name}'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.storage_name, None)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        setattr(obj, self.storage_name, value)

class ModelConfig:
    learning_rate = TypedAttribute('learning_rate', float)
    batch_size = TypedAttribute('batch_size', int)
    model_name = TypedAttribute('model_name', str)

    def __init__(self, lr, bs, name):
        self.learning_rate = lr
        self.batch_size = bs
        self.model_name = name

# 使用
config = ModelConfig(0.001, 32, "resnet50")
config.learning_rate = 0.01   # OK
# config.learning_rate = "fast"  # TypeError!
```

**工程应用**: PyTorch的 `nn.Parameter` 就利用了描述符机制，使得 `model.weight` 既能像普通属性访问，又能被 `model.parameters()` 自动收集。

---

## 2.3 元类 (Metaclass)

元类是"类的类"——控制类的创建过程。普通类是 `type` 的实例。

```python
# 元类：自动注册所有子类
class RegistryMeta(type):
    """自动注册所有子类的元类"""
    _registry = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # 跳过基类本身的注册
        if bases:  # 只注册有父类的（即子类）
            RegistryMeta._registry[name] = cls
        return cls

    @classmethod
    def get_registry(mcs):
        return dict(mcs._registry)

class BaseModel(metaclass=RegistryMeta):
    """基类：所有子类自动注册"""
    def predict(self, x):
        raise NotImplementedError

class ResNetModel(BaseModel):
    def predict(self, x):
        return f"ResNet predicting {x}"

class ViTModel(BaseModel):
    def predict(self, x):
        return f"ViT predicting {x}"

# 使用注册表
print(RegistryMeta.get_registry())
# {'ResNetModel': <class 'ResNetModel'>, 'ViTModel': <class 'ViTModel'>}

# 工厂模式：根据名字创建模型
def create_model(name):
    registry = RegistryMeta.get_registry()
    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(registry.keys())}")
    return registry[name]()

model = create_model("ResNetModel")
print(model.predict([1,2,3]))  # "ResNet predicting [1, 2, 3]"
```

**面试关键**: 元类通常不需要直接使用。大部分场景用类装饰器或 `__init_subclass__`(Python 3.6+) 更简洁。但理解元类能帮你看懂PyTorch/Django等框架的源码。

```python
# 更简洁的替代方案：__init_subclass__
class BaseModel:
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseModel._registry[cls.__name__] = cls
```

---

## 2.4 设计模式 (ML工程常用)

### 2.4.1 工厂模式

```python
class ModelFactory:
    """模型工厂：根据配置创建模型"""
    _creators = {}

    @classmethod
    def register(cls, name):
        def decorator(model_cls):
            cls._creators[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._creators:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available: {list(cls._creators.keys())}"
            )
        return cls._creators[name](**kwargs)

@ModelFactory.register("resnet50")
class ResNet50:
    def __init__(self, num_classes=1000, pretrained=True):
        self.num_classes = num_classes
        self.pretrained = pretrained

@ModelFactory.register("vit_base")
class ViTBase:
    def __init__(self, num_classes=1000, patch_size=16):
        self.num_classes = num_classes
        self.patch_size = patch_size

# 使用：配置驱动
config = {"model": "resnet50", "num_classes": 10, "pretrained": False}
model = ModelFactory.create(config.pop("model"), **config)
```

**工程价值**: 训练脚本可以通过YAML配置文件切换模型，不需要修改代码。这就是MMDetection、HuggingFace等框架的核心设计。

### 2.4.2 策略模式

```python
from abc import ABC, abstractmethod

class AugmentationStrategy(ABC):
    @abstractmethod
    def augment(self, image):
        pass

class TrainAugmentation(AugmentationStrategy):
    def augment(self, image):
        # 随机翻转、裁剪、颜色抖动...
        return f"train_aug({image})"

class EvalAugmentation(AugmentationStrategy):
    def augment(self, image):
        # 只做resize和normalize
        return f"eval_aug({image})"

class TestTimeAugmentation(AugmentationStrategy):
    def augment(self, image):
        # 多尺度+翻转ensemble
        return f"tta_aug({image})"

class DataPipeline:
    def __init__(self, strategy: AugmentationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: AugmentationStrategy):
        self._strategy = strategy

    def process(self, image):
        return self._strategy.augment(image)

# 使用
pipeline = DataPipeline(TrainAugmentation())
print(pipeline.process("img.jpg"))  # train_aug(img.jpg)

pipeline.set_strategy(EvalAugmentation())
print(pipeline.process("img.jpg"))  # eval_aug(img.jpg)
```

### 2.4.3 观察者模式

```python
from typing import List, Callable, Dict, Any

class TrainingEventEmitter:
    """训练事件发射器 — PyTorch Lightning/Keras callback的简化版"""
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event: str, callback: Callable):
        self._listeners.setdefault(event, []).append(callback)
        return self  # 链式调用

    def emit(self, event: str, **kwargs):
        for callback in self._listeners.get(event, []):
            callback(**kwargs)

# 定义各种回调
def log_metrics(epoch, loss, **kwargs):
    print(f"[LOG] Epoch {epoch}, Loss: {loss:.4f}")

def save_checkpoint(epoch, loss, **kwargs):
    if loss < 0.1:
        print(f"[SAVE] Saving checkpoint at epoch {epoch}")

def early_stop(epoch, loss, patience_counter=[0], best_loss=[float('inf')], **kwargs):
    if loss < best_loss[0]:
        best_loss[0] = loss
        patience_counter[0] = 0
    else:
        patience_counter[0] += 1
        if patience_counter[0] >= 5:
            print(f"[STOP] Early stopping at epoch {epoch}")

# 组装
trainer = TrainingEventEmitter()
trainer.on("epoch_end", log_metrics)
trainer.on("epoch_end", save_checkpoint)
trainer.on("epoch_end", early_stop)

# 模拟训练
for epoch in range(10):
    loss = 1.0 / (epoch + 1)
    trainer.emit("epoch_end", epoch=epoch, loss=loss)
```

**工程价值**: PyTorch Lightning 的 Callback系统、HuggingFace Trainer 的回调都是观察者模式。理解这个模式能帮你快速自定义训练流程。

---

## 2.5 Day2 产出物验收

### 验收清单

- [ ] `@timer`, `@retry`, `@cache_with_ttl` 三个装饰器能独立运行
- [ ] 元类注册表或 `__init_subclass__` 注册表能工作
- [ ] 工厂模式的 `ModelFactory` 能根据配置创建不同模型
- [ ] 策略模式和观察者模式能独立运行
- [ ] 能口头解释：装饰器三层结构的每一层是什么；描述符协议的三个方法

---

# Day 3 (周三): PyTorch Tensor与Autograd

## 学习目标

面试验收标准：

1. Tensor的 Storage、Stride、View 是什么关系？什么时候需要 `contiguous()`？
2. `requires_grad=True` 后发生了什么？计算图怎么构建的？反向传播怎么执行的？
3. `torch.no_grad()` vs `detach()` 的区别？什么时候用哪个？

---

## 3.1 Tensor底层：Storage + Stride

### 3.1.1 Storage：实际数据

Tensor不直接存数据，而是通过一个一维的 `Storage` 存储实际数据。多个Tensor可以共享同一个Storage。

```python
import torch

a = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

# Storage 是一维连续内存
print(a.storage())       # [1, 2, 3, 4, 5, 6]
print(a.storage_offset())  # 0 — 从Storage的第0个元素开始

# 切片共享Storage
b = a[1]  # [4, 5, 6]
print(b.storage())       # [1, 2, 3, 4, 5, 6] — 同一个Storage！
print(b.storage_offset())  # 3 — 从第3个元素开始
print(a.data_ptr() == b.storage().data_ptr())  # True — 同一块内存
```

### 3.1.2 Stride：索引映射

Stride定义了在每个维度上移动一个元素需要在Storage中跳过多少个元素：

```python
a = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])
print(a.stride())  # (3, 1)
# 含义: 沿dim0移动1步 → 跳3个元素 (1→4)
#        沿dim1移动1步 → 跳1个元素 (1→2)

# 转置不复制数据，只改变stride
b = a.t()
print(b.stride())  # (1, 3)
print(b.is_contiguous())  # False！
# 此时b在内存中不是连续的
```

### 3.1.3 View vs Contiguous

`view()` 要求Tensor是连续的（contiguous）；`reshape()` 自动处理非连续情况：

```python
a = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

# view: 不复制数据，要求contiguous
b = a.view(6)     # OK, a是连续的
c = a.t()          # 转置后不连续
# c.view(6)       # RuntimeError! not contiguous

# 解决方案
d = c.contiguous().view(6)   # 先复制为连续内存，再view
e = c.reshape(6)              # reshape自动处理（如果不连续会复制）
```

**面试关键记忆**: `view` = 零拷贝 reshape（要求连续），`reshape` = 智能 reshape（必要时拷贝），`contiguous()` = 复制为行优先连续存储。

### 3.1.4 实战：Tensor内存实验

```python
import torch

def tensor_memory_experiment():
    """验证Storage/Stride/View的行为"""

    print("=" * 50)
    print("实验1: Storage共享")
    print("=" * 50)
    a = torch.arange(12).reshape(3, 4)
    b = a[1:3, 1:3]  # 切片
    print(f"a.data_ptr() == b.storage().data_ptr(): "
          f"{a.storage().data_ptr() == b.storage().data_ptr()}")
    print(f"b = {b}")
    b[0, 0] = 999
    print(f"修改b后, a = \n{a}")  # a也被修改了！

    print("\n" + "=" * 50)
    print("实验2: Stride与连续性")
    print("=" * 50)
    a = torch.arange(12).reshape(3, 4)
    print(f"a.stride() = {a.stride()}")
    print(f"a.is_contiguous() = {a.is_contiguous()}")

    b = a.t()
    print(f"a.t().stride() = {b.stride()}")
    print(f"a.t().is_contiguous() = {b.is_contiguous()}")

    c = a.permute(1, 0)
    print(f"a.permute(1,0).stride() = {c.stride()}")

    print("\n" + "=" * 50)
    print("实验3: view vs reshape vs contiguous")
    print("=" * 50)
    a = torch.arange(12).reshape(3, 4).t()  # 非连续
    print(f"a.is_contiguous() = {a.is_contiguous()}")

    try:
        a.view(-1)
    except RuntimeError as e:
        print(f"a.view(-1) → RuntimeError: {e}")

    b = a.reshape(-1)
    print(f"a.reshape(-1) works: {b}")
    print(f"共享内存? {a.storage().data_ptr() == b.storage().data_ptr()}")
    # False — reshape对非连续tensor会复制

    c = a.contiguous()
    print(f"a.contiguous() 共享内存? "
          f"{a.storage().data_ptr() == c.storage().data_ptr()}")
    # False — contiguous创建新的连续存储

tensor_memory_experiment()
```

---

## 3.2 Autograd：自动微分

### 3.2.1 计算图概念

PyTorch使用**动态计算图**（Define-by-Run）：前向传播时动态构建计算图，反向传播时自动计算梯度后图被销毁。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 前向传播：构建计算图
z = x * y + x ** 2  # z = xy + x²

# 每个tensor知道自己是怎么来的
print(z.grad_fn)             # <AddBackward0>
print(z.grad_fn.next_functions)  # (MulBackward0, PowBackward0)

# 反向传播：计算 dz/dx 和 dz/dy
z.backward()
print(f"dz/dx = {x.grad}")  # y + 2x = 3 + 4 = 7
print(f"dz/dy = {y.grad}")  # x = 2
```

### 3.2.2 计算图的关键特性

```python
# 1. 图是动态的 — 每次前向传播可以不同
x = torch.randn(3, requires_grad=True)
if x.sum() > 0:
    y = x * 2
else:
    y = x * 3
y.sum().backward()  # 根据实际执行路径计算梯度

# 2. 非叶子节点的梯度默认不保留
a = torch.tensor(2.0, requires_grad=True)  # 叶子节点
b = a * 3                                    # 非叶子
c = b + 1
c.backward()
print(a.grad)  # tensor(3.) — 叶子节点梯度保留
print(b.grad)  # None — 非叶子节点梯度不保留

# 如果需要，使用 retain_grad()
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
b.retain_grad()
c = b + 1
c.backward()
print(b.grad)  # tensor(1.) — 现在有了

# 3. 默认反向传播后图被销毁
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
b.backward()
# b.backward()  # RuntimeError! 图已销毁
# 解决: b.backward(retain_graph=True)

# 4. 梯度是累加的！
a = torch.tensor(2.0, requires_grad=True)
for _ in range(3):
    b = a * 3
    b.backward()
print(a.grad)  # tensor(9.) 而不是 3.0 — 累加了3次
# 所以训练循环中必须: optimizer.zero_grad()
```

### 3.2.3 `torch.no_grad()` vs `detach()`

```python
x = torch.tensor(2.0, requires_grad=True)

# torch.no_grad(): 上下文管理器，内部所有操作不追踪梯度
with torch.no_grad():
    y = x * 3
    print(y.requires_grad)  # False
# 用途: 推理时节省内存和计算

# detach(): 从计算图中分离出一个tensor
z = x * 3
w = z.detach()  # w与z共享数据，但w不在计算图中
print(w.requires_grad)  # False
# 用途: 阻断梯度流动（如GAN训练中冻结判别器）
```

| 方法 | 作用范围 | 数据共享 | 典型用途 |
|------|---------|---------|---------|
| `torch.no_grad()` | 整个代码块 | N/A | 推理、评估 |
| `detach()` | 单个tensor | 是 | 阻断梯度流、target构造 |
| `.data` | 单个tensor | 是 | 不推荐（不安全） |

### 3.2.4 手写反向传播

```python
"""
手写3层全连接网络的前向+反向传播，然后用PyTorch验证。
这个练习的目的是确保你真正理解链式法则在代码中的体现。
"""
import torch
import torch.nn.functional as F

# ========== 手动实现 ==========

def manual_forward_backward():
    torch.manual_seed(42)

    # 初始化（与PyTorch相同的参数）
    x = torch.randn(4, 3)   # batch=4, input=3
    y = torch.randint(0, 2, (4,))  # 二分类标签

    W1 = torch.randn(3, 5, requires_grad=True)
    b1 = torch.zeros(5, requires_grad=True)
    W2 = torch.randn(5, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)

    # --- 前向传播 ---
    # Layer 1: linear + ReLU
    z1 = x @ W1 + b1          # (4, 5)
    a1 = torch.relu(z1)       # (4, 5)

    # Layer 2: linear + softmax + cross entropy
    z2 = a1 @ W2 + b2         # (4, 2)
    loss = F.cross_entropy(z2, y)

    # --- 反向传播（手动） ---
    # d(loss)/d(z2) = softmax(z2) - one_hot(y)  (cross entropy + softmax 的联合梯度)
    probs = F.softmax(z2, dim=1)        # (4, 2)
    one_hot_y = F.one_hot(y, 2).float() # (4, 2)
    dz2 = (probs - one_hot_y) / x.shape[0]  # (4, 2), 除以batch_size

    # d(loss)/d(W2) = a1^T @ dz2
    dW2_manual = a1.t() @ dz2  # (5, 2)
    db2_manual = dz2.sum(dim=0)  # (2,)

    # d(loss)/d(a1) = dz2 @ W2^T
    da1 = dz2 @ W2.t()  # (4, 5)

    # ReLU backward: 梯度 * (z1 > 0)
    dz1 = da1 * (z1 > 0).float()  # (4, 5)

    # d(loss)/d(W1) = x^T @ dz1
    dW1_manual = x.t() @ dz1  # (3, 5)
    db1_manual = dz1.sum(dim=0)  # (5,)

    # --- PyTorch自动反向传播验证 ---
    loss.backward()

    # 对比
    print("W2 grad match:", torch.allclose(dW2_manual, W2.grad, atol=1e-5))
    print("b2 grad match:", torch.allclose(db2_manual, b2.grad, atol=1e-5))
    print("W1 grad match:", torch.allclose(dW1_manual, W1.grad, atol=1e-5))
    print("b1 grad match:", torch.allclose(db1_manual, b1.grad, atol=1e-5))

manual_forward_backward()
```

---

## 3.3 Day3 产出物验收

- [ ] Tensor内存实验运行通过，能解释Storage/Stride/contiguous
- [ ] 手写反向传播与PyTorch autograd结果一致
- [ ] 能口头回答：什么时候需要 `contiguous()`？`no_grad()` vs `detach()`？

---

# Day 4 (周四): PyTorch Module、优化器与数据管道

## 学习目标

面试验收标准：

1. `nn.Module` 的 `__init__` 和 `forward` 有什么约定？`parameters()` 怎么自动收集参数的？
2. Adam和AdamW的区别？为什么Transformer训练偏好AdamW？
3. AMP混合精度的原理？为什么loss要scale？
4. `DataLoader` 的 `num_workers` 和 `pin_memory` 怎么设？

---

## 4.1 nn.Module深入

### 4.1.1 Module的核心机制

```python
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Parameter 自动注册为参数
        self.weight = nn.Parameter(torch.randn(3, 5))
        # 子Module 自动注册
        self.linear = nn.Linear(5, 2)
        # 普通tensor不会被parameters()收集
        self.buffer = torch.randn(3)
        # 注册buffer: 不参与梯度计算，但会被state_dict保存
        self.register_buffer('running_mean', torch.zeros(5))

    def forward(self, x):
        return self.linear(x @ self.weight)

model = MyModule()

# parameters() 递归收集所有 nn.Parameter
print("Parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
# weight: (3, 5)
# linear.weight: (2, 5)
# linear.bias: (2,)

# buffers() 收集所有 register_buffer
print("\nBuffers:")
for name, buf in model.named_buffers():
    print(f"  {name}: {buf.shape}")
# running_mean: (5,)

# state_dict() 包含parameters和buffers
print("\nState dict keys:", list(model.state_dict().keys()))
```

**`parameters()` 的自动收集原理**: `nn.Module` 重写了 `__setattr__`。当你写 `self.linear = nn.Linear(...)` 时，`__setattr__` 检测到值是 `nn.Module` 或 `nn.Parameter`，自动注册到内部的 `_modules` 或 `_parameters` 字典中。

### 4.1.2 Hook机制

Hook是PyTorch的AOP（面向切面编程），用于在不修改forward代码的前提下插入调试/分析逻辑：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# ---- Forward Hook: 捕获中间层输出 ----
activations = {}

def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# 注册到每一层
for name, layer in model.named_modules():
    if name:  # 跳过顶层Sequential
        layer.register_forward_hook(get_activation(name))

# 前向传播
x = torch.randn(2, 784)
output = model(x)

print("Captured activations:")
for name, act in activations.items():
    print(f"  {name}: shape={act.shape}, "
          f"mean={act.mean():.4f}, std={act.std():.4f}")

# ---- Backward Hook: 检查梯度 ----
def grad_hook(name):
    def hook(module, grad_input, grad_output):
        print(f"[GRAD] {name}: "
              f"grad_output norm = {grad_output[0].norm():.4f}")
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        layer.register_full_backward_hook(grad_hook(name))

loss = output.sum()
loss.backward()
```

**工程应用场景**:
- 特征可视化（捕获中间层激活）
- 梯度裁剪诊断（检查梯度是否爆炸/消失）
- 模型压缩分析（统计各层激活的稀疏度）
- 你的ConSE项目：可以用hook分析CLIP各层对建筑场景的响应

### 4.1.3 state_dict 保存与加载

```python
# 保存（只保存参数，不保存模型结构）
torch.save(model.state_dict(), 'model_weights.pth')

# 加载
model = MyModel()  # 先创建模型结构
state_dict = torch.load('model_weights.pth', weights_only=True)
model.load_state_dict(state_dict, strict=True)
# strict=True: 要求key完全匹配
# strict=False: 允许缺失/多余的key（用于迁移学习）

# 部分加载（常见于微调）
pretrained = torch.load('pretrained.pth', weights_only=True)
model_dict = model.state_dict()
# 过滤不匹配的key
pretrained = {k: v for k, v in pretrained.items()
              if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained)
model.load_state_dict(model_dict)
```

---

## 4.2 优化器

### 4.2.1 SGD → Adam → AdamW

**SGD + Momentum**:

```
v_t = β * v_{t-1} + g_t              # 动量项
θ_t = θ_{t-1} - lr * v_t
```

**Adam**: 自适应学习率，结合了Momentum（一阶矩估计）和RMSProp（二阶矩估计）

```
m_t = β1 * m_{t-1} + (1-β1) * g_t    # 一阶矩（动量）
v_t = β2 * v_{t-1} + (1-β2) * g_t²   # 二阶矩（自适应学习率）
m̂_t = m_t / (1 - β1^t)               # 偏差修正
v̂_t = v_t / (1 - β2^t)               # 偏差修正
θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
```

**AdamW**: Adam + **解耦权重衰减** (Decoupled Weight Decay)

```
# Adam (L2 regularization): 把weight decay加到梯度里
g_t = g_t + λ * θ_{t-1}   # 然后再做Adam更新

# AdamW: 把weight decay从梯度中解耦出来
θ_t = (1 - lr * λ) * θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
# weight decay直接作用于参数，不经过Adam的自适应缩放
```

**面试关键**: 为什么Transformer偏好AdamW而不是Adam？

在Adam中，L2正则化的梯度也会被自适应学习率缩放，这导致大梯度参数的正则化被削弱，小梯度参数的正则化被过度放大。AdamW的解耦设计让weight decay对所有参数均匀作用。

### 4.2.2 Adam vs SGD 对比实验

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x.view(x.size(0), -1))
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x.view(x.size(0), -1))
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total

def compare_optimizers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=1024)

    # 定义简单的MLP
    def make_model():
        return nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)

    optimizers_config = {
        'SGD(lr=0.01)': lambda m: optim.SGD(m.parameters(), lr=0.01),
        'SGD+Momentum(0.9)': lambda m: optim.SGD(m.parameters(), lr=0.01, momentum=0.9),
        'Adam(lr=0.001)': lambda m: optim.Adam(m.parameters(), lr=0.001),
        'AdamW(lr=0.001)': lambda m: optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.01),
    }

    results = {}
    for name, opt_fn in optimizers_config.items():
        print(f"\n--- {name} ---")
        torch.manual_seed(42)
        model = make_model()
        optimizer = opt_fn(model)

        train_losses = []
        test_accs = []

        for epoch in range(5):
            loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            train_losses.append(loss)
            test_accs.append(test_acc)
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, "
                  f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        results[name] = {'losses': train_losses, 'accs': test_accs}

    return results

# 运行
# results = compare_optimizers()
```

---

## 4.3 AMP混合精度训练

### 4.3.1 原理

混合精度：大部分计算用FP16（速度快、省显存），关键部分保持FP32（精度高）。

```
FP32: 1位符号 + 8位指数 + 23位尾数 → 精度高，4字节
FP16: 1位符号 + 5位指数 + 10位尾数 → 精度低，2字节
BF16: 1位符号 + 8位指数 + 7位尾数  → 范围大精度低，2字节
```

FP16的问题：数值范围小 [6e-8, 65504]，梯度容易下溢。解决方案：**Loss Scaling**。

### 4.3.2 PyTorch AMP实现

```python
import torch
from torch.amp import autocast, GradScaler

model = make_model().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()  # 管理loss scaling

for x, y in train_loader:
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()

    # autocast: 自动选择FP16/FP32
    with autocast(device_type='cuda'):
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
    # 注意: loss本身是FP32的

    # GradScaler: 缩放loss → 防止FP16梯度下溢
    scaler.scale(loss).backward()
    scaler.step(optimizer)  # unscale梯度后执行optimizer.step
    scaler.update()         # 动态调整scale factor
```

**面试追问**:

Q: 为什么需要Loss Scaling？
A: FP16的最小正数约为6e-8，小梯度会变成0。把loss乘以一个大数（如1024），梯度也等比放大，计算后再缩回来。

Q: 哪些操作用FP16，哪些用FP32？
A: 矩阵乘法、卷积用FP16（计算密集，Tensor Core加速）。归一化、Softmax、loss计算用FP32（对数值敏感）。

Q: AMP能省多少显存？
A: 模型参数：前向计算用FP16，但优化器保持FP32 master copy。大约省30-50%激活内存，训练速度提升1.5-2x。

---

## 4.4 Dataset与DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import time

class CustomDataset(Dataset):
    def __init__(self, size=10000, input_dim=784):
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 这里可以加数据增强、读文件等I/O操作
        return self.data[idx], self.labels[idx]

dataset = CustomDataset()

# DataLoader参数详解
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,          # 训练时打乱
    num_workers=4,         # 多进程加载（CPU密集型预处理时有效）
    pin_memory=True,       # 锁页内存 → 加速CPU→GPU数据传输
    drop_last=True,        # 丢弃最后不完整的batch（BN需要）
    persistent_workers=True,  # 保持worker进程（减少每个epoch的启动开销）
    prefetch_factor=2,     # 每个worker预取2个batch
)

# Benchmark
def benchmark_loader(num_workers):
    loader = DataLoader(dataset, batch_size=64, num_workers=num_workers,
                       pin_memory=True)
    start = time.perf_counter()
    for x, y in loader:
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)  # 配合pin_memory
    elapsed = time.perf_counter() - start
    print(f"num_workers={num_workers}: {elapsed:.3f}s")

for nw in [0, 1, 2, 4]:
    benchmark_loader(nw)
```

**工程最佳实践**:
- `num_workers`: 一般设为CPU核数的1/4到1/2。太多会引起内存竞争。
- `pin_memory=True`: 几乎总是应该开启（GPU训练时）。
- `non_blocking=True`: 配合 `pin_memory`，实现CPU→GPU的异步传输。
- `persistent_workers=True`: 多epoch训练时必须开启，避免每个epoch重新fork worker进程。

---

## 4.5 Day4 产出物验收

- [ ] Hook可视化实验：捕获模型中间层激活，打印shape和统计量
- [ ] Adam vs SGD(+Momentum) vs AdamW 对比实验在MNIST上运行
- [ ] AMP训练实验运行（如有GPU），记录速度提升和显存节省
- [ ] 能口头解释：AdamW vs Adam的区别；AMP的Loss Scaling原因；DataLoader的关键参数

---

# Day 5 (周五): 【实战】从零手写micrograd引擎

## 学习目标

今天的目标是跟着Karpathy的micrograd，从零实现一个自动微分引擎。这不是"看懂就行"，而是**自己写一遍**。

完成后你应该能脱稿画出计算图、手推反向传播。

---

## 5.1 micrograd核心：Value类

```python
"""
micrograd: 一个极简的自动微分引擎
参考: https://github.com/karpathy/micrograd

核心思路:
1. Value包装标量值，追踪运算历史
2. 每个运算定义_backward()（局部梯度 × 上游梯度）
3. backward()按拓扑序反向调用所有_backward()
"""

import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # 梯度
        self._backward = lambda: None  # 反向传播函数
        self._prev = set(_children)  # 子节点
        self._op = _op  # 产生此节点的运算
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh²(x)
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """反向传播：拓扑排序后逆序调用_backward"""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  # dL/dL = 1
        for node in reversed(topo):
            node._backward()
```

---

## 5.2 构建神经网络

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        # w·x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        """nouts: 每一层的输出维度列表"""
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

---

## 5.3 训练验证

```python
def train_micrograd():
    """用micrograd训练一个简单的分类器"""
    random.seed(42)

    # 数据: 简单的二分类
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # 目标值

    # 模型: 3 → 4 → 4 → 1
    model = MLP(3, [4, 4, 1])
    print(f"Total parameters: {len(model.parameters())}")

    # 训练
    for epoch in range(100):
        # 前向传播
        ypred = [model(x) for x in xs]
        # MSE loss
        loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

        # 反向传播
        # 重要：先清零梯度
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # 更新参数
        lr = 0.05
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.data:.6f}")

    # 验证
    print("\nPredictions:")
    for x, yt in zip(xs, ys):
        yp = model(x)
        print(f"  input={x}, target={yt}, pred={yp.data:.4f}")

train_micrograd()
```

---

## 5.4 与PyTorch对比验证

```python
def verify_with_pytorch():
    """用同样的计算验证micrograd的正确性"""
    import torch

    # micrograd
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L.backward()

    print("micrograd:")
    print(f"  L = {L.data}, da = {a.grad}, db = {b.grad}, dc = {c.grad}, df = {f.grad}")

    # PyTorch
    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(-3.0, requires_grad=True)
    c_t = torch.tensor(10.0, requires_grad=True)
    f_t = torch.tensor(-2.0, requires_grad=True)
    e_t = a_t * b_t
    d_t = e_t + c_t
    L_t = d_t * f_t
    L_t.backward()

    print("PyTorch:")
    print(f"  L = {L_t.data}, da = {a_t.grad}, db = {b_t.grad}, "
          f"dc = {c_t.grad}, df = {f_t.grad}")

    # 验证一致
    assert abs(a.grad - a_t.grad.item()) < 1e-6
    assert abs(b.grad - b_t.grad.item()) < 1e-6
    print("\n✓ Results match!")

verify_with_pytorch()
```

---

## 5.5 Day5 产出物验收

- [ ] `Value` 类支持 +, *, **, -, /, tanh, relu, exp 运算及其反向传播
- [ ] `MLP` 能训练收敛（loss < 0.01）
- [ ] 与PyTorch对比验证通过
- [ ] 能画出一个简单表达式的计算图，标注前向值和梯度
- [ ] 代码推送到GitHub

---

# Day 6 (周六): CNN + ResNet + ViT

## 学习目标

面试验收标准：

1. 卷积的参数量和计算量怎么算？感受野怎么算？
2. ResNet为什么work？残差连接解决了什么问题？
3. ViT的核心组件：Patch Embedding、Position Embedding、CLS Token、Multi-Head Self-Attention
4. ResNet vs ViT 各自的优劣？什么场景选哪个？

---

## 6.1 卷积核心概念

### 6.1.1 参数量与计算量

对于一个卷积层 `Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=P)`:

```
参数量 = C_out × (C_in × K × K + 1)  # +1是bias
输出尺寸 = (H_in + 2P - K) / S + 1
计算量(FLOPs) = C_out × C_in × K × K × H_out × W_out × 2  # ×2因为乘加各算一次
```

```python
import torch.nn as nn

conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
params = sum(p.numel() for p in conv.parameters())
print(f"Parameters: {params}")
# 128 × (64 × 3 × 3) + 128 = 73,856

# 对于 32×32 输入
H_out = (32 + 2*1 - 3) // 1 + 1  # = 32
FLOPs = 128 * 64 * 3 * 3 * 32 * 32 * 2
print(f"FLOPs: {FLOPs:,}")  # ~150M
```

### 6.1.2 感受野

```
感受野 = 1 + Σ (kernel_size_i - 1) × Π stride_j  (j < i)

简化（所有层stride=1, padding=same）:
感受野 = 1 + Σ (kernel_size_i - 1)
```

3×3卷积堆叠的感受野：2层=5×5, 3层=7×7。这就是为什么VGGNet用多个3×3代替大卷积核——相同感受野但参数更少。

---

## 6.2 手写ResNet

### 6.2.1 残差连接的核心思想

没有残差连接时，训练深层网络会出现**退化问题**（不是过拟合！）：更深的网络训练误差反而更高。

残差连接让网络学习 F(x) = H(x) - x（残差），而不是直接学习 H(x)。如果恒等映射是最优解，网络只需要让F(x)=0，这比学习一个完整的恒等映射简单得多。

从梯度角度：残差连接提供了梯度的"高速公路"，∂L/∂x = ∂L/∂F × ∂F/∂x + ∂L/∂x，保证梯度至少有一个1的直通路径。

### 6.2.2 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet基本残差块 (ResNet-18/34)"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 对齐维度的1×1卷积

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """ResNet瓶颈残差块 (ResNet-50/101/152)
    1×1降维 → 3×3卷积 → 1×1升维
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        block: BasicBlock or Bottleneck
        layers: 每个stage的block数量, 如[3,4,6,3] for ResNet-50
        """
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 4个Stage
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化
        self._init_weights()

    def _make_layer(self, block, channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion,
                         1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# 测试
model = resnet50()
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f"Output shape: {out.shape}")  # (2, 10)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params / 1e6:.1f}M")  # ~23.5M
```

---

## 6.3 手写Vision Transformer (ViT)

### 6.3.1 ViT架构概览

```
Input Image (3×224×224)
    ↓ Patch Embedding: 分成16×16的patch → 线性映射
    ↓ [CLS] token + Position Embedding
    ↓ Transformer Encoder × L layers
    ↓ 取CLS token的输出
    ↓ MLP Head → 分类
```

### 6.3.2 实现

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """将图像分成patch并线性映射为embedding"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # 用卷积实现patch分割+线性映射（等价于reshape+Linear但更高效）
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)        # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # 一次性计算Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 各自: (B, heads, N, head_dim)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_dropout(self.proj(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm (ViT用Pre-norm, 原始Transformer用Post-norm)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=10, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token: 可学习的分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position Embedding: 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer Encoder
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+num_patches, embed_dim)

        # Add Position Embedding
        x = self.pos_dropout(x + self.pos_embed)

        # Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # 取CLS token的输出作为分类结果
        cls_output = x[:, 0]
        return self.head(cls_output)

# ViT-Tiny (适合CIFAR-10实验)
def vit_tiny(num_classes=10):
    return VisionTransformer(
        img_size=32,       # CIFAR-10是32×32
        patch_size=4,      # 4×4 patches → 8×8=64 patches
        embed_dim=192,
        depth=6,
        num_heads=3,
        num_classes=num_classes,
    )

# 测试
model = vit_tiny()
x = torch.randn(2, 3, 32, 32)
out = model(x)
print(f"Output shape: {out.shape}")  # (2, 10)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params / 1e6:.1f}M")
```

---

## 6.4 CIFAR-10训练对比

```python
"""
ResNet-18 vs ViT-Tiny on CIFAR-10
验收标准: 两个模型都能训练，观察收敛曲线差异
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def get_cifar10_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=transform_train)
    test_set = datasets.CIFAR10('./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=2)

    return train_loader, test_loader

def train_and_evaluate(model, train_loader, test_loader, epochs=20, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        test_acc = correct / total

        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"loss={avg_loss:.4f}, test_acc={test_acc:.4f}")

    return history

# 运行对比（如有GPU，实际跑一下；CPU上建议减少epoch数）
# train_loader, test_loader = get_cifar10_loaders()

# print("=== ResNet-18 ===")
# resnet = ResNet(BasicBlock, [2,2,2,2], num_classes=10)  # 简化版ResNet-18
# # 修改stem适配CIFAR-10的32×32输入
# resnet.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
# resnet.maxpool = nn.Identity()
# history_resnet = train_and_evaluate(resnet, train_loader, test_loader)

# print("\n=== ViT-Tiny ===")
# vit = vit_tiny()
# history_vit = train_and_evaluate(vit, train_loader, test_loader)
```

### ResNet vs ViT 对比总结

| 维度 | ResNet | ViT |
|------|--------|-----|
| 归纳偏置 | 强（局部性+平移等变性） | 弱（需要更多数据） |
| 小数据集 | 好（归纳偏置帮助） | 差（容易过拟合） |
| 大数据集 | 被ViT超越 | 更强（容量大） |
| 计算复杂度 | O(C²K²HW) | O(N²D)，N=patch数 |
| 位置信息 | 隐式（卷积核位置） | 显式（Position Embedding） |
| 工程部署 | 优化成熟（TensorRT） | 逐渐追上 |
| 你的合成数据项目 | 适合做baseline | 适合做high-capacity模型 |

---

## 6.5 Day6 产出物验收

- [ ] ResNet-50 实现能正确前向传播，参数量~23.5M
- [ ] ViT实现能正确前向传播
- [ ] CIFAR-10训练脚本能运行（至少验证前向+反向可以执行）
- [ ] 能口头解释：残差连接为什么work；ViT的Patch Embedding是怎么做的；为什么ViT在小数据上不如CNN

---

# Day 7 (周日): 复盘 + ConSE工程深挖

## 学习目标

1. 知识巩固：能对W1每个主题做2分钟脱稿讲解
2. 工程深挖：用工程视角重新审视你的ConSE博士项目

---

## 7.1 知识脱稿讲解（每个2分钟）

### 准备清单

用你自己的话准备以下6个主题的2分钟讲解（录音自评）：

**1. Python GC机制**

关键点：引用计数（主要机制，即时回收）→ 循环引用问题 → 分代GC（标记-清除，三代策略）→ `weakref` 的应用。

**2. GIL**

关键点：GIL保护的是CPython内部数据结构 → CPU密集用多进程，I/O密集用asyncio → GIL在I/O和C扩展时释放 → PyTorch的数据加载器用多进程绕过GIL。

**3. Autograd计算图**

关键点：动态图（define-by-run）→ 前向时构建DAG → `backward()` 拓扑排序逆序遍历 → 链式法则累加梯度 → `zero_grad()` 的必要性。

**4. AMP混合精度**

关键点：FP16省内存+Tensor Core加速 → 梯度下溢问题 → Loss Scaling → autocast自动选择精度 → GradScaler动态调整scale。

**5. ResNet残差连接**

关键点：深层网络退化问题 → 学习残差F(x)=H(x)-x更容易 → 梯度直通路径 → Bottleneck结构（1×1降维→3×3卷积→1×1升维）。

**6. ViT核心组件**

关键点：Patch Embedding（Conv2d实现）→ CLS Token → Positional Embedding → MHSA → Pre-LN → 弱归纳偏置需要大数据。

---

## 7.2 ConSE项目工程深挖

以面试官的视角重新审视你的ConSE项目。按以下框架整理：

### 7.2.1 系统架构复盘

回答以下问题并写下来：

```
1. ConSE的整体系统架构是什么？画一张数据流图。
   输入 → 处理步骤 → 输出

2. 数据流是怎样的？
   - 原始数据来源（施工现场图片从哪来？什么格式？多大量？）
   - 本体如何构建的？
   - 图像标注怎么做的？
   - 训练数据pipeline

3. 技术选型的 why：
   - 为什么选 OWL 作为本体语言？和 RDF/RDFS 对比？
   - 为什么选 ResNet/ViT/CLIP 这几个backbone？
   - 为什么选 ControlNet 生成合成数据？

4. 如果这个项目不是学术研究而是工程产品，你会怎么改？
```

### 7.2.2 性能瓶颈分析

```
1. 整个pipeline中，最慢的环节是什么？如何优化？
2. 合成数据生成的效率瓶颈在哪里？
3. 如果需要处理10万张图片（而不是几千张），哪些环节会崩？
```

### 7.2.3 "如果重来"问题

这是面试中最常问的问题之一：

```
1. 如果你重新做这个项目，会有哪些不同的决策？
2. 最大的技术debt是什么？
3. 你遇到的最困难的bug/问题是什么？怎么解决的？
4. 如果给你团队（2个工程师），你怎么拆分工作？
```

### 7.2.4 扩展性思考

```
1. ConSE本体能扩展到其他领域（如医疗、制造）吗？需要改什么？
2. 如果要把这个系统部署为在线服务，需要加什么组件？
   （API? 数据库? 队列? 监控?）
3. 合成数据方法如果要generalize到其他垂直领域，核心挑战是什么？
```

---

## 7.3 Day7 产出物格式

创建一个 `week1_review.md`，包含：

```markdown
# W1 复盘笔记

## 知识点自测
| 主题 | 自评(1-5) | 薄弱点 | 补强计划 |
|------|----------|--------|---------|
| GC/GIL | | | |
| Autograd | | | |
| AMP | | | |
| ResNet | | | |
| ViT | | | |
| micrograd | | | |

## ConSE 工程复盘
### 系统架构
(画图 + 文字描述)

### 性能瓶颈
...

### 如果重来
...

### 扩展性
...

## 面试模拟QA
Q1: Python的GC机制？
A1: (你的回答)

Q2: 为什么用合成数据？domain gap怎么处理？
A2: (你的回答)

(... 至少10个QA)
```

---

# 附录A: W1面试高频题速查

## Python部分

| # | 问题 | 关键词 |
|---|------|--------|
| 1 | Python的GC机制 | 引用计数 + 分代GC + 循环引用 |
| 2 | GIL是什么，怎么绕过 | 多进程 / asyncio / C扩展释放GIL |
| 3 | 深拷贝vs浅拷贝 | copy.copy vs copy.deepcopy, 可变对象 |
| 4 | 装饰器的原理 | 高阶函数 + 闭包 + functools.wraps |
| 5 | `__slots__` 作用 | 省内存 + 限制动态属性 |
| 6 | 元类的作用 | 控制类的创建 + 自动注册 |
| 7 | `is` vs `==` | id比较 vs 值比较 + 小整数缓存 |
| 8 | 生成器vs列表 | 惰性求值 + 内存效率 + yield |

## PyTorch部分

| # | 问题 | 关键词 |
|---|------|--------|
| 1 | Tensor的内存模型 | Storage + Stride + View + contiguous |
| 2 | Autograd工作原理 | 动态计算图 + 拓扑排序 + 链式法则 |
| 3 | `no_grad` vs `detach` | 上下文管理器 vs 单tensor分离 |
| 4 | 梯度累加问题 | zero_grad的必要性 |
| 5 | Adam vs AdamW | 解耦权重衰减 |
| 6 | AMP原理 | FP16 + Loss Scaling + autocast |
| 7 | DataLoader优化 | num_workers + pin_memory + persistent_workers |
| 8 | hook的用途 | 特征可视化 + 梯度检查 |

## 模型部分

| # | 问题 | 关键词 |
|---|------|--------|
| 1 | ResNet为什么work | 退化问题 + 残差学习 + 梯度直通 |
| 2 | ViT核心组件 | Patch Embed + CLS + PosEmbed + MHSA |
| 3 | ResNet vs ViT | 归纳偏置 + 数据需求 + 部署成熟度 |
| 4 | 卷积参数量计算 | Cout × (Cin × K² + 1) |
| 5 | 感受野计算 | 逐层累加 (K-1)×累积stride |
| 6 | BN vs LN | BN跨样本统计 / LN跨特征统计 |

---

# 附录B: W1 算法刷题指引

本周并行刷题主题：**数组 + 哈希表 + 双指针**（每天30-45min，共约10题）

## 推荐题目

### 数组基础
1. **LC 1 Two Sum** — 哈希表一遍扫描 O(n)
2. **LC 26 Remove Duplicates** — 双指针原地操作
3. **LC 283 Move Zeroes** — 双指针保持相对顺序
4. **LC 88 Merge Sorted Array** — 从后往前双指针

### 哈希表
5. **LC 49 Group Anagrams** — sorted(str)作为key
6. **LC 128 Longest Consecutive Sequence** — HashSet + 只从序列起点开始扫
7. **LC 242 Valid Anagram** — 计数数组/Counter

### 双指针
8. **LC 11 Container With Most Water** — 左右指针 + 贪心
9. **LC 15 3Sum** — 排序 + 固定一个 + 双指针 + 去重
10. **LC 167 Two Sum II** — 有序数组双指针

## 刷题方法论

1. **先想15分钟**，想不出来看题解
2. **理解后合上题解自己写**
3. **分析时间/空间复杂度**
4. **第二天复习昨天的题**（间隔重复）
5. **总结模式**：哈希表适合"查找配对"，双指针适合"有序数组/相向搜索"

---

# 附录C: 本周学习资源

## 必读
- Karpathy micrograd: https://github.com/karpathy/micrograd (Day5核心)
- PyTorch Autograd 官方教程: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- ResNet 原论文: He et al., "Deep Residual Learning" (重点读Section 3)
- ViT 原论文: Dosovitskiy et al., "An Image is Worth 16×16 Words" (重点读Section 3)

## 推荐
- Real Python: Python Memory Management
- PyTorch Internals: ezyang.github.io/stride-visualizer
- The Illustrated Transformer (Jay Alammar): jalammar.github.io

---

*W1 教材完 — 下一步: 完成每天的产出物，周日复盘时用自评表诚实评分。*
