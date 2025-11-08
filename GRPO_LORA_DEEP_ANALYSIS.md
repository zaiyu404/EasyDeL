# GRPO LoRA推理问题深度分析

## 问题现象
即使关闭cache，使用LoRA后：
- 第一个token生成正确 ✅
- 后续所有token都是0 ❌

## 根本原因分析

### 问题1：JIT环境中的模块闭包

在 `generation.py:1380-1410`，生成循环的关键代码：

```python
model = self.decode if self.config.is_encoder_decoder else self  # 闭包变量

def sample_search_body_fn(state):
    ...
    model_outputs = model(state.running_token, **call_kwargs)  # 使用闭包中的model
```

在GRPO的generate函数中：
```python
@ejit(...)
def generate(graphdef, graphstate, graphother, input_ids, attention_mask):
    module = flax.nnx.merge(graphdef, graphstate, graphother)
    sequences = module.generate(...)  # 这里调用_sample
```

**核心问题**：
1. `module.generate()` 内部调用 `_sample()`
2. `_sample()` 创建了一个 `while_loop`
3. 在 `while_loop` 的body函数中，使用闭包捕获的 `model` 对象（即 `self`）
4. 当这个 `module` 是在JIT函数内部通过 `merge()` 创建时，JAX的追踪器**无法正确追踪LoRA层的完整状态**

### 问题2：Flax NNX模块在while_loop中的状态管理

Flax NNX的模块包含三部分：
- `graphdef`: 模块结构定义（静态）
- `graphstate`: 可训练参数（动态）
- `graphother`: 其他状态如dropout、buffer等（动态）

当模块作为**闭包变量**在 `while_loop` 中使用时：
- JAX需要将模块转换为pytree
- 但由于模块是在JIT函数内部创建的，这个转换可能**不完整**
- 特别是LoRA层（`flax.nnx.LoRA`是一个包装模块），它的状态可能无法正确追踪

### 问题3：LoRA层的forward在decode阶段失效

LoRA的forward逻辑：
```python
output = base_module(x) + lora_adapter(x) * scaling
```

在while_loop的decode阶段：
- 如果LoRA的state没有正确传递
- `lora_adapter(x)` 可能返回0或未初始化的值
- 导致输出只有base_module的结果，而base_module可能没有被训练

## 为什么第一个token正确？

**Prefill阶段**（生成第一个token）：
- 处理完整的prompt序列
- 此时是直接forward调用，不在while_loop内
- LoRA参数完整应用 ✅

**Decode阶段**（生成后续tokens）：
- 在while_loop中，每次处理一个新token
- 模块对象通过闭包传递，状态不完整
- LoRA层失效，导致输出异常 ❌

## 为什么生成的token是0？

可能的原因：
1. logits变成全0或极小值，categorical采样倾向于返回0索引
2. 如果pad_token_id是0，采样结果被错误地识别为padding
3. LoRA层失效后，base model的输出可能没有被正确训练，导致logits分布异常

## 正确的修复方案

### 方案1：避免模块作为闭包变量（推荐）

修改GRPO trainer，不在JIT函数内调用 `module.generate()`，而是**不使用JIT**：

```python
def configure_functions(self):
    mesh = self.model.mesh

    # 不使用@ejit装饰generate函数
    def generate(state: EasyDeLState, input_ids, attention_mask):
        module = state.model

        with module.mesh:
            # shard inputs
            input_ids = module.config.partition_manager.shard(...)
            attention_mask = module.config.partition_manager.shard(...)

            sequences = module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(...),
            ).sequences
            return sequences, input_ids, attention_mask

    self.generate_function = generate
    ...
```

**优点**：
- 避免嵌套JIT导致的状态追踪问题
- generate内部的while_loop仍然会被JIT编译（通过trace=True）
- 简单直接

**缺点**：
- generate函数本身不被JIT，可能有轻微性能损失（但内部while_loop仍然是JIT的）

### 方案2：将graphstate传入while_loop状态

修改 `_sample` 方法，将模块状态作为while_loop状态的一部分：

```python
# 在SampleState中添加graphstate
state = SampleState(
    cur_len=cur_len,
    sequences=sequences,
    running_token=input_ids,
    is_sent_finished=is_sent_finished,
    model_kwargs=...,
    prng_key=prng_key,
    graphstate=self.graphstate,  # 新增
    graphother=self.graphother,  # 新增
)

def sample_search_body_fn(state):
    # 在body内部merge模块
    module = flax.nnx.merge(self.graphdef, state.graphstate, state.graphother)
    model_outputs = module(state.running_token, **call_kwargs)
    ...
```

**优点**：
- 彻底解决状态追踪问题
- 确保LoRA状态在每次迭代中正确传递

**缺点**：
- 需要修改generation.py的核心代码
- 影响范围较大
- 每次迭代都需要merge，可能有性能开销

### 方案3：使用trace=False禁用JIT（调试用）

临时解决方案，仅用于验证问题：

```python
sequences = module.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    generation_config=GenerationConfig(...),
    trace=False,  # 禁用while_loop的JIT
).sequences
```

这会显著降低性能，但可以验证问题是否确实在JIT追踪上。

## 推荐的修复步骤

1. **立即实施方案1**：移除generate函数的@ejit装饰器
2. **测试验证**：确认问题解决
3. **性能评估**：测试性能影响
4. **长期方案**：如果性能影响显著，考虑实施方案2

## 相关代码位置

- `easydel/trainers/group_relative_policy_optimization/grpo_trainer.py:334-374`
- `easydel/infra/mixins/generation.py:1380-1460`
- `easydel/infra/base_state.py:386-392`
