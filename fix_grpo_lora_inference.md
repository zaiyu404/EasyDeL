# GRPO中LoRA推理问题修复方案

## 问题根源

在GRPO训练器中使用LoRA后，生成只有第一个token正确的问题源于：

### 1. 核心问题位置

**文件**: `easydel/trainers/group_relative_policy_optimization/grpo_trainer.py:338-367`

```python
@ejit(
    in_shardings=(self.state_shardings, empty_sharding, empty_sharding),
    out_shardings=(empty_sharding, empty_sharding, empty_sharding),
)
def generate(state: EasyDeLState, input_ids, attention_mask):
    module = state.model  # ❌ 问题在这里！

    with module.mesh:
        ...
        sequences = module.generate(...)
```

### 2. 问题原因

`state.model` 是一个property（`easydel/infra/base_state.py:386-392`）：

```python
@property
def model(self) -> EasyDeLBaseModule:
    return nn.merge(self.graphdef, self.graphstate, self.graphother)
```

**问题**：
- 在JIT编译的函数内部，每次访问`state.model`都会调用`nn.merge()`重新构建模块
- 当JIT函数被追踪时，这个merge操作可能不会正确地处理LoRA层的内部状态
- 在自回归生成循环中（第2个token开始），模块可能使用了不完整的LoRA状态

### 3. 为什么第一个token正确？

- **Prefill阶段**（第一个token）：处理完整的prompt序列，LoRA参数正确应用
- **Decode阶段**（后续tokens）：使用KV cache，每次只处理一个新token，此时通过JIT追踪的merge可能丢失LoRA状态

## 修复方案

### 方案1：在JIT函数外部merge模块（推荐）

修改 `grpo_trainer.py` 的 `configure_functions` 方法：

```python
def configure_functions(self) -> TrainerConfigureFunctionOutput:
    mesh = self.model.mesh
    empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

    # ✅ 在JIT函数外部merge模块
    @ejit(
        in_shardings=(
            self.state_shardings.graphdef,      # graphdef
            self.state_shardings.graphstate,    # graphstate
            self.state_shardings.graphother,    # graphother
            empty_sharding,                      # input_ids
            empty_sharding,                      # attention_mask
        ),
        out_shardings=(empty_sharding, empty_sharding, empty_sharding),
    )
    def generate(graphdef, graphstate, graphother, input_ids, attention_mask):
        # 在JIT函数内部merge
        module = flax.nnx.merge(graphdef, graphstate, graphother)

        with module.mesh:
            input_ids = module.config.partition_manager.shard(
                input_ids,
                axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                mode=common_types.MODE_PREFILL,
            )
            attention_mask = module.config.partition_manager.shard(
                attention_mask,
                axes=[common_types.BATCH, common_types.SEQUENCE_PARALLEL],
                mode=common_types.MODE_PREFILL,
            )
            sequences = module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    top_p=self.arguments.top_p,
                    top_k=self.arguments.top_k,
                    temperature=self.arguments.temperature,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    max_new_tokens=self.arguments.max_completion_length,
                    max_length=self.arguments.max_completion_length + self.arguments.max_prompt_length,
                    num_return_sequences=self.num_generations,
                    do_sample=True,
                ),
            ).sequences
            return sequences, input_ids, attention_mask

    self.generate_function = generate
    # ...
```

然后修改调用处（`_preprocess_batch_input` 方法）：

```python
def _preprocess_batch_input(self, state: EasyDeLState, batch, is_train):
    with capture_time() as preprocessing_time_fn:
        prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

        with capture_time() as generation_time_fn:
            sequences, prompt_ids, prompt_mask = jax.block_until_ready(
                self.generate_function(
                    state.graphdef,     # ✅ 显式传递图组件
                    state.graphstate,
                    state.graphother,
                    prompt_ids,
                    prompt_mask
                )
            )
        # ...
```

### 方案2：检查LoRA层状态是否在graphother中

验证LoRA层的状态是否正确包含：

```python
# 临时调试代码
def debug_lora_state(state: EasyDeLState):
    model = state.model

    from easydel.utils.traversals import iter_module_search
    import flax.nnx as nn

    print("=== Checking LoRA layers ===")
    for path, mod in iter_module_search(model, nn.LoRA):
        print(f"LoRA at: {'.'.join(map(str, path))}")
        # 检查LoRA参数
        graphdef, graphstate, graphother = nn.split(mod, nn.Param, ...)
        print(f"  graphstate keys: {graphstate.keys() if hasattr(graphstate, 'keys') else 'N/A'}")
        print(f"  graphother keys: {graphother.keys() if hasattr(graphother, 'keys') else 'N/A'}")
```

### 方案3：如果问题仍存在，禁用generation时的JIT编译

在生成配置中添加trace=False：

```python
# 在generate函数中
sequences = module.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    generation_config=GenerationConfig(...),
    trace=False,  # ✅ 禁用JIT追踪
).sequences
```

注意：这会显著降低性能，仅用于调试。

## 测试修复

创建测试脚本：

```python
import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer

# 1. 加载模型并应用LoRA
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    auto_shard_model=True,
)

print("Applying LoRA...")
model = model.apply_lora_to_layers(
    rank=32,
    lora_pattern=".*(q_proj|k_proj|v_proj|o_proj).*"
)

# 2. 创建state
state = model.to_state()

# 3. 准备输入
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
test_prompt = "Hello, how are"
inputs = tokenizer(test_prompt, return_tensors="jax")

# 4. 测试生成
print("\n=== Testing generation ===")
output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=10,
)

print("Generated:", tokenizer.decode(output.sequences[0]))

# 5. 测试通过state生成
print("\n=== Testing through state ===")
module = state.model
output2 = module.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=10,
)

print("Generated:", tokenizer.decode(output2.sequences[0]))
```

## 预期结果

修复后，两种生成方式应该产生相同的、语义连贯的输出，而不是只有第一个token正确。
