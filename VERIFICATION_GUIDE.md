# GRPO LoRA修复 - 验证指南

## 目标

验证GRPO中使用LoRA时，推理能够正确生成多个连贯的tokens，而不是只有第一个token正确，后续全是0。

## 前置条件

- EasyDeL已更新到包含修复的版本
- 分支：`claude/fix-lora-inference-tokens-011CUtb4T6au41mCpzWcw3Sb`
- 包含两个关键修复：
  1. 移除generate函数的@ejit装饰器
  2. 从eos_token_id中移除pad_token_id

## 验证步骤

### 第一步：基础功能验证

#### 1.1 检查tokenizer配置

```bash
python debug_token_zero.py
```

**期望输出**：
```
Tokenizer Configuration Analysis
================================
Model: Qwen/Qwen2.5-0.5B
pad_token_id: 151643
eos_token_id: 151643
bos_token_id: 151643

⚠️  WARNING: Removing pad_token_id (151643) from eos_token_id list
   to prevent premature generation termination.
```

**验证点**：
- ✅ 如果看到warning，说明次修复生效
- ✅ 确认pad_token_id不在最终的eos列表中

#### 1.2 运行快速测试

```bash
python test_grpo_lora_fix.py
```

**期望输出**：
```
Testing GRPO LoRA Inference Fix
================================
1. Loading model: Qwen/Qwen2.5-0.5B
   ✓ Model loaded successfully

2. Applying LoRA to model
   ✓ LoRA applied successfully

3. Creating EasyDeLState
   ✓ State created successfully

4. Preparing test input
   ✓ Test prompt: 'Hello, how are you today? I am'

5. Testing direct model generation (baseline)
   Generated: 'Hello, how are you today? I am doing great...'
   ✓ Direct generation successful

6. Testing generation through state.model property
   Generated: 'Hello, how are you today? I am doing great...'
   ✓ State.model generation successful

7. Testing generation with explicit graph components (fixed method)
   Generated: 'Hello, how are you today? I am doing great...'
   ✓ Fixed method generation successful

8. Verifying output consistency
   ✓ All generation methods produce identical outputs

   Input length: 10 tokens
   Output length: 20 tokens
   Generated: 10 new tokens
   ✓ Multi-token generation successful

9. Checking output coherence
   Generated text: 'doing great...'
   ✓ Generated text appears coherent

================================
✓ TEST PASSED: GRPO LoRA inference is working correctly
================================
```

**验证点**：
- ✅ 生成多个token（不止1个）
- ✅ tokens不是全0
- ✅ 生成的文本连贯
- ✅ 三种方法输出一致

### 第二步：GRPO训练验证

#### 2.1 创建最小测试脚本

```python
# minimal_grpo_lora_test.py
import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import Dataset

# 1. 配置
config = ed.GRPOConfig(
    save_directory="test_grpo_lora",
    num_train_epochs=1,
    total_batch_size=2,
    max_prompt_length=128,
    max_completion_length=64,
    num_return_sequences=2,
    learning_rate=1e-6,
    top_k=10,
    top_p=0.95,
    temperature=0.7,
)

# 2. 加载模型
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    auto_shard_model=True,
    sharding_axis_dims=(1, -1, 1, 1, 1),
)

# 3. 应用LoRA
print("Applying LoRA...")
model = model.apply_lora_to_layers(
    lora_rank=16,
    lora_pattern=".*(q_proj|k_proj|v_proj|o_proj).*",
)
print(f"LoRA applied to model")

# 4. 准备数据
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = Dataset.from_dict({
    "question": [
        "What is 2+2?",
        "What is the capital of France?",
    ]
})

def tokenize_fn(batch, tokenizer, tools):
    prompts = []
    for q in batch["question"]:
        prompts.append([
            {"role": "user", "content": q}
        ])
    return tokenizer(prompts, return_tensors="np", padding="max_length", max_length=128, truncation=True)

# 5. 创建reward函数
def simple_reward(prompts, completions, **kwargs):
    # 简单reward：生成长度 > 10就给奖励
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        reward = 1.0 if len(text) > 10 else 0.0
        rewards.append(reward)
    return rewards

# 6. 创建trainer
trainer = ed.GRPOTrainer(
    model=model,
    reward_funcs=[simple_reward],
    processing_class=tokenizer,
    train_dataset=dataset,
    arguments=config,
    data_tokenize_fn=tokenize_fn,
)

# 7. 测试生成（不训练，只测试生成）
print("\nTesting generation before training...")
state = trainer.model_state

# 准备测试batch
test_batch = tokenizer(
    [{"role": "user", "content": "Hello"}],
    return_tensors="jax",
    padding="max_length",
    max_length=128,
)

# 调用generate
sequences, _, _ = trainer.generate_function(
    state,
    test_batch["input_ids"],
    test_batch["attention_mask"],
)

# 解码
generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

# 验证
tokens = tokenizer.encode(generated_text)
print(f"Token IDs: {tokens[:20]}")  # 显示前20个token

# 检查是否全是0
all_zero = all(t == 0 for t in tokens[10:20])  # 检查10-20位置
if all_zero:
    print("❌ FAILED: Generated tokens are all 0")
else:
    print("✅ PASSED: Generated tokens are diverse")

# 检查是否有多个token
if len(tokens) > 15:
    print(f"✅ PASSED: Generated {len(tokens)} tokens")
else:
    print(f"❌ FAILED: Only generated {len(tokens)} tokens")
```

**运行**：
```bash
python minimal_grpo_lora_test.py
```

**期望输出**：
```
Applying LoRA...
LoRA applied to model

Testing generation before training...
Generated text: Hello! How can I help you today? I'm here to assist...
Token IDs: [9906, 0, 4340, 646, 358, 1492, 498, 3351, 30, 358, 2776, 1688, 311, 7789, 1917, 438, 4340, 646, 358, 1568]
✅ PASSED: Generated tokens are diverse
✅ PASSED: Generated 35 tokens
```

**验证点**：
- ✅ 生成的token IDs多样化（不是全0）
- ✅ 生成了足够多的tokens（>15）
- ✅ 解码后的文本连贯

#### 2.2 检查训练日志

如果运行完整训练：
```python
trainer.train()
```

在训练日志中查找：
```
⚠️  WARNING: Removing pad_token_id (151643) from eos_token_id list
   to prevent premature generation termination.
```

**验证点**：
- ✅ 看到warning说明配置检查生效
- ✅ 训练过程中generation不报错
- ✅ 生成的completions不是全0

### 第三步：深度调试（可选）

如果需要更详细的调试信息：

#### 3.1 添加调试输出

编辑 `easydel/infra/mixins/generation.py`，在 `sample_search_body_fn` 中：

```python
def sample_search_body_fn(state):
    """state update fn."""
    prng_key, prng_key_next = jax.random.split(state.prng_key)

    call_kwargs = {k: v for k, v in state.model_kwargs.items() if k != "attention_mask"}
    model_outputs = model(state.running_token, **call_kwargs)

    logits = model_outputs.logits[:, -1]

    # === 添加调试输出 ===
    if state.cur_len < 5:  # 只输出前5步
        import jax.numpy as jnp
        print(f"\n=== Generation Step {state.cur_len} ===")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits range: [{float(logits.min()):.2f}, {float(logits.max()):.2f}]")
        print(f"Logits mean: {float(logits.mean()):.2f}")
        print(f"Logits std: {float(logits.std()):.2f}")
        print(f"Has NaN: {bool(jnp.isnan(logits).any())}")
        print(f"Has Inf: {bool(jnp.isinf(logits).any())}")
    # === 调试输出结束 ===

    logits = logits_processor(state.sequences, logits, state.cur_len)
    logits = logits_warper(state.sequences, logits, state.cur_len)

    next_token = (
        jax.random.categorical(prng_key, logits, axis=-1) * ~state.is_sent_finished
        + pad_token_id * state.is_sent_finished
    )

    # === 添加调试输出 ===
    if state.cur_len < 5:
        print(f"Sampled token: {int(next_token[0])}")
        print(f"is_sent_finished: {bool(state.is_sent_finished[0])}")
        if eos_token_id is not None:
            eos_arr = jnp.atleast_1d(jnp.array(eos_token_id, jnp.int32))
            print(f"Token in EOS: {bool(jnp.isin(next_token[0], eos_arr))}")
    # === 调试输出结束 ===

    ...
```

#### 3.2 运行并查看输出

```bash
python minimal_grpo_lora_test.py 2>&1 | tee debug_output.log
```

**期望输出**（修复后）：
```
=== Generation Step 0 ===
Logits shape: (50000,)
Logits range: [-12.45, 10.23]
Logits mean: -1.23
Logits std: 4.56
Has NaN: False
Has Inf: False
Sampled token: 9906
is_sent_finished: False
Token in EOS: False

=== Generation Step 1 ===
Logits shape: (50000,)
Logits range: [-11.89, 9.87]
Logits mean: -1.15
Logits std: 4.42
Has NaN: False
Has Inf: False
Sampled token: 4340
is_sent_finished: False
Token in EOS: False

... (后续步骤类似，logits保持正常)
```

**异常输出**（如果修复失效）：
```
=== Generation Step 0 ===
Logits range: [-12.45, 10.23]
Sampled token: 9906
is_sent_finished: False

=== Generation Step 1 ===
Logits range: [-inf, inf]  ← 异常！
Has NaN: True              ← 异常！
Sampled token: 0           ← 返回0！
is_sent_finished: False
Token in EOS: True         ← 触发终止！

=== Generation Step 2 ===
Sampled token: 0           ← 后续全是0
is_sent_finished: True     ← 已终止
```

### 第四步：性能验证

#### 4.1 基准测试

```python
# benchmark_grpo_lora.py
import time
import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型 + LoRA
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    auto_shard_model=True,
)
model = model.apply_lora_to_layers(lora_rank=16, lora_pattern=".*(q_proj).*")

# 准备输入
inputs = tokenizer("Hello, how are you", return_tensors="jax")

# 预热
for _ in range(3):
    _ = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

# 基准测试
num_runs = 10
start = time.time()
for _ in range(num_runs):
    output = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=50,
    )
end = time.time()

avg_time = (end - start) / num_runs
tokens_per_sec = 50 / avg_time

print(f"Average generation time: {avg_time:.3f}s")
print(f"Tokens per second: {tokens_per_sec:.1f}")
print(f"Generated sample: {tokenizer.decode(output.sequences[0])}")
```

**期望结果**：
- 性能与修复前相近（因为核心while_loop仍然JIT）
- 生成速度应该在合理范围内（取决于硬件）

### 第五步：回归测试

确保修复没有破坏其他功能：

#### 5.1 测试不使用LoRA的GRPO
```python
# 使用相同配置，但不apply_lora_to_layers
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(...)
# 不调用 apply_lora_to_layers

trainer = ed.GRPOTrainer(model=model, ...)
trainer.train()
```

**验证点**：
- ✅ 正常训练，无报错
- ✅ 生成质量正常

#### 5.2 测试其他trainer
```python
# 测试SFT trainer with LoRA
sft_config = ed.SFTConfig(...)
sft_trainer = ed.SFTTrainer(model=model_with_lora, ...)
sft_trainer.train()
```

**验证点**：
- ✅ 其他trainer不受影响
- ✅ LoRA在SFT中正常工作

## 验证清单

使用此清单确保所有方面都已验证：

- [ ] tokenizer配置检查通过（debug_token_zero.py）
- [ ] 基础测试通过（test_grpo_lora_fix.py）
- [ ] GRPO生成测试通过（minimal_grpo_lora_test.py）
- [ ] 生成的tokens多样化（不是全0）
- [ ] 生成了足够数量的tokens（>15）
- [ ] 生成的文本连贯
- [ ] 看到pad_token移除的warning
- [ ] logits在所有步骤都正常（无NaN/Inf）
- [ ] 不会提前触发is_sent_finished
- [ ] 性能在合理范围内
- [ ] 不使用LoRA的GRPO仍然正常
- [ ] 其他trainer不受影响

## 问题排查

### 问题1：仍然生成全0

**检查**：
```bash
git log --oneline -5
```

确认包含这两个commit：
- `fix(grpo): remove JIT from generate to fix LoRA multi-token generation`
- `fix(grpo): prevent pad_token in eos list causing all-zero generation`

**解决**：
```bash
git pull origin claude/fix-lora-inference-tokens-011CUtb4T6au41mCpzWcw3Sb
```

### 问题2：性能显著下降

**原因**：可能是其他瓶颈

**检查**：
```python
# 添加profiling
import jax
jax.profiler.start_trace("/tmp/tensorboard")
# ... 运行代码 ...
jax.profiler.stop_trace()
```

查看哪里慢了。

### 问题3：生成质量差

**原因**：LoRA可能未训练或rank太小

**检查**：
```python
# 检查LoRA参数是否有梯度
state = trainer.model_state
for path, param in jax.tree_util.tree_leaves_with_path(state.graphstate):
    if 'lora' in str(path):
        print(f"{path}: mean={param.mean()}, std={param.std()}")
```

## 成功标准

验证通过的标准：

1. **功能性**：
   - ✅ 多token生成
   - ✅ tokens多样化
   - ✅ 文本连贯

2. **正确性**：
   - ✅ 无NaN/Inf
   - ✅ 无提前终止
   - ✅ LoRA参数生效

3. **性能**：
   - ✅ 与修复前相近
   - ✅ 核心循环高效

4. **兼容性**：
   - ✅ 不破坏其他功能
   - ✅ 配置检查生效

---

**验证完成**后，可以在实际项目中使用GRPO + LoRA了！
