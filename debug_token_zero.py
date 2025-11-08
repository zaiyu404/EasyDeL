#!/usr/bin/env python3
"""
Debug script to understand why tokens become 0 in GRPO LoRA generation.

This script helps identify the exact mechanism:
1. LoRA state loss → logits corruption
2. Token sampling returns 0
3. 0 triggers is_sent_finished
4. All subsequent tokens become pad_token_id (0)
"""

import easydel as ed
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer


def analyze_tokenizer_config(model_id: str):
    """Check if tokenizer has overlapping pad/eos token IDs."""
    print("=" * 60)
    print("Tokenizer Configuration Analysis")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"\nModel: {model_id}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"bos_token_id: {tokenizer.bos_token_id}")

    # Check for dangerous overlaps
    eos_list = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]

    if tokenizer.pad_token_id in eos_list:
        print(f"\n⚠️  WARNING: pad_token_id ({tokenizer.pad_token_id}) is in eos_token_id list!")
        print("   This can cause premature termination!")

    if 0 in eos_list:
        print(f"\n⚠️  WARNING: 0 is in eos_token_id list!")
        print("   If sampling returns 0, generation will stop immediately!")

    if tokenizer.pad_token_id == 0:
        print(f"\n⚠️  WARNING: pad_token_id is 0!")
        print("   All padding will be 0, making debug harder!")

    return tokenizer


def test_base_model_logits(model_id: str):
    """Test what happens to logits when LoRA is applied vs not applied."""
    print("\n" + "=" * 60)
    print("Base Model Logits Test")
    print("=" * 60)

    # Load model without LoRA
    print("\n1. Loading base model (no LoRA)...")
    model_base = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
    )

    # Prepare test input
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    test_input = "Hello"
    inputs = tokenizer(test_input, return_tensors="jax")

    # Get base logits
    print("\n2. Testing base model forward pass...")
    outputs_base = model_base(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    logits_base = outputs_base.logits[0, -1, :]  # Last token logits

    print(f"   Base logits shape: {logits_base.shape}")
    print(f"   Base logits range: [{logits_base.min():.4f}, {logits_base.max():.4f}]")
    print(f"   Base logits mean: {logits_base.mean():.4f}")
    print(f"   Base logits std: {logits_base.std():.4f}")
    print(f"   Contains NaN: {jnp.isnan(logits_base).any()}")
    print(f"   Contains Inf: {jnp.isinf(logits_base).any()}")

    # Get argmax
    base_top_token = jnp.argmax(logits_base)
    print(f"   Argmax token ID: {base_top_token}")

    # Now apply LoRA
    print("\n3. Applying LoRA...")
    model_lora = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
    )
    model_lora = model_lora.apply_lora_to_layers(
        lora_rank=32,
        lora_pattern=".*(q_proj|k_proj|v_proj|o_proj).*",
        verbose=False,
    )

    # Test LoRA model (without training)
    print("\n4. Testing LoRA model forward pass (untrained adapters)...")
    outputs_lora = model_lora(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    logits_lora = outputs_lora.logits[0, -1, :]

    print(f"   LoRA logits shape: {logits_lora.shape}")
    print(f"   LoRA logits range: [{logits_lora.min():.4f}, {logits_lora.max():.4f}]")
    print(f"   LoRA logits mean: {logits_lora.mean():.4f}")
    print(f"   LoRA logits std: {logits_lora.std():.4f}")
    print(f"   Contains NaN: {jnp.isnan(logits_lora).any()}")
    print(f"   Contains Inf: {jnp.isinf(logits_lora).any()}")

    # Get argmax
    lora_top_token = jnp.argmax(logits_lora)
    print(f"   Argmax token ID: {lora_top_token}")

    # Check if logits at position 0
    print(f"\n5. Checking logit value at position 0:")
    print(f"   Base model logits[0]: {logits_base[0]:.4f}")
    print(f"   LoRA model logits[0]: {logits_lora[0]:.4f}")

    # Check difference
    logits_diff = logits_lora - logits_base
    print(f"\n6. LoRA vs Base difference:")
    print(f"   Max absolute difference: {jnp.abs(logits_diff).max():.4f}")
    print(f"   Mean absolute difference: {jnp.abs(logits_diff).mean():.4f}")

    # Simulate what happens if LoRA state is lost
    print("\n7. Simulating LoRA state loss...")
    state = model_lora.to_state()

    # Access through state.model (property)
    print("   Accessing via state.model property...")
    module_via_property = state.model
    outputs_property = module_via_property(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    logits_property = outputs_property.logits[0, -1, :]

    print(f"   Via property logits range: [{logits_property.min():.4f}, {logits_property.max():.4f}]")
    print(f"   Via property argmax: {jnp.argmax(logits_property)}")

    # Check if they match
    if jnp.allclose(logits_lora, logits_property):
        print("   ✓ Logits match - state preserved correctly")
    else:
        print("   ✗ Logits differ - potential state loss!")
        print(f"   Max diff: {jnp.abs(logits_lora - logits_property).max():.4f}")


def check_eos_triggering(model_id: str):
    """Check if generated tokens trigger EOS condition."""
    print("\n" + "=" * 60)
    print("EOS Triggering Analysis")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    eos_list = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]

    print(f"\nEOS token IDs: {eos_list}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")

    # Simulate token generation
    print("\n Simulating token sequence:")
    test_tokens = [0, 1, 2, 151643]  # Common problematic tokens

    for token in test_tokens:
        is_eos = token in eos_list
        is_pad = token == tokenizer.pad_token_id
        print(f"  Token {token:6d}: EOS={is_eos}, PAD={is_pad}")

        if is_eos:
            print(f"    ⚠️  Would trigger is_sent_finished=True!")


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B"  # Use small model for testing

    print("GRPO LoRA Token=0 Debug Analysis")
    print("=" * 60)

    # 1. Analyze tokenizer
    tokenizer = analyze_tokenizer_config(model_id)

    # 2. Test logits behavior
    # test_base_model_logits(model_id)  # Commented out for speed, uncomment if needed

    # 3. Check EOS triggering
    check_eos_triggering(model_id)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print("\nKey findings will help identify why tokens become 0.")
    print("If you see warnings about overlapping IDs, that's likely the issue!")
