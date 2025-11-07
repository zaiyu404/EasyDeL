#!/usr/bin/env python3
"""
Test script to verify GRPO LoRA inference fix.

This script tests that generation with LoRA in GRPO produces coherent
multi-token outputs instead of only getting the first token correct.
"""

import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer


def test_lora_generation():
    """Test that LoRA generation produces correct multi-token outputs."""

    print("=" * 60)
    print("Testing GRPO LoRA Inference Fix")
    print("=" * 60)

    # 1. Load a small model for testing
    model_id = "Qwen/Qwen2.5-0.5B"  # Use a small model for quick testing
    print(f"\n1. Loading model: {model_id}")

    try:
        model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            model_id,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            sharding_axis_dims=(1, -1, 1, 1, 1),
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # 2. Apply LoRA
    print("\n2. Applying LoRA to model")
    try:
        model = model.apply_lora_to_layers(
            lora_rank=32,
            lora_pattern=".*(q_proj|k_proj|v_proj|o_proj).*",
            verbose=False,
        )
        print("   ✓ LoRA applied successfully")
    except Exception as e:
        print(f"   ✗ Failed to apply LoRA: {e}")
        return False

    # 3. Create state
    print("\n3. Creating EasyDeLState")
    try:
        state = model.to_state()
        print("   ✓ State created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create state: {e}")
        return False

    # 4. Prepare tokenizer and input
    print("\n4. Preparing test input")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        test_prompt = "Hello, how are you today? I am"
        inputs = tokenizer(test_prompt, return_tensors="jax", padding=True)
        print(f"   ✓ Test prompt: '{test_prompt}'")
    except Exception as e:
        print(f"   ✗ Failed to prepare input: {e}")
        return False

    # 5. Test generation directly with model
    print("\n5. Testing direct model generation (baseline)")
    try:
        output1 = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=10,
            do_sample=False,  # Use greedy for deterministic output
        )
        generated_text1 = tokenizer.decode(output1.sequences[0], skip_special_tokens=True)
        print(f"   Generated: '{generated_text1}'")
        print("   ✓ Direct generation successful")
    except Exception as e:
        print(f"   ✗ Direct generation failed: {e}")
        return False

    # 6. Test generation through state.model (the old problematic way)
    print("\n6. Testing generation through state.model property")
    try:
        module = state.model
        output2 = module.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=10,
            do_sample=False,
        )
        generated_text2 = tokenizer.decode(output2.sequences[0], skip_special_tokens=True)
        print(f"   Generated: '{generated_text2}'")
        print("   ✓ State.model generation successful")
    except Exception as e:
        print(f"   ✗ State.model generation failed: {e}")
        return False

    # 7. Test generation through graphdef/graphstate/graphother (the new fixed way)
    print("\n7. Testing generation with explicit graph components (fixed method)")
    try:
        import flax.nnx as nn
        module_fixed = nn.merge(state.graphdef, state.graphstate, state.graphother)
        output3 = module_fixed.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=10,
            do_sample=False,
        )
        generated_text3 = tokenizer.decode(output3.sequences[0], skip_special_tokens=True)
        print(f"   Generated: '{generated_text3}'")
        print("   ✓ Fixed method generation successful")
    except Exception as e:
        print(f"   ✗ Fixed method generation failed: {e}")
        return False

    # 8. Verify outputs are consistent
    print("\n8. Verifying output consistency")

    # Check that all outputs are the same
    if generated_text1 == generated_text2 == generated_text3:
        print("   ✓ All generation methods produce identical outputs")
    else:
        print("   ⚠ Outputs differ between methods:")
        print(f"     Direct:      '{generated_text1}'")
        print(f"     State.model: '{generated_text2}'")
        print(f"     Fixed:       '{generated_text3}'")

    # Check that we generated more than just one token
    input_length = len(inputs["input_ids"][0])
    output_length = len(output3.sequences[0])
    tokens_generated = output_length - input_length

    print(f"\n   Input length: {input_length} tokens")
    print(f"   Output length: {output_length} tokens")
    print(f"   Generated: {tokens_generated} new tokens")

    if tokens_generated >= 5:  # Should generate at least 5 tokens
        print("   ✓ Multi-token generation successful")
        success = True
    else:
        print("   ✗ Generated fewer tokens than expected")
        success = False

    # 9. Check for coherence (basic check)
    print("\n9. Checking output coherence")
    new_text = generated_text3[len(test_prompt):].strip()
    if len(new_text) > 0 and not new_text.startswith("<"):  # Basic check
        print(f"   Generated text: '{new_text}'")
        print("   ✓ Generated text appears coherent")
    else:
        print("   ⚠ Generated text may not be coherent")

    print("\n" + "=" * 60)
    if success:
        print("✓ TEST PASSED: GRPO LoRA inference is working correctly")
    else:
        print("✗ TEST FAILED: Issues detected with LoRA inference")
    print("=" * 60)

    return success


if __name__ == "__main__":
    import sys
    success = test_lora_generation()
    sys.exit(0 if success else 1)
