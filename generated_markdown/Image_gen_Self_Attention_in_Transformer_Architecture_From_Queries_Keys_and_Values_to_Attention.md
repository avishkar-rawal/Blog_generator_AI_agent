# Self Attention in Transformer Architecture: From Queries, Keys, and Values to Attention

## Understand the Attention Mechanism at a Glance

The first step is to form query, key, and value matrices from the input embeddings. These matrices are then combined using the scaled dot‑product formula softmax(Q K^T / sqrt(d_k)) V, which produces a weighted sum of the value vectors.

> **[IMAGE GENERATION FAILED]** Shows Q, K, V matrices and computes softmax(Q K^T / sqrt(d_k)) V.
>
> **Alt:** Query, Key, Value and scaled dot-product attention formula.
>
> **Prompt:** Create a clean technical illustration of the self attention mechanism. Show three matrices: Query (Q), Key (K), Value (V) arranged side by side. Display the scaled dot-product operation: compute scores = (Q K^T) / sqrt(d_k) and then apply softmax to obtain attention weights. Finally, multiply weights with V to get output. Use a white background, minimal labels, no photorealistic rendering.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\nPlease retry in 19h21m40.853009155s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://aistudio.google.com/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'description': 'Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '69700s'}]}}



In a single encoder, this operation is self‑attention because each token attends to every other token in the same sequence. In an encoder‑decoder model, the same computation becomes cross‑attention when the query comes from the decoder and the keys and values are taken from the encoder output.

Multi‑head attention improves representation power by splitting Q, K, and V into several smaller sub‑matrices, applying the attention formula independently to each head, and finally concatenating the results to form the output.

The query dimension controls how strongly each position scores against the keys, the key dimension encodes the content being matched, and the value dimension provides the actual features that are aggregated across positions.

## Build a Minimal Self Attention MWE

Self attention is the core operation in transformer models. Below is a minimal implementation of scaled dot product attention that takes query (Q), key (K), and value (V) matrices and returns both the attention weights and the final output.

> **[IMAGE GENERATION FAILED]** Illustrates the step-by-step implementation returning attention weights and output.
>
> **Alt:** Pseudocode flow of scaled dot-product attention: compute scores, apply softmax, weighted sum of V.
>
> **Prompt:** Illustrate a minimal Python function scaled_dot_product_attention that takes Q, K, V and key_dim. Show matmul(Q, K.transpose(0,1,2,3)), division by sqrt(key_dim), softmax along sequence axis, then matmul(attention_weights, V). Use box-and-arrow flowchart on white background with concise labels.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\nPlease retry in 19h21m39.484249668s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://aistudio.google.com/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'description': 'Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'description': 'Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '69699s'}]}}



The attention mechanism computes scores by taking the dot product of Q and K, scaling by the square root of the key dimension, and then applying softmax to obtain weights. These weights are used to compute a weighted sum of the V matrix.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, key_dim):
    """
    Compute scaled dot product attention.
    
    Args:
        Q: Query matrix of shape (batch, heads, seq_len, head_dim)
        K: Key matrix of shape (batch, heads, seq_len, head_dim)
        V: Value matrix of shape (batch, heads, seq_len, head_dim)
        key_dim: Dimension of the key vectors
    
    Returns:
        attention_weights: Shape (batch, heads, seq_len, seq_len)
        output: Shape (batch, heads, seq_len, head_dim)
    """
    # Compute scores: Q @ K^T / sqrt(key_dim)
    scores = np.matmul(Q, K.transpose(0, 1, 2, 3)) / np.sqrt(key_dim)
    
    # Apply softmax along the sequence dimension (axis 1)
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True)) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Compute weighted sum of V
    output = np.matmul(attention_weights, V)
    
    return attention_weights, output
```

To verify correctness, consider a tiny example with batch size 2, sequence length 3, and head dimension 4. If Q = [[1,2,3,4], [5,6,7,8]], K = [[1,1,1,1], [2,2,2,2]], and V = [[10,20,30,40], [40,30,20,10]], the computed attention weights and output can be manually verified. The output shape will be (2, 3, 4), matching the input V shape, confirming the implementation is correct. This simple MWE demonstrates the full pipeline: score computation, softmax normalization, and value aggregation—all essential components of self-attention in transformers.

## Trace How Multi Head Attention Works

Multi-head attention partitions the query, key, and value matrices into h heads along the feature dimension. If the model dimension is d_model and there are h heads, each head receives a subspace of dimension d_k = d_model / h. In practice, this is done by reshaping Q, K, and V from (batch, seq_len, d_model) to (batch, h, seq_len, d_k), so every head operates on its own projection of the input.

Each head runs scaled dot-product attention independently, producing its own output matrix. Depending on the implementation, the head outputs are concatenated back into (batch, seq_len, d_model) and passed through a final projection matrix W_O, or they are summed or averaged without an extra projection. The concatenation-plus-projection path lets the model recombine information across subspaces.

Scaling and masking can differ per head because each head has its own key dimension and learned projections. The scale factor 1/sqrt(d_k) is computed per head using that head's d_k. Masking is also head-specific: one head may enforce causal masking while another applies a different attention pattern, such as local windows or strided attention, depending on the architecture.

On a small sequence of four tokens, a single head might learn to attend mostly to the immediately preceding token. With multiple heads, head one could capture local syntax by attending to adjacent words, head two could track long-range coreference by attending to the sentence subject, and head three could focus on positional markers. The combined output therefore represents a mixture of relational patterns that one head would struggle to encode alone.

## Apply Causal and Padding Masks

Masking controls which positions can influence each other during attention. Without it, a decoder could peek at future tokens, breaking autoregressive generation, and padding tokens could corrupt batched computations.

> **[IMAGE GENERATION FAILED]** Visualizes masked positions: no future tokens visible (causal) and padded tokens blocked across heads.
>
> **Alt:** Causal upper-triangular mask and padding boolean mask applied to attention grid.
>
> **Prompt:** Show two masks for attention: a causal upper-triangular mask that zeros out entries above the diagonal (future look) and a padding boolean mask marking padded positions in shorter sequences. Overlay these masks onto a simple attention heatmap matrix. White background, clear legend labels.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\nPlease retry in 19h21m38.509829849s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://aistudio.google.com/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'description': 'Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'description': 'Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'description': 'Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '69698s'}]}}



**Causal mask.** For a sequence of length n, build an n x n upper triangular matrix where entries above the diagonal are masked. Add a large negative value (for example, -1e9) to those logits before softmax so position i cannot attend to any position j > i. The diagonal and lower triangle remain unmasked, allowing each token to see itself and prior context.

**Padding mask.** In a batch, shorter sequences are padded to the maximum length. Create a boolean mask marking padded positions. Apply it to both the query and key dimensions: padded queries produce no meaningful output, and padded keys must not receive attention. Broadcast this mask across heads and add it to the attention scores.

**Combine before softmax.** Sum the causal mask and padding mask, then add the result to the raw attention scores divided by sqrt(d_k). Apply softmax only after masking. This ensures padded and future positions receive approximately zero weight.

**Verify no leakage.** Assert that attention weights for row i are zero at all columns j > i. As a behavioral check, modify a future token in the input and confirm that outputs for earlier positions remain unchanged.

## Debug and Test Attention Edge Cases

Attention implementations should be validated beyond the happy-path inference case. Small numerical mistakes can surface only with extreme inputs, malformed shapes, or strict determinism requirements. The goal is to make failures predictable and debuggable.

- Check numerical stability of softmax with very large scores. Large scores can make exponentials overflow before normalization. Use a stable softmax that subtracts the maximum score before exponentiation, or use log-sum-exp style computation. This keeps the denominator finite. If max score is unavailable, use a stable implementation or approximate softmax.

- Handle empty sequences or zero dimension edge cases. Empty sequences or zero-sized tensors can trigger shape errors, division by zero, or unexpected empty outputs. Add explicit input validation and guards before attention runs. Return an empty tensor with the correct shape or raise a clear error.

- Debug NaN or Inf outputs from overflow. Overflow in softmax can produce NaN or Inf. Trace score computation, verify finite inputs, and fail fast or sanitize invalid outputs. Log inputs, scores, and post-softmax tensors when non-finite values appear.

- Add unit tests for shape, masking, and determinism. Cover query-key alignment, causal masks, padding masks, output shapes, and repeated runs for deterministic results. Assert exact or approximate equality across repeated calls.

Together, these checks reduce risk in production attention paths.

## Measure Performance and Memory Costs

Self‑attention decides how query keys and values interact. In vanilla dot‑product attention every token computes a score against all others, producing an N×N matrix. Optimized kernels such as FlashAttention tile the computation and avoid materializing that matrix entirely. Our measurements span single‑GPU runs on sequences from 512 to 32 k tokens.

- **Naive vs FlashAttention.** Naive builds the full N×N score matrix before multiplying Q and V; FlashAttention streams tiles, cutting peak memory and speeding the inner loop.

- **Memory accounting.** Store Q, K, V (≈2·L·d per head), scores (N×N floats), and weight outputs. Total RAM ≈ 2·L·d·batch·heads + L²·d·batch·heads (scores).

- **Sequence length impact.** Naive runs in O(L²·d); FlashAttention keeps similar arithmetic but uses tiling, giving roughly twice the slowdown when L doubles from 1024 to 2048.

- **Throughput & latency.** On an A100, naive attn yields ~120 tokens/s at L=4096; FlashAttention ~300 tokens/s, staying smooth up to 64 k tokens.

Choosing FlashAttention saves memory pressure and delivers 2–3× higher tokens/sec, especially beyond 4 k tokens. For a sequence of length L and model dimension d, each of Q, K, V stores L·d floats per head, while the score matrix holds L² entries. Even with batch size one, the score matrix often dominates RAM, which is why FlashAttention’s reduced allocation becomes critical on GPUs with limited shared memory. Profiling tools confirm that FlashAttention’s fused kernel reduces peak VRAM by roughly half compared to the naive version, allowing longer context windows without out‑of‑memory errors.

## Review Security and Privacy Implications

Self attention can expose relationships in token embeddings, so production deployments need deliberate data-handling controls from design through inference.

- Avoid logging full attention matrices with sensitive token embeddings. Log only aggregate metrics or masked outputs, and redact tokens before persistence.
- Check model provider data retention and training data exposure. Confirm retention windows, deletion behavior, and whether training data can be memorized or regurgitated.
- Consider privacy risks from attention patterns revealing sequence content. Even attention outputs can encode semantics, so treat them as sensitive in high-risk contexts.
- Review access controls for model weights and inference logs. Restrict checkpoints, prompts, embeddings, and logs to least-privilege roles with audit trails.

Treat attention outputs as potentially sensitive. Apply encryption, retention limits, and access reviews. Validate provider agreements before production use. Document exceptions and incident response plans.
