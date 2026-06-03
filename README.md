# Primed Generation in vLLM

This guide explains how to use the **"Primed Generation"** feature in vLLM. Priming is a technique where a "starter" model generates the first $k$ tokens of a response, and then a larger or more capable "completer" (main) model takes over to finish the sequence.

---

## 1. How to Use Primed Generation

The priming feature is fully integrated into both the standard Python API (offline inference) and the OpenAI-compatible API server.

### Configuration Parameters
You configure priming by passing a `priming_config` dictionary (or JSON string). The key parameters are:
- `model`: The huggingface model ID for the starter model.
- `num_priming_tokens`: The exact number of tokens the starter model should generate (e.g., `128`).
- `cuda_visible_devices`: (Optional) Comma-separated GPU IDs to restrict the starter model to specific GPUs.
- Any other standard vLLM argument (e.g., `gpu_memory_utilization`, `tensor_parallel_size`, `max_model_len`).

*Note: You can also use `--cuda-visible-devices` on the main model to strictly isolate the main model and the priming model on different GPUs.*

### A. Offline Python API Usage

```python
from vllm import LLM, SamplingParams

# Initialize the main model (Completer) on GPU 0
llm = LLM(
    model="Qwen/Qwen3.5-0.8B",
    cuda_visible_devices="0",
    max_model_len=32768,
    priming_config={
        # Initialize the starter model on GPU 1
        "model": "Qwen/Qwen3.5-4B",
        "num_priming_tokens": 128,
        "cuda_visible_devices": "1",
        "gpu_memory_utilization": 0.4,
        "max_model_len": 32768,
    },
)

# Total sequence length (including the 128 priming tokens) will be 256
sampling_params = SamplingParams(max_tokens=256, temperature=0.7)

prompts = [
    "The philosophical implications of sentient AI are profound because",
    "Artificial intelligence is a branch of computer science that"
]

outputs = llm.generate(prompts, sampling_params)

for out in outputs:
    print(f"Prompt: {out.prompt}")
    print(f"Generated text: {out.outputs[0].text}")
```

### B. Standard vLLM CLI (API Server) Usage

You can launch the OpenAI-compatible server with priming enabled using the standard `vllm serve` CLI (or `api_server` module).

**1. Start the server:**
```bash
# Main model on GPU 1, Starter model on GPU 2
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-0.8B \
    --cuda-visible-devices 1 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 32768 \
    --priming-config '{
        "model": "Qwen/Qwen3.5-4B", 
        "num_priming_tokens": 128, 
        "cuda_visible_devices": "2", 
        "gpu_memory_utilization": 0.4, 
        "max_model_len": 32768
    }'
```

**2. Query the Chat Completions endpoint:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen3.5-0.8B",
           "messages": [
             {"role": "system", "content": "You are a helpful and detailed AI assistant."},
             {"role": "user", "content": "Explain the significance of the Turing Test in the history of artificial intelligence."}
           ],
           "max_tokens": 256,
           "temperature": 0.7
         }'
```
*The server will use the starter model for the first 128 tokens, seamlessly hand off to the main model for the remaining 128 tokens, and return a perfectly formatted OpenAI JSON response with accurate token counts.*

---

## 2. Implementation Details: What, How, and Why

### What was modified
To support this feature gracefully, surgical edits were made to the core entrypoints and configuration files of the vLLM repository:
- `vllm/config/vllm.py`: Added `priming_config` to `VllmConfig`.
- `vllm/engine/arg_utils.py`: Added CLI arguments `--priming-config` and `--cuda-visible-devices` and plumbed them into the `EngineArgs` dataclass.
- `vllm/entrypoints/llm.py`: Modified the offline `LLM` class to support instantiating and orchestrating the priming model.
- `vllm/v1/engine/async_llm.py`: Modified the `AsyncLLM` class (used by the API server) to asynchronously execute the priming phase and stitch the results into the final output stream.

### How it was implemented
1. **Model Initialization**: When `priming_config` is provided in the configuration, a secondary, entirely independent `LLM` instance (`self.priming_llm`) is instantiated.
2. **GPU Isolation**: A context manager temporarily alters the `CUDA_VISIBLE_DEVICES` environment variable during the initialization of the engines. This ensures that the main engine and the priming engine do not collide or compete for memory on the same GPU.
3. **The `generate` Intercept**:
   - Both `LLM.generate` and `AsyncLLM.generate` were intercepted.
   - Before handing the request to the main engine, a request is sent to `self.priming_llm` to generate exactly `num_priming_tokens`.
   - The original user prompt token IDs are concatenated with the generated priming token IDs to create an extended prompt.
   - The user's `SamplingParams` are deep-copied, and properties like `max_tokens` and `min_tokens` are dynamically decreased by `num_priming_tokens`.
   - The extended prompt is then passed to the main engine for completion.
4. **Asynchronous Execution**: In `AsyncLLM`, the synchronous priming generation is wrapped in `asyncio.get_running_loop().run_in_executor()` to prevent the offline priming model from blocking the API server's event loop.
5. **Output Reconstruction**: In `AsyncLLM`, the generated priming text is dynamically prepended to the `RequestOutput` chunks yielded by the main engine. The original prompt (before priming) is restored on the output object. This tricks the OpenAI API wrapper into correctly counting the original prompt tokens while returning the fully merged text stream.

### Why it was implemented this way
- **Surgical and Non-Invasive**: By intercepting the top-level `generate` methods rather than digging into the core `EngineCore`, `Scheduler`, or `PagedAttention` mechanisms, the implementation avoids breaking vLLM's highly optimized continuous batching and memory management logic.
- **Resource Safety**: Instantiating a completely separate `LLM` object guarantees that the models do not share KV caches, CUDA graphs, or internal states. Using `CUDA_VISIBLE_DEVICES` ensures no NCCL or memory contention occurs when running two models in the same Python process.
- **Compatibility**: The implementation is completely compatible with vLLM V1 architecture optimizations, including **torch.compile**, **CUDA Graphs**, and the **OpenAI API Server** standard formatting. By restoring the original prompt strings on the output object, tools like `curl` and the OpenAI Python SDK remain completely unaware that a handoff occurred.