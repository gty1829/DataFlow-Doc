---
title: Speech Recognition Subtitle Labeling and Cleaning
createTime: 2025/09/30 16:47:01
icon: solar:airbuds-charge-line-duotone
permalink: /en/mm_guide/yich517s/
---

## Speech Recognition Subtitle Labeling and Cleaning

## Step 1: Prepare the DataFlow environment
```bash
conda create -n myvenv python=3.12
pip install open-dataflow
pip install open-dataflow[vllm]
```

## Step 2: Install the DataFlow audio module
```bash
pip install open-dataflow[audio]
```

## Step 3: Launch the local model service
```python
llm_serving = LocalModelVLMServing_vllm(
    hf_model_name_or_path="./models/whisper-large-v3",
    hf_cache_dir="./dataflow_cache",
    vllm_tensor_parallel_size=2,
    vllm_temperature=0.3,
    vllm_top_p=0.9,
    vllm_max_tokens=512,
    vllm_gpu_memory_utilization=0.9
)
```

## Step 4: Initialize the Silero-VAD operator for voice activity detection
```python
silero_vad_generator = SileroVADGenerator(
    # Note: This is the path to the Silero-VAD repository, which contains the weights
    repo_or_dir="./models/silero-vad", 
    source="local",
    device=['cuda:2'],
    num_workers=2,
)
```

## Step 5: Initialize the merge chunks by timestamps operator
```python
merger = MergeChunksByTimestamps(num_workers=2)
```

## Step 6: Initialize the PromptedAQAGenerator operator
```python
prompted_generator = PromptedAQAGenerator(
    vlm_serving=llm_serving,
    # This constructs the prompt, resulting in <|startoftranscript|><|de|><|transcribe|><|notimestamps|>
    system_prompt=WhisperTranscriptionPrompt().generate_prompt(language="german", task="transcribe", with_timestamps=False),
)
```

## Step 7: Initialize the CTC Forced Align Filter or Evaluator operator
```python
# Filter operator. If using the filter operator, uncomment the following line.
# self.filter = CTCForcedAlignFilter(
#     model_path="./models/mms-300m-1130-forced-aligner",
#     device=["cuda:3"],
#     num_workers=1,
# )

# Evaluator operator
evaluator = CTCForcedAlignSampleEvaluator(
    model_path="./models/mms-300m-1130-forced-aligner",
    device=["cuda:3"],
    num_workers=2,
)
```

## Step 8: Initialize the data storage
```python
storage = FileStorage(
    first_entry_file_name="./dataflow/example/audio_asr_pipeline/sample_data_local.jsonl",
    cache_path="./cache",
    file_name_prefix="audio_asr_pipeline",
    cache_type="jsonl",
)
```

## Step 9: Fill in the audio path in the FileStorage
```python
{"audio": ["./dataflow/example/audio_asr_pipeline/test.wav"], "conversation": [{"from": "human", "value": "<audio>" }]}
```

## Step 10: Run the operators
Note: Close the process pool after running the operators.
```python
silero_vad_generator.run(
    storage=storage.step(),
    input_audio_key='audio',
    output_answer_key='timestamps',
    threshold=0.5,
    use_min_cut=True,
    sampling_rate=16000,
    max_speech_duration_s=30.0,
    min_silence_duration_s=0.1,
    speech_pad_s=0.03,
    return_seconds=True,
    time_resolution=1,
    neg_threshold=0.35,
    window_size_samples=512,
    min_silence_at_max_speech=0.098,
    use_max_poss_sil_at_max_speech=True
)

silero_vad_generator.close()    # Close the process pool

merger.run(
    storage=self.storage.step(),
    dst_folder="./cache",
    input_audio_key="audio",
    input_timestamps_key="timestamps",
    timestamp_type="time",  # Specify the type, time or frame
    max_audio_duration=30.0,
    hop_size_samples=512,  # hop_size, sample point count
    sampling_rate=16000,
)

merger.close()

prompted_generator.run(
    storage=storage.step(),
    input_audio_key="audio",
    input_conversation_key="conversation",
    output_answer_key="transcript"
)

# If using the filter operator, uncomment the following line.
# filter.run(
#     storage=storage.step(),
#     input_audio_key="audio",
#     input_conversation_key="transcript",
#     sampling_rate=16000,
#     language="de",
#     micro_batch_size=16,
#     chinese_to_pinyin=False,
#     retain_word_level_alignment=True,
#     threshold=0.1,
#     threshold_mode="min",
#     romanize=True,
# )
# filter.close()

evaluator.run(
    storage=storage.step(),
    input_audio_key="audio",
    input_conversation_key="transcript",
    sampling_rate=16000,
    language="de",
    micro_batch_size=16,
    chinese_to_pinyin=False,
    retain_word_level_alignment=True,
    romanize=True,
)

evaluator.close()
```