---
title: 语音字幕标注与清洗
createTime: 2025/09/30 14:57:15
icon: solar:airbuds-charge-line-duotone
permalink: /zh/mm_guide/eh0i2ywu/
---

## 语音字幕标注与清洗

## 第一步: 准备Dataflow环境
```bash
conda create -n myvenv python=3.12
pip install open-dataflow
pip install open-dataflow[vllm]
```

## 第二步: 安装Dataflow音频模块
```bash
pip install open-dataflow[audio]
```

## 第三步: 启动本地模型服务
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

## 第四步: 初始化Silero-VAD算子, 用于给音频数据添加语音活动检测标签
```python
silero_vad_generator = SileroVADGenerator(
    repo_or_dir="./models/silero-vad", # 注意这个是silero-vad的github仓库路径, 里面自带权重
    source="local",
    device=['cuda:2'],
    num_workers=2,
)
```

## 第五步: 初始化按标签合并音频算子
```python
merger = MergeChunksByTimestamps(num_workers=2)
```

## 第六步: 初始化whisper打标算子
```python
prompted_generator = PromptedAQAGenerator(
    vlm_serving=llm_serving,
    # 这里是构造prompt, 产生的prompt为<|startoftranscript|><|de|><|transcribe|><|notimestamps|>
    system_prompt=WhisperTranscriptionPrompt().generate_prompt(language="german", task="transcribe", with_timestamps=False),    
)
```

## 第七步: 初始化CTC强制对齐过滤/评估算子
```python
# 过滤算子。如果使用filter算子, 取消注释即可
# self.filter = CTCForcedAlignFilter(
#     model_path="./models/mms-300m-1130-forced-aligner",
#     device=["cuda:3"],
#     num_workers=1,
# )

# 评估算子
evaluator = CTCForcedAlignSampleEvaluator(
    model_path="./models/mms-300m-1130-forced-aligner",
    device=["cuda:3"],
    num_workers=2,
)
```

## 第八步: 初始化数据存储
```python
storage = FileStorage(
    first_entry_file_name="./dataflow/example/audio_asr_pipeline/sample_data_local.jsonl",
    cache_path="./cache",
    file_name_prefix="audio_asr_pipeline",
    cache_type="jsonl",
)
```

## 第九步: 按下述格式将数据路径填入FileStorage中
```python
{"audio": ["./dataflow/example/audio_asr_pipeline/test.wav"], "conversation": [{"from": "human", "value": "<audio>" }]}
```

## 第十步: 执行算子
注意到close函数是用来关闭进程池的。
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

silero_vad_generator.close()     # 关闭多进程

merger.run(
    storage=self.storage.step(),
    dst_folder="./cache",
    input_audio_key="audio",
    input_timestamps_key="timestamps",
    timestamp_type="time",  # 手动指定类型
    max_audio_duration=30.0,
    hop_size_samples=512,  # hop_size, 是样本点数量
    sampling_rate=16000,
)

merger.close()

prompted_generator.run(
    storage=storage.step(),
    input_audio_key="audio",
    input_conversation_key="conversation",
    output_answer_key="transcript"
)

# 如果使用filter算子, 取消注释即可
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