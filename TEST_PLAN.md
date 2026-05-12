# myTTS 测试大纲

## 1. 测试策略

| 层级 | 框架 | 范围 |
|---|---|---|
| 单元测试 | pytest | 引擎封装、存储层、数据模型 |
| API 测试 | pytest + httpx (ASGI) | FastAPI 路由、请求验证、响应格式 |
| 集成测试 | pytest | 模型加载 → 推理 → 音频落盘 全链路 |
| 前端测试 | Streamlit AppTest | UI 渲染、会话状态、API 调用 |

---

## 2. 单元测试

### 2.1 CosyVoice 引擎封装 (`engine/cosyvoice_engine.py`)

```
class TestCosyVoiceEngine:

    test_model_load_success()
    → 给定正确模型路径，加载成功，model 属性非 None

    test_model_load_invalid_path_raises()
    → 给定错误路径，抛出明确异常

    test_sample_rate_after_load()
    → 加载后 sample_rate == 24000

    test_tts_zero_shot_returns_tensor()
    → 给定合法 text + prompt_text + prompt_audio，返回 Tensor generator

    test_tts_zero_shot_empty_text_raises()
    → text 为空字符串，抛出异常

    test_tts_zero_shot_missing_prompt_audio_raises()
    → prompt_audio 文件不存在，抛出异常

    test_tts_cross_lingual_returns_tensor()
    → 跨语言模式正常返回音频 Tensor

    test_tts_instruct_returns_tensor()
    → 指令模式正常返回音频 Tensor

    test_stream_mode_yields_chunks()
    → stream=True 时返回多个 chunk

    test_non_stream_mode_returns_single_chunk()
    → stream=False 时返回单个完整音频

    test_unload_releases_model()
    → unload() 后 model 为 None，显存释放
```

### 2.2 音频存储 (`storage/audio_store.py`)

```
class TestAudioStore:

    test_save_audio_creates_file()
    → 保存 Tensor 生成 WAV 文件，文件存在且大小 > 0

    test_save_audio_returns_valid_path()
    → 返回路径在 storage/audio/ 下

    test_get_audio_returns_bytes()
    → 根据 audio_id 读取并返回字节流

    test_get_audio_not_found_returns_none()
    → 请求不存在的 audio_id，返回 None

    test_save_register_voice()
    → 保存参考音频 + meta.json 到 storage/voices/{voice_id}/

    test_list_registered_voices()
    → 返回已注册声音列表，字段包含 voice_id/voice_name/created_at

    test_delete_registered_voice()
    → 删除指定声音，对应目录及文件不存在

    test_cleanup_old_audio()
    → 超出 TTL 的临时音频被自动清理
```

### 2.3 数据模型 (`models/schemas.py`)

```
class TestTTSRequest:

    test_valid_request_passes_validation()
    → text="你好", voice_id="default" → 通过验证

    test_empty_text_fails_validation()
    → text="" → 422 错误

    test_text_too_long_fails_validation()
    → text > 5000 字符 → 422 错误

    test_invalid_language_returns_default()
    → language="xx" → 使用默认值

    test_stream_defaults_to_false()
    → 不传 stream → 默认为 False


class TestCloneRequest:

    test_valid_request_passes()
    → text + voice_id → 通过

    test_missing_voice_id_fails()
    → 缺少 voice_id → 422

    test_voice_id_format_validation()
    → 非法的 voice_id 格式 → 422
```

---

## 3. API 测试 (FastAPI 路由)

### 3.1 TTS 接口 (`POST /api/tts/text-to-speech`)

```
class TestTTSEndpoint:

    test_tts_with_default_voice_returns_200()
    → 发送合法请求，返回 200 + success=true + audio_url

    test_tts_response_audio_url_is_accessible()
    → 返回的 audio_url 可 GET 访问，返回音频流

    test_tts_empty_text_returns_422()
    → 发送空文本，返回 422 + 错误详情

    test_tts_with_instruction_returns_200()
    → 包含 instruction 字段的请求，正常返回

    test_tts_with_stream_flag_returns_200()
    → stream=true，返回 200（或 SSE 流）

    test_tts_audio_content_type_is_wav()
    → GET audio_url，Content-Type = audio/wav

    test_tts_latency_under_timeout()
    → 短文本 (<50字) 合成延迟 < 30 秒
```

### 3.2 声音克隆接口 (`POST /api/clone/`)

```
class TestCloneEndpoint:

    test_register_voice_with_valid_audio_returns_200()
    → 上传合法音频 + voice_name，返回 voice_id

    test_register_voice_without_name_returns_422()
    → 缺少 voice_name，返回 422

    test_register_voice_unsupported_format_returns_400()
    → 上传 .txt 文件，返回 400 + 格式不支持

    test_synthesize_with_registered_voice_returns_200()
    → 使用 voice_id 合成，返回 audio_url

    test_synthesize_with_unknown_voice_returns_404()
    → 使用不存在的 voice_id，返回 404

    test_synthesize_with_temp_audio_returns_200()
    → 直接上传参考音频合成（不注册），返回 audio_url
```

### 3.3 声音管理接口 (`GET /api/voices`)

```
class TestVoicesEndpoint:

    test_list_voices_returns_array()
    → 返回 voices 数组，包含已注册声音

    test_list_voices_empty_on_fresh_start()
    → 无已注册声音时返回空数组

    test_get_nonexistent_voice_returns_404()
    → GET /api/voices/invalid-id → 404
```

---

## 4. 集成测试

```
class TestTTSPipeline:

    test_full_zero_shot_pipeline()
    → 输入文本 "你好世界" + prompt_audio + prompt_text
    → 引擎推理 → 保存 WAV → 文件可播放
    → 音频 duration > 0

    test_full_cross_lingual_pipeline()
    → 中文参考音频 + 英文目标文本
    → 输出有效音频

    test_full_instruct_pipeline()
    → 指令 "请用开心语气朗读" + 文本
    → 输出有效音频

    test_voice_register_then_synthesize()
    → 注册声音 → 取出 voice_id → 合成 → 两次音色相似

    test_concurrent_requests_are_serialized()
    → 同时发送 2 个请求，不会因显存竞争崩溃
    → 两个请求均返回 200


class TestAudioFormat:

    test_output_is_valid_wav()
    → 用 `soundfile` / `wave` 库可正常解析输出文件

    test_sample_rate_matches_model()
    → 输出音频 sample_rate == 24000

    test_audio_is_mono()
    → 输出为单声道


class TestModelEdgeCases:

    test_very_short_text()
    → 输入 "嗯" (单字) → 正常输出

    test_special_characters()
    → 输入包含数字、标点、英文混合的文本 → 正常输出

    test_multilingual_mixed_text()
    → "今天天气nice，我们去shopping吧" → 正常输出

    test_empty_prompt_audio()
    → 参考音频为静音 → 行为符合预期（不崩溃）

    test_very_long_text()
    → 输入 500 字文本 → 正常输出，超时内完成
```

---

## 5. 前端测试 (Streamlit AppTest)

```
class TestStreamlitApp:

    test_app_loads_without_error()
    → app = AppTest.from_file("app.py"); app.run() → 不抛异常

    test_sidebar_navigation_exists()
    → sidebar 包含 "文本转语音" / "声音克隆" / "声音管理"

    test_tts_tab_has_text_input()
    → 文本转语音页签包含 text_area 和 生成按钮

    test_tts_tab_shows_audio_player_on_success()
    → 点击生成 → st.audio 组件出现

    test_tts_tab_shows_error_on_empty_text()
    → 空文本点击生成 → 显示错误提示

    test_clone_tab_has_file_uploader()
    → 声音克隆页签包含 file_uploader

    test_clone_tab_rejects_invalid_file_type()
    → 上传非音频文件 → 显示格式错误

    test_voice_list_shows_registered_voices()
    → 声音管理页签显示声音列表
```

---

## 6. 测试数据准备

### 6.1 测试用音频

| 文件 | 用途 | 内容 |
|---|---|---|
| `tests/fixtures/prompt_zh.wav` | 中文参考音频 | 3-5秒，清晰朗读，无噪声 |
| `tests/fixtures/prompt_en.wav` | 英文参考音频 | 3-5秒，清晰朗读，无噪声 |
| `tests/fixtures/silence.wav` | 边界测试 | 1秒静音 |
| `tests/fixtures/long_prompt.wav` | 长参考音频 | 15秒 |
| `tests/fixtures/invalid.txt` | 错误格式测试 | 非音频文件 |
| `tests/fixtures/noisy_prompt.wav` | 噪声测试 | 含背景噪声 |

### 6.2 Mock 策略

当模型未下载或 GPU 不可用时，使用以下 mock 策略：

```python
# tests/conftest.py
@pytest.fixture
def mock_engine():
    """Mock CosyVoice 引擎，返回模拟音频 Tensor"""
    with patch('engine.cosyvoice_engine.AutoModel') as mock:
        mock.return_value.inference_zero_shot.return_value = [
            {"tts_speech": torch.randn(1, 24000)}  # 1秒假音频
        ]
        yield mock

@pytest.fixture
def mock_httpx_client():
    """FastAPI TestClient with mock engine"""
    ...
```

---

## 7. 测试执行

```bash
# 全部测试
pytest tests/ -v

# 仅单元测试 (跳过需要模型的测试)
pytest tests/ -v -m "not slow"

# 仅 API 测试
pytest tests/ -v -m "api"

# 包含集成测试 (需要模型)
pytest tests/ -v --run-slow

# 覆盖率报告
pytest tests/ --cov=backend --cov-report=html
```

---

## 8. 测试优先级

| 优先级 | 测试范围 | 说明 |
|---|---|---|
| P0 | API 请求验证 (schemas) | 阻止非法请求，安全第一 |
| P0 | 引擎加载/卸载 | 模型是核心依赖 |
| P1 | TTS 接口 200/422/404 | 核心业务流程 |
| P1 | 声音克隆接口 | 核心业务流程 |
| P1 | 音频存储读写 | 数据持久化 |
| P2 | 并发控制 | 生产稳定性 |
| P2 | 前端 UI 渲染 | 用户体验 |
| P3 | 边界情况 (长文本/特殊字符) | 鲁棒性 |
| P3 | 音频格式验证 | 兼容性 |
