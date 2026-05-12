# myTTS 方案设计文档

## 1. 项目目标

基于 Fun-CosyVoice3-0.5B-2512 模型，构建一个本地化 TTS Web 应用，提供文本转语音与声音克隆两大核心能力。

## 2. 技术栈

| 层级 | 技术 | 说明 |
|---|---|---|
| 前端 | Streamlit | 快速构建 Web UI，支持文件上传、音频播放 |
| 后端 | FastAPI | 异步高性能 API，对接模型推理 |
| 模型 | Fun-CosyVoice3-0.5B-2512 | 阿里通义语音团队开源，0.5B 参数，Apache 2.0 许可 |
| 环境 | Python 3.10 + Conda | 模型官方推荐 |
| 音频处理 | torchaudio / pydub | 格式转换、音频裁剪 |

## 3. 系统架构

```
┌─────────────────────────────────────────────────┐
│                    Streamlit 前端                 │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  文本转语音   │  │       声音克隆            │  │
│  │  - 文本输入   │  │  - 参考音频上传           │  │
│  │  - 语言选择   │  │  - prompt文本输入         │  │
│  │  - 指令控制   │  │  - 目标文本输入           │  │
│  │  - 流式选项   │  │  - 合成音频播放/下载      │  │
│  │  - 音频播放   │  │                          │  │
│  └──────┬───────┘  └───────────┬──────────────┘  │
│         │                      │                  │
│         └──────────┬───────────┘                  │
│                    │ HTTP/REST                    │
└────────────────────┼──────────────────────────────┘
                     │
┌────────────────────┼──────────────────────────────┐
│                    ▼                              │
│              FastAPI 后端                          │
│  ┌─────────────────────────────────────────────┐  │
│  │  POST /api/tts          文本转语音           │  │
│  │  POST /api/clone         声音克隆            │  │
│  │  GET  /api/voices        已学习声音列表      │  │
│  │  GET  /api/audio/{id}    获取音频文件        │  │
│  └────────────────────┬────────────────────────┘  │
│                       │                           │
│  ┌────────────────────┼────────────────────────┐  │
│  │              TTS Engine Layer                │  │
│  │  - CosyVoice AutoModel 加载与管理            │  │
│  │  - 模型预热 / 显存管理                       │  │
│  │  - 推理接口封装 (zero-shot / cross-lingual)  │  │
│  │  - 音频后处理 (格式转换、分段)               │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## 4. 模块设计

### 4.1 后端 (FastAPI)

#### 4.1.1 目录结构

```
backend/
├── main.py              # FastAPI 应用入口
├── config.py            # 配置管理 (模型路径、音频存储路径等)
├── engine/
│   ├── __init__.py
│   └── cosyvoice_engine.py   # CosyVoice 模型封装
├── api/
│   ├── __init__.py
│   ├── tts.py           # TTS 相关路由
│   └── clone.py         # 声音克隆相关路由
├── models/
│   ├── __init__.py
│   └── schemas.py       # Pydantic 请求/响应模型
├── storage/
│   ├── __init__.py
│   └── audio_store.py   # 音频文件存储管理
└── requirements.txt
```

#### 4.1.2 API 设计

**POST /api/tts/text-to-speech**

```json
// Request
{
  "text": "要合成的文本内容",
  "voice_id": "default",            // 可选，选择已注册的声音
  "language": "zh",                 // 可选，语言提示
  "instruction": "请用正常语速朗读",   // 可选，指令控制 (情感/语速/方言)
  "stream": false                   // 是否流式返回
}

// Response (非流式)
{
  "success": true,
  "audio_url": "/api/audio/tts_abc123.wav",
  "duration": 3.5,
  "format": "wav",
  "sample_rate": 24000
}
```

**POST /api/clone/register-voice**

```json
// Request (multipart/form-data)
// - audio_file: 参考音频文件 (.wav/.mp3/.m4a)
// - voice_name: 声音名称 (用于后续标识)
// - prompt_text: 参考音频对应文本 (可选，不填则使用默认prompt)

// Response
{
  "success": true,
  "voice_id": "voice_xyz789",
  "voice_name": "张三的声音",
  "sample_rate": 24000,
  "duration": 10.2,
  "message": "声音学习成功"
}
```

**POST /api/clone/synthesize**

```json
// Request
{
  "text": "要合成的目标文本",
  "voice_id": "voice_xyz789",        // 使用已注册的声音
  "language": "zh",
  "instruction": "",
  "stream": false
}

// Response
{
  "success": true,
  "audio_url": "/api/audio/clone_abc123.wav",
  "duration": 3.5
}
```

**GET /api/voices**

```json
// Response
{
  "voices": [
    {
      "voice_id": "voice_xyz789",
      "voice_name": "张三的声音",
      "created_at": "2026-05-12T10:30:00",
      "duration": 10.2
    }
  ]
}
```

**GET /api/audio/{audio_id}**

返回音频文件流 (audio/wav)。

#### 4.1.3 CosyVoice 引擎封装

```python
# engine/cosyvoice_engine.py 核心设计

class CosyVoiceEngine:
    """CosyVoice 模型生命周期管理"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.sample_rate = 24000

    def load(self):
        """加载模型到显存/内存"""
        self.model = AutoModel(model_dir=self.model_dir)

    def unload(self):
        """释放模型资源"""
        ...

    def tts_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str,
        stream: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        """零样本 TTS / 声音克隆"""
        ...

    def tts_cross_lingual(
        self,
        text: str,
        prompt_audio_path: str,
        stream: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        """跨语言合成"""
        ...

    def tts_instruct(
        self,
        text: str,
        instruction: str,
        prompt_audio_path: str,
        stream: bool = False
    ) -> Generator[torch.Tensor, None, None]:
        """指令控制合成"""
        ...
```

#### 4.1.4 声音注册机制

- 用户上传参考音频 → 后端保存到 `storage/voices/{voice_id}/prompt.wav`
- 声音元信息保存到 `storage/voices/{voice_id}/meta.json`
- 后续 TTS 请求通过 `voice_id` 引用已注册声音
- 同时支持直接上传临时参考音频进行一次性合成（不注册）

### 4.2 前端 (Streamlit)

#### 4.2.1 页面结构

```
┌─────────────────────────────────────────────┐
│  🎙️ myTTS - 智能语音合成                     │
│  ─────────────────────────────────────────── │
│  [文本转语音]  [声音克隆]  [声音管理]           │  ← Sidebar 导航
└─────────────────────────────────────────────┘

Tab 1 - 文本转语音:
  ┌──────────────────────────────────────────┐
  │  输入文本                                 │
  │  ┌──────────────────────────────────────┐│
  │  │ [多行文本输入框]                       ││
  │  └──────────────────────────────────────┘│
  │                                          │
  │  声音选择: [默认 ▼]  语言: [中文 ▼]       │
  │  指令: [正常语速朗读 ____________]        │
  │  □ 启用流式输出                           │
  │                                          │
  │  [🔊 生成语音]                            │
  │  ─────────────────────────────────────── │
  │  🎵 生成结果 (播放器 + 下载按钮)           │
  └──────────────────────────────────────────┘

Tab 2 - 声音克隆:
  ┌──────────────────────────────────────────┐
  │  步骤1: 上传参考音频                       │
  │  ┌──────────────────────────────────────┐│
  │  │ [拖拽或点击上传 .wav/.mp3]             ││
  │  │ 参考文本 (可选): [_________]          ││
  │  └──────────────────────────────────────┘│
  │                                          │
  │  步骤2: 输入目标文本                       │
  │  ┌──────────────────────────────────────┐│
  │  │ [多行文本输入框]                       ││
  │  └──────────────────────────────────────┘│
  │                                          │
  │  [🎤 克隆并合成]   [💾 注册此声音]         │
  │  ─────────────────────────────────────── │
  │  🎵 合成结果 (播放器 + 下载按钮)           │
  └──────────────────────────────────────────┘

Tab 3 - 声音管理:
  ┌──────────────────────────────────────────┐
  │  已注册声音列表                            │
  │  ┌────┬────────┬──────────┬────────────┐ │
  │  │ ID │ 名称   │ 创建时间   │ 操作       │ │
  │  ├────┼────────┼──────────┼────────────┤ │
  │  │  1 │ 张三声 │ 05-12    │ [试听][删除]│ │
  │  └────┴────────┴──────────┴────────────┘ │
  └──────────────────────────────────────────┘
```

#### 4.2.2 核心组件

| 组件 | 功能 |
|---|---|
| 文本输入区 | `st.text_area` 多行文本输入 |
| 音频上传 | `st.file_uploader` 上传参考音频 (.wav/.mp3/.m4a) |
| 参数选择 | `st.selectbox` 选择声音/语言/指令 |
| 音频播放 | `st.audio` 播放生成的音频 |
| 声音管理 | `st.dataframe` 展示已注册声音列表 |
| 状态提示 | `st.spinner` / `st.toast` 展示处理状态 |

## 5. 数据流

### 5.1 文本转语音流程

```
用户输入文本
  → Streamlit 构造 POST /api/tts/text-to-speech 请求
  → FastAPI 接收请求
  → CosyVoiceEngine.tts_zero_shot() 调用模型推理
  → 模型返回音频 Tensor
  → 保存为 WAV 文件到 storage/audio/
  → 返回 audio_url
  → Streamlit 通过 GET /api/audio/{id} 获取音频
  → st.audio 播放
```

### 5.2 声音克隆流程

```
用户上传参考音频 + 输入目标文本
  → Streamlit 构造 POST /api/clone/synthesize (multipart)
  → FastAPI 保存参考音频到临时目录
  → CosyVoiceEngine.tts_zero_shot(text, prompt_text, ref_audio_path)
  → 模型基于参考音频音色合成目标文本
  → 保存合成音频
  → 返回 audio_url
  → Streamlit 播放

可选: 用户点击"注册此声音"
  → POST /api/clone/register-voice
  → 后端将参考音频持久化到 storage/voices/{voice_id}/
  → 写入 meta.json
```

## 6. 模型加载策略

Fun-CosyVoice3-0.5B 为 0.5B 参数，对消费级 GPU 友好：

- **启动加载**: FastAPI 启动时通过 `@app.on_event("startup")` 加载模型
- **常驻内存**: 模型保持常驻，避免每次请求重新加载
- **显存预估**: 0.5B 模型约需 1-2GB 显存 (FP16)，CPU 推理约需 2-4GB 内存
- **并发控制**: 使用 `asyncio.Lock` 保证推理互斥，避免显存溢出

```python
# main.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    engine = CosyVoiceEngine(model_dir=settings.model_dir)
    engine.load()
    app.state.engine = engine
    yield
    # 关闭时释放资源
    engine.unload()
```

## 7. 部署环境要求

| 项目 | 最低配置 | 推荐配置 |
|---|---|---|
| Python | 3.10 | 3.10 |
| GPU | 无 (CPU 可运行) | NVIDIA GPU 4GB+ VRAM |
| 内存 | 8GB | 16GB+ |
| 磁盘 | 10GB (模型+依赖) | 20GB+ (含音频存储) |
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |

## 8. 依赖清单

```
# 模型推理
cosyvoice                    # 从 GitHub 安装
torch >= 2.0
torchaudio
transformers
onnxruntime

# Web 框架
fastapi
uvicorn[standard]
python-multipart

# 前端
streamlit

# 音频处理
pydub
librosa

# 工具
huggingface_hub
pydantic
```

## 9. 风险与注意事项

| 风险 | 应对措施 |
|---|---|
| 模型加载速度慢 | 启动时预热，加载一次常驻内存 |
| 推理延迟较高 | 提供流式选项 (stream=True)，首包延迟 ~150ms；短文本直接非流式 |
| 参考音频质量影响克隆效果 | 前端给出音频质量提示 (建议 3-10 秒，无背景噪声，16kHz+) |
| 声音克隆伦理合规 | 界面添加使用协议勾选，提示用户需获得声音授权 |
| Streamlit 不支持实时流式音频 | 非流式模式下生成完整文件后播放；未来可扩展 WebSocket 实现真正流式 |

## 10. 开发计划

| 阶段 | 内容 | 预估周期 |
|---|---|---|
| Phase 1 | 环境搭建：Conda 环境、模型下载、依赖安装 | 0.5 天 |
| Phase 2 | 后端核心：FastAPI + CosyVoice 引擎封装 + API | 1 天 |
| Phase 3 | 前端页面：Streamlit 三页签 UI | 1 天 |
| Phase 4 | 联调与优化：端到端测试、错误处理、音频格式兼容 | 0.5 天 |

总计约 **3 天** 可完成 MVP。
