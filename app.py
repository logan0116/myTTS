"""myTTS — Streamlit frontend."""

import io
import time
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = st.secrets.get("backend_url", "http://127.0.0.1:8000")
API_TIMEOUT = 120

st.set_page_config(
    page_title="myTTS - 智能语音合成",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "last_audio_url" not in st.session_state:
    st.session_state.last_audio_url = None
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🎙️ myTTS")
st.sidebar.markdown("智能语音合成与声音克隆")

page = st.sidebar.radio(
    "导航",
    ["文本转语音", "声音克隆", "声音管理"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(f"Backend: {BACKEND_URL}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_post(endpoint: str, json_data: dict | None = None, files: dict | None = None, data: dict | None = None):
    """Call FastAPI backend. Returns (ok, result)."""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if files:
            resp = requests.post(url, files=files, data=data or {}, timeout=API_TIMEOUT)
        else:
            resp = requests.post(url, json=json_data, timeout=API_TIMEOUT)
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.json().get("detail", resp.text)
    except requests.ConnectionError:
        return False, f"无法连接到后端 ({BACKEND_URL})，请确保 FastAPI 已启动。"
    except requests.Timeout:
        return False, "请求超时，请尝试更短的文本。"
    except Exception as e:
        return False, str(e)


def api_delete(endpoint: str):
    """DELETE request to backend."""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        resp = requests.delete(url, timeout=10)
        if resp.status_code == 200:
            return True, resp.json()
        return False, resp.json().get("detail", resp.text)
    except Exception as e:
        return False, str(e)


def api_get(endpoint: str):
    """GET request to backend."""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return True, resp.json() if endpoint.startswith("/api/voices") else resp.content
        return False, resp.json().get("detail", resp.text)
    except requests.ConnectionError:
        return False, f"无法连接到后端 ({BACKEND_URL})"
    except Exception as e:
        return False, str(e)


def fetch_audio(audio_url: str) -> bytes | None:
    """Download generated audio for playback."""
    ok, content = api_get(audio_url)
    if ok and isinstance(content, bytes):
        return content
    return None


def show_audio_player(audio_url: str):
    """Show audio player for a backend audio URL."""
    audio_bytes = fetch_audio(audio_url)
    if audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        st.audio(audio_bytes, format="audio/wav")
        st.download_button(
            "💾 下载音频",
            data=audio_bytes,
            file_name="tts_output.wav",
            mime="audio/wav",
        )
    else:
        st.error("无法加载音频。")


# ---------------------------------------------------------------------------
# Page 1: Text-to-Speech
# ---------------------------------------------------------------------------

if page == "文本转语音":
    st.title("📝 文本转语音")
    st.markdown("输入文本，选择声音参数，生成自然语音。")

    col1, col2 = st.columns([3, 1])

    with col1:
        text = st.text_area(
            "输入文本",
            placeholder="请在此粘贴或输入要合成的文本...",
            height=200,
            max_chars=5000,
        )

    with col2:
        language = st.selectbox(
            "语言",
            ["zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"],
            format_func=lambda x: {
                "zh": "中文", "en": "English", "ja": "日本語",
                "ko": "한국어", "de": "Deutsch", "es": "Español",
                "fr": "Français", "it": "Italiano", "ru": "Русский",
            }.get(x, x),
            index=0,
        )
        instruction = st.text_input(
            "指令控制（可选）",
            placeholder="例如：请用开心语气朗读 / 用广东话表达 / 慢速",
        )
        stream = st.checkbox("启用流式输出", value=False)
        st.caption("流式延迟更低，但输出为单段音频。")

    if st.button("🔊 生成语音", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("请输入文本内容。")
        else:
            with st.spinner("正在合成语音..."):
                started = time.time()
                ok, result = api_post("/api/tts/text-to-speech", json_data={
                    "text": text.strip(),
                    "language": language,
                    "instruction": instruction.strip(),
                    "stream": stream,
                })

            if ok:
                elapsed = time.time() - started
                st.success(f"合成完成 ✓（耗时 {elapsed:.1f}s，时长 {result['duration']}s）")
                show_audio_player(result["audio_url"])
            else:
                st.error(f"合成失败：{result}")

    # Show last audio if available
    if st.session_state.last_audio_bytes and not st.session_state.get("_shown", False):
        st.divider()
        st.caption("上一次生成的音频：")
        st.audio(st.session_state.last_audio_bytes, format="audio/wav")


# ---------------------------------------------------------------------------
# Page 2: Voice Cloning
# ---------------------------------------------------------------------------

elif page == "声音克隆":
    st.title("🎤 声音克隆")
    st.markdown("上传一段参考音频，AI 将学习该声音并进行合成。")

    tab1, tab2 = st.tabs(["一次合成", "注册并合成"])

    # -- Tab: one-shot clone + synthesize --
    with tab1:
        st.subheader("直接上传参考音频并合成")

        ref_audio = st.file_uploader(
            "上传参考音频（用于学习音色）",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="oneshot_audio",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            prompt_text = st.text_input(
                "参考音频对应文本（可选）",
                placeholder="参考音频中说的话语内容",
                key="oneshot_prompt",
            )
        with col_b:
            clone_instruction = st.text_input(
                "合成指令（可选）",
                placeholder="例如：用开心语气朗读",
                key="oneshot_instruct",
            )

        target_text = st.text_area(
            "要合成的目标文本",
            placeholder="输入希望克隆声音说出的文本...",
            height=120,
            max_chars=5000,
            key="oneshot_target",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎤 克隆并合成", type="primary", use_container_width=True):
                if not ref_audio:
                    st.warning("请上传参考音频。")
                elif not target_text.strip():
                    st.warning("请输入目标文本。")
                else:
                    with st.spinner("正在学习声音并合成..."):
                        files = {"audio_file": (ref_audio.name, ref_audio.read(), ref_audio.type)}
                        data = {"voice_name": "temp", "prompt_text": prompt_text}
                        ok, result = api_post("/api/clone/register-voice", files=files, data=data)

                        if not ok:
                            st.error(f"声音学习失败：{result}")
                        else:
                            voice_id = result["voice_id"]
                            ok2, synth = api_post("/api/clone/synthesize", json_data={
                                "text": target_text.strip(),
                                "voice_id": voice_id,
                                "instruction": clone_instruction.strip(),
                            })
                            if ok2:
                                st.success(f"合成完成 ✓（时长 {synth['duration']}s）")
                                show_audio_player(synth["audio_url"])
                            else:
                                st.error(f"合成失败：{synth}")

        with c2:
            pass

    # -- Tab: register voice then use --
    with tab2:
        st.subheader("注册声音到声音库")

        reg_audio = st.file_uploader(
            "上传参考音频",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="reg_audio",
        )
        voice_name = st.text_input("声音名称", placeholder="给这个声音起个名字", key="reg_name")
        reg_prompt = st.text_input(
            "参考音频对应文本（可选）",
            placeholder="参考音频中说的话语",
            key="reg_prompt",
        )

        if st.button("💾 注册声音", use_container_width=True):
            if not reg_audio:
                st.warning("请上传参考音频。")
            elif not voice_name.strip():
                st.warning("请输入声音名称。")
            else:
                with st.spinner("正在注册声音..."):
                    files = {"audio_file": (reg_audio.name, reg_audio.read(), reg_audio.type)}
                    data = {"voice_name": voice_name.strip(), "prompt_text": reg_prompt.strip()}
                    ok, result = api_post("/api/clone/register-voice", files=files, data=data)
                if ok:
                    st.success(f"声音 '{result['voice_name']}' 注册成功！ID: {result['voice_id']}")
                    st.rerun()
                else:
                    st.error(f"注册失败：{result}")


# ---------------------------------------------------------------------------
# Page 3: Voice Management
# ---------------------------------------------------------------------------

elif page == "声音管理":
    st.title("🗂️ 声音管理")
    st.markdown("管理已注册的声音。")

    ok, result = api_get("/api/voices")
    if not ok:
        st.error(f"无法获取声音列表：{result}")
        voices = []
    else:
        voices = result.get("voices", [])

    if not voices:
        st.info("暂无已注册的声音。去「声音克隆」页面上传参考音频注册吧。")
    else:
        st.caption(f"共 {len(voices)} 个声音")

        for i, voice in enumerate(voices):
            with st.expander(f"🎵 {voice['voice_name']}  ({voice['voice_id']})"):
                st.markdown(
                    f"**创建时间**: {voice.get('created_at', 'N/A')[:19]} | "
                    f"**时长**: {voice.get('duration', 0):.1f}s"
                )

                col_a, col_b = st.columns([4, 1])
                with col_a:
                    test_text = st.text_input(
                        "测试文本",
                        value="你好，这是我的声音测试。",
                        key=f"test_text_{voice['voice_id']}",
                    )
                    if st.button("🔊 试听", key=f"listen_{voice['voice_id']}"):
                        with st.spinner("合成中..."):
                            ok2, synth = api_post("/api/clone/synthesize", json_data={
                                "text": test_text,
                                "voice_id": voice["voice_id"],
                            })
                        if ok2:
                            st.audio(fetch_audio(synth["audio_url"]), format="audio/wav")
                        else:
                            st.error(f"合成失败：{synth}")

                with col_b:
                    if st.button("🗑️ 删除", key=f"del_{voice['voice_id']}", type="secondary"):
                        ok_del, msg = api_delete(f"/api/voices/{voice['voice_id']}")
                        if ok_del:
                            st.success("已删除")
                            st.rerun()
                        else:
                            st.error(f"删除失败：{msg}")
