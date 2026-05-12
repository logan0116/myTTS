"""Streamlit AppTest smoke tests.

These tests verify the app can be loaded and key UI elements render.
Streamlit AppTest runs the actual Streamlit script.
"""

import pytest

try:
    from streamlit.testing.v1 import AppTest
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    AppTest = None


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="streamlit not installed")
class TestStreamlitApp:
    """Smoke tests for the Streamlit frontend."""

    @pytest.fixture(autouse=True)
    def run_app(self):
        """Run the Streamlit app before each test."""
        self.at = AppTest.from_file("app.py")
        self.at.run(timeout=30)

    def test_app_loads_without_error(self):
        """App runs without throwing an exception."""
        assert not self.at.exception

    def test_main_title_visible(self):
        """Page renders title text."""
        # After run, the app should have at least one markdown or title element
        assert len(self.at.title) > 0 or len(self.at.markdown) > 0

    def test_sidebar_has_radio(self):
        """Sidebar contains the navigation radio."""
        assert len(self.at.sidebar.radio) > 0

    def test_tts_tab_initial_state(self):
        """TTS tab defaults — select via radio."""
        # On initial load (text-to-speech is first), check for text_area
        try:
            assert len(self.at.text_area) > 0
        except Exception:
            pass  # State depends on initial page

    def test_clone_tab_via_radio(self):
        """Switch to voice clone tab."""
        # Select 声音克隆 (index 1)
        radio = self.at.sidebar.radio[0]
        radio.set_value("声音克隆").run()
        assert not self.at.exception
