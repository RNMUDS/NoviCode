"""Tests for novicode.spinner."""

from __future__ import annotations

import threading
import time
from io import StringIO
from unittest.mock import patch

from novicode.spinner import Spinner


class TestSpinnerBasic:
    """Core start / stop behaviour."""

    def test_start_and_stop(self):
        """start() launches a thread; stop() joins it."""
        sp = Spinner()
        sp._is_tty = True
        with patch("sys.stderr", new_callable=StringIO):
            sp.start("testing")
            assert sp._running
            assert sp._thread is not None and sp._thread.is_alive()
            sp.stop()
            assert not sp._running
            assert sp._thread is None

    def test_stop_clears_line(self):
        """stop() writes the ANSI clear-line sequence."""
        buf = StringIO()
        sp = Spinner()
        sp._is_tty = True
        with patch("sys.stderr", buf):
            sp.start("clear test")
            time.sleep(0.15)  # let at least one frame render
            sp.stop()
        output = buf.getvalue()
        assert "\033[2K\r" in output

    def test_stop_without_start_is_noop(self):
        """Calling stop() when not running does not raise."""
        sp = Spinner()
        sp.stop()  # should not raise

    def test_double_start_reuses_thread(self):
        """Calling start() twice does not spawn a second thread."""
        sp = Spinner()
        sp._is_tty = True
        with patch("sys.stderr", new_callable=StringIO):
            sp.start("first")
            t1 = sp._thread
            sp.start("second")
            t2 = sp._thread
            assert t1 is t2, "Thread should be reused on double start"
            sp.stop()

    def test_update_changes_message(self):
        """update() changes the rendered message."""
        buf = StringIO()
        sp = Spinner()
        sp._is_tty = True
        with patch("sys.stderr", buf):
            sp.start("alpha")
            time.sleep(0.15)
            sp.update("beta")
            time.sleep(0.15)
            sp.stop()
        output = buf.getvalue()
        assert "alpha" in output
        assert "beta" in output

    def test_rapid_start_stop_no_zombie_threads(self):
        """Rapid start/stop cycles do not leak threads."""
        baseline = threading.active_count()
        sp = Spinner()
        sp._is_tty = True
        with patch("sys.stderr", new_callable=StringIO):
            for _ in range(20):
                sp.start("cycle")
                sp.stop()
        # Allow a tiny margin for other threads
        assert threading.active_count() <= baseline + 1


class TestSpinnerNonTTY:
    """When stderr is not a TTY the spinner is a no-op."""

    def test_start_stop_noop_on_non_tty(self):
        sp = Spinner()
        sp._is_tty = False
        sp.start("hello")
        # Thread should NOT be spawned
        assert sp._thread is None
        sp.stop()
