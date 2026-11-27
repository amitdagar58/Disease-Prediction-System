"""Start Streamlit with a Windows-compatible asyncio event-loop policy.

On Windows, Python 3.8+ defaults to ProactorEventLoop. Under some
Python/Streamlit/browser combinations, a browser disconnect can produce
ConnectionResetError (WinError 10054) in Proactor pipe callbacks. Using the
Selector event-loop policy before Streamlit starts avoids that noisy callback.
"""
import asyncio
import sys

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from streamlit.web import cli as stcli  # noqa: E402


if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
    stcli.main()
