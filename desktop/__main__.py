"""Start the desktop app:  python -m desktop [--browser] [--port N] [--selftest]"""

import argparse
import multiprocessing
import sys
import time
import webbrowser


def main(argv=None):
    multiprocessing.freeze_support()
    from .paths import WEB_ROOT, fix_streams, unblock_bundle
    fix_streams()
    p = argparse.ArgumentParser(prog="LatticeGenerator")
    p.add_argument("--browser", action="store_true",
                   help="open in the default browser instead of an app window")
    p.add_argument("--port", type=int, default=0, help="local port (default: any free one)")
    p.add_argument("--selftest", action="store_true", help="check every feature and exit")
    p.add_argument("--report", help="selftest: also write the result to this file")
    args = p.parse_args(argv)

    if args.selftest:
        from .selftest import run
        sys.exit(run(args.report))

    from .server import Engine
    engine = Engine(WEB_ROOT, port=args.port)
    url = engine.start()
    try:
        if args.browser:
            print(f"Lattice Generator running at {url}\nPress Ctrl+C to stop.")
            webbrowser.open(url)
            while True:
                time.sleep(1)
        else:
            unblock_bundle()
            try:
                from .window import open_window
                open_window(engine, url)
            except Exception as e:                   # no WebView2 / .NET: use the browser
                _browser_fallback(url, f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()


def _browser_fallback(url, reason):
    """Run in the default browser; a message box keeps the app alive until OK."""
    webbrowser.open(url)
    if sys.platform != "win32":
        print(f"No app window ({reason}); running at {url}. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    import ctypes
    ctypes.windll.user32.MessageBoxW(
        None,
        "The app window could not open, so Lattice Generator is running in your "
        "web browser instead.\n\nKeep this message open while you work. "
        f"Click OK to quit.\n\n(Reason: {reason[:300]})",
        "Lattice Generator", 0x40)                  # MB_ICONINFORMATION


if __name__ == "__main__":
    main()
