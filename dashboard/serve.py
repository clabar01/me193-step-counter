"""Serve the repo over http so the dashboard can load its 3D avatar files.

Browsers block a page opened as a file (double-clicked) from loading other local
files such as animation/avatar.glb, so the 3D avatar needs a local web server.
This is `python -m http.server --bind 127.0.0.1` from the repo root, plus it
opens the dashboard in the browser. Only this computer can reach it.

    python dashboard/serve.py                     # http://localhost:8000/dashboard/dashboard.html
    python dashboard/serve.py --port 8080 --no-open
    python dashboard/serve.py --query "prefix=ME193/ceci/steps/test"   # simulator test topics

Ctrl+C stops it.
"""

import argparse
import functools
import http.server
import pathlib
import webbrowser

REPO = pathlib.Path(__file__).resolve().parent.parent   # dashboard/ and animation/


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass                        # quiet: only errors are printed (log_error)

    def log_error(self, fmt, *args):
        print(f"serve.py: {fmt % args}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Serve the dashboard on localhost.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--query", default="", help="URL query for the dashboard, e.g. 'prefix=...&name=Ceci'")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    handler = functools.partial(Handler, directory=str(REPO))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/dashboard/dashboard.html" + (f"?{args.query}" if args.query else "")
    print(f"dashboard: {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
