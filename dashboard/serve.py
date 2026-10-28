"""Serve the dashboard pages over http (the 3D avatar can't load from a file).

Browsers block a page opened as a file (double-clicked) from loading other local
files such as animation/avatar.glb, so the dashboard needs a local web server.
Only dashboard/ and animation/ are served, never the rest of the repo (data/).

    python dashboard/serve.py                  # http://localhost:8000/dashboard/dashboard.html
    python dashboard/serve.py --trial          # opens the Trial page instead
    python dashboard/serve.py --lan            # also reachable from a phone on the same network
    python dashboard/serve.py --port 8080 --no-open
    python dashboard/serve.py --query "prefix=ME193/ceci/steps/test"   # simulator test topics

Without --lan only this computer can reach it. With --lan anyone on the same
network can open the pages (they only show what the public MQTT broker already
carries). The phone needs internet too, for the broker. Ctrl+C stops it.
"""

import argparse
import functools
import http.server
import pathlib
import socket
import webbrowser

REPO = pathlib.Path(__file__).resolve().parent.parent   # dashboard/ and animation/
SERVED = ("/dashboard/", "/animation/")


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/", "/dashboard", "/dashboard/"):
            self.send_response(302)
            self.send_header("Location", "/dashboard/trial.html")
            self.end_headers()
            return None
        if not path.startswith(SERVED) or ".." in path:
            self.send_error(404)
            return None
        return super().send_head()

    def log_message(self, fmt, *args):
        pass                        # quiet: only errors are printed (log_error)

    def log_error(self, fmt, *args):
        print(f"serve.py: {fmt % args}", flush=True)


def lan_ip():
    """This computer's address on the local network (no packets are sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))     # TEST-NET address: only picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return None


def main():
    ap = argparse.ArgumentParser(description="Serve the dashboard pages.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--query", default="", help="URL query for the pages, e.g. 'prefix=...'")
    ap.add_argument("--trial", action="store_true", help="open the Trial page instead of the dashboard")
    ap.add_argument("--lan", action="store_true", help="listen on the local network too (for a phone)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    handler = functools.partial(Handler, directory=str(REPO))
    server = http.server.ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), handler)
    query = f"?{args.query}" if args.query else ""
    page = "trial.html" if args.trial else "dashboard.html"
    url = f"http://localhost:{args.port}/dashboard/{page}{query}"
    print(f"dashboard: http://localhost:{args.port}/dashboard/dashboard.html{query}", flush=True)
    print(f"trial:     http://localhost:{args.port}/dashboard/trial.html{query}", flush=True)
    if args.lan:
        ip = lan_ip()
        if ip:
            print(f"phone:     http://{ip}:{args.port}/dashboard/trial.html{query}", flush=True)
        else:
            print("phone:     couldn't find this computer's network address", flush=True)
    print("(Ctrl+C to stop)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
