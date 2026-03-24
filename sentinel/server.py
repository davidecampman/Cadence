"""Server entry point — run with `sentinel-server` or `python -m sentinel.server`."""

from __future__ import annotations

import sys


def main():
    import uvicorn

    host = "0.0.0.0"
    port = 8000

    # Simple arg parsing
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ("--host",) and i + 1 < len(args):
            host = args[i + 1]
        if arg in ("--port", "-p") and i + 1 < len(args):
            port = int(args[i + 1])

    print(f"\n  Sentinel API server starting on http://{host}:{port}")
    print(f"  Frontend: http://localhost:{port}\n")

    uvicorn.run("sentinel.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
