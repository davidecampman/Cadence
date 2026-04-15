"""Server entry point — run with `cadence-server` or `python -m cadence.server`."""

from __future__ import annotations

import sys


def main():
    import uvicorn

    host = "127.0.0.1"
    port = 8000

    # Simple arg parsing
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ("--host",) and i + 1 < len(args):
            host = args[i + 1]
        elif arg in ("--port", "-p") and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"Error: invalid port number: {args[i + 1]}")
                sys.exit(1)

    print(f"\n  Cadence API server starting on http://{host}:{port}")
    print(f"  Frontend: http://localhost:{port}\n")

    uvicorn.run("cadence.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
