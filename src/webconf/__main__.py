"""Run the web interface: python -m src.webconf [--host HOST] [--port PORT]"""
import argparse

from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="rpi-auto-gym web interface")
    parser.add_argument("--host", default=None, help="Bind address (default: AUTOGYM_WEB_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Port (default: AUTOGYM_WEB_PORT)")
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
