"""Run one local process, or separate web/worker processes from the same image."""

import argparse
import logging
import signal
from threading import Event, Thread

import uvicorn

from .api import create_app
from .db import Database
from .settings import Settings
from .worker import Worker


def main():
    parser = argparse.ArgumentParser(prog="jev-studio")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve the browser workspace")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--with-worker", action="store_true", help="Run a local worker in this process")
    commands.add_parser("worker", help="Run the durable background worker")
    commands.add_parser("initdb", help="Initialize the database")
    args = parser.parse_args()
    settings = Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    database = Database(settings)
    database.initialize()
    if args.command == "initdb":
        return
    stop = Event()
    if args.command == "worker":
        signal.signal(signal.SIGTERM, lambda *_args: stop.set())
        signal.signal(signal.SIGINT, lambda *_args: stop.set())
        Worker(database, settings).run_forever(stop)
        return
    thread = None
    if args.with_worker:
        thread = Thread(target=Worker(database, settings).run_forever, args=(stop,), daemon=True)
        thread.start()
    try:
        uvicorn.run(create_app(settings), host=args.host, port=args.port, access_log=False)
    finally:
        stop.set()
        if thread:
            thread.join(timeout=5)
        database.engine.dispose()


if __name__ == "__main__":
    main()
