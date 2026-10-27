"""The hand-written harness. One file per responsibility; every runtime action
is printed with the name of the file that performed it."""


def logger(filename: str):
    def log(msg: str) -> None:
        print(f"[{filename}] {msg}", flush=True)
    return log
