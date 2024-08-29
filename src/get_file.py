from pathlib import Path
import sys


def get_this_filepath() -> str:
    """returns the path of the called filepath from the users
    current working directory"""
    return sys.argv[0]


def get_file(relative_path: str) -> Path:
    """concatenates the passed filepath with the directory
    of the called file"""
    path = Path(get_this_filepath()).absolute()
    return path.parent.joinpath(relative_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] not enough args passed")
    else:
        print(get_file(sys.argv[1]))
