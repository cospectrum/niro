import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstalledCli:
    executable: Path
    env: dict[str, str]

    def run(
        self,
        *arguments: str | Path,
        input_data: bytes | None = None,
    ) -> str:
        result = subprocess.run(
            [str(self.executable), *(str(argument) for argument in arguments)],
            env=self.env,
            input=input_data,
            capture_output=True,
            check=True,
        )
        return result.stdout.decode()


def install_cli(project_root: Path, root: Path) -> InstalledCli:
    tool_dir = root / "tools"
    bin_dir = root / "bin"
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tool_dir),
        "UV_TOOL_BIN_DIR": str(bin_dir),
    }
    subprocess.run(
        [
            "uv",
            "tool",
            "install",
            "--python",
            sys.executable,
            str(project_root),
        ],
        cwd=project_root,
        env=env,
        check=True,
    )
    executable = bin_dir / ("niro.exe" if os.name == "nt" else "niro")
    return InstalledCli(executable, env)
