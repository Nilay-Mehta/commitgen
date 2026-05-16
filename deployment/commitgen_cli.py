"""commitgen CLI - generate conventional commit messages from staged git diffs
using the locally-deployed Ollama model.

Subcommands:
  commitgen msg     - generate a message and print it (no commit)
  commitgen commit  - generate, ask to confirm, commit (interactive staging if nothing staged)
  commitgen push    - everything `commit` does, plus git push afterwards

Requires:
  - Run inside a git repository
  - Ollama installed and the `commitgen` model registered
    (see deployment/Modelfile and `ollama create commitgen -f Modelfile`)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

MODEL = "commitgen"

# Many real commits in CommitBench end with emoji shortcodes (:wrench:,
# :octocat:) or unicode emojis. The model picked this up. We strip them
# from the subject portion after the conventional-commit colon.
_SHORTCODE_RE = re.compile(r"\s*:[A-Za-z0-9_+-]+:")
_FIRST_COLON_SPACE_RE = re.compile(r": ")


def _clean_message(msg: str) -> str:
    """Trim trailing emoji shortcodes and unicode emojis from a commit message,
    while keeping the conventional-commit prefix intact."""
    m = _FIRST_COLON_SPACE_RE.search(msg)
    subject_start = m.end() if m else 0
    prefix, subject = msg[:subject_start], msg[subject_start:]

    cut = len(subject)
    sc = _SHORTCODE_RE.search(subject)
    if sc:
        cut = min(cut, sc.start())
    for i, ch in enumerate(subject):
        cp = ord(ch)
        if 0x1F300 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
            cut = min(cut, i)
            break
    # Cut at first sentence boundary - conventional commit subjects are
    # one sentence; anything after ". " is the model rambling into a body.
    period_idx = subject.find(". ")
    if period_idx != -1:
        cut = min(cut, period_idx)

    return (prefix + subject[:cut]).rstrip(" .;,:")


def _run(cmd: list[str], check: bool = True, input: str | None = None) -> str:
    # Force UTF-8 decoding. On Windows the default is cp1252 which chokes on
    # ollama's emoji/non-ASCII output. errors="replace" keeps the message
    # readable if a byte still can't decode.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
        input=input,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def check_in_git_repo() -> None:
    try:
        _run(["git", "rev-parse", "--is-inside-work-tree"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        _err("not inside a git repository")
        sys.exit(1)


def get_staged_diff() -> str:
    return _run(["git", "diff", "--cached"])


def get_unstaged_files() -> list[str]:
    """Files git knows about that are dirty + files git doesn't know about yet."""
    changed = _run(["git", "diff", "--name-only"]).splitlines()
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    return [f for f in (changed + untracked) if f]


def prompt_stage_files() -> bool:
    """Show file list, ask user what to stage. Returns True if anything got staged."""
    files = get_unstaged_files()
    if not files:
        _err("nothing to stage and nothing already staged")
        return False
    print("Nothing staged. Unstaged/untracked files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {f}")
    try:
        choice = input("\nWhat to stage? [all / numbers separated by space / cancel]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return False

    if not choice or choice.lower() in ("cancel", "c", "q"):
        print("Cancelled.")
        return False
    if choice.lower() == "all":
        _run(["git", "add", "-A"])
        return True
    try:
        indices = [int(x) for x in choice.split()]
        selected = [files[i - 1] for i in indices]
    except (ValueError, IndexError):
        _err(f"invalid selection: {choice!r}")
        return False
    _run(["git", "add"] + selected)
    return True


def generate_message(diff: str) -> str:
    """Call `ollama run commitgen` with the diff, return the first non-empty line."""
    prompt = f"Write a conventional-commit message for this diff:\n\n{diff}"
    try:
        output = _run(["ollama", "run", MODEL], input=prompt)
    except FileNotFoundError:
        _err("ollama not found on PATH - is it installed?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        _err(f"ollama failed: {e.stderr.strip() if e.stderr else e}")
        sys.exit(1)
    for line in output.splitlines():
        line = line.strip()
        if line:
            return _clean_message(line)
    return ""


def edit_message(initial: str) -> str:
    """Open $EDITOR (or notepad on Windows) with the initial message; return edited."""
    editor = os.environ.get("EDITOR")
    if not editor:
        editor = "notepad" if os.name == "nt" else "nano"
    fd, tmp_path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(initial + "\n")
        subprocess.run([editor, tmp_path], check=True)
        with open(tmp_path, encoding="utf-8") as f:
            edited = f.read().strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    if not edited:
        return initial
    # Use only the first line; rest is treated as discarded body
    return edited.splitlines()[0].strip()


def confirm_loop(diff: str) -> str | None:
    """Show message, accept Y/n/e/r. Returns final message string, or None to cancel."""
    msg = generate_message(diff)
    while True:
        print(f"\nSuggested:\n  {msg}\n")
        try:
            choice = input("Action? [Y/n/e/r]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None
        if choice in ("", "y", "yes"):
            return msg
        if choice in ("n", "no"):
            print("Cancelled.")
            return None
        if choice == "e":
            edited = edit_message(msg)
            print(f"Using edited: {edited}")
            return edited
        if choice == "r":
            print("Retrying...")
            msg = generate_message(diff)
            continue
        print(f"unknown choice: {choice!r} (expected Y/n/e/r)")


def cmd_msg(args: argparse.Namespace) -> int:
    check_in_git_repo()
    diff = get_staged_diff()
    if not diff:
        _err("nothing staged. Run `git add` first or use `commitgen commit`.")
        return 1
    print(generate_message(diff))
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    check_in_git_repo()
    diff = get_staged_diff()
    if not diff:
        if not prompt_stage_files():
            return 0
        diff = get_staged_diff()
        if not diff:
            _err("staging produced no diff - did you select empty files?")
            return 1
    msg = confirm_loop(diff)
    if msg is None:
        return 0
    _run(["git", "commit", "-m", msg])
    print(f"Committed: {msg}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    rc = cmd_commit(args)
    if rc != 0:
        return rc
    print("Pushing...")
    subprocess.run(["git", "push"], check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="commitgen",
        description="LoRA-tuned commit message generator (Qwen2.5-Coder-0.5B fine-tune via Ollama)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("msg", help="Generate a commit message and print it (no commit)")
    sub.add_parser("commit", help="Generate a message and commit (interactive staging if nothing staged)")
    sub.add_parser("push", help="Generate, commit, and git push")
    args = parser.parse_args()

    handlers = {"msg": cmd_msg, "commit": cmd_commit, "push": cmd_push}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
