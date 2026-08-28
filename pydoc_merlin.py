#!/usr/bin/env python3
"""Run ``pydoc`` with inherited third-party members hidden.

``pydoc`` always walks the full MRO via :func:`inspect.getmembers`, so every
class deriving from ``torch.nn.Module`` drags the whole ``nn.Module`` API into
the output.  Sphinx avoids this by leaving ``inherited-members`` out of
``autodoc_default_options``; ``pydoc`` has no such switch.

Rather than forking ``pydoc``, this wrapper monkeypatches the single seam both
renderers go through -- the module-level ``pydoc.classify_class_attrs`` helper,
which ``TextDoc.docclass`` and ``HTMLDoc.docclass`` call to enumerate a class's
attributes.  Dropping entries there means the ``Methods inherited from ...``
sections are never emitted (``spill()`` only prints a header when it has
something to put under it), and nothing else about pydoc's behaviour changes.

Usage mirrors ``python -m pydoc``::

    python pydoc_merlin.py merlin                 # text, torch noise removed
    python pydoc_merlin.py merlin.algorithms      # any dotted name works
    python pydoc_merlin.py -w merlin              # HTML, same filtering
    python pydoc_merlin.py --inherited none merlin
    python pydoc_merlin.py --inherited all merlin # stock pydoc behaviour

Modes for ``--inherited``:

``internal`` (default)
    Keep members inherited from classes inside ``--package`` (``merlin`` by
    default); drop everything inherited from outside it.  MerLin's own class
    hierarchy stays visible, ``torch.nn.Module`` disappears.
``none``
    Drop every inherited member.  This is what the Sphinx build renders.
``all``
    No filtering; identical to ``python -m pydoc``.
"""

from __future__ import annotations

import argparse
import pydoc
import sys

DEFAULT_PACKAGES = ("merlin",)

_STOCK_CLASSIFY = pydoc.classify_class_attrs

# Set by --debug-classify. Traces go to stderr, never stdout: stdout is the
# rendered document and anything printed there ends up inside the doc text.
_DEBUG = False


def _module_of(cls: type) -> str:
    return getattr(cls, "__module__", "") or ""


def _is_internal(cls: type, packages: tuple[str, ...]) -> bool:
    """True if *cls* is defined inside one of *packages*."""
    module = _module_of(cls)
    val = any(module == pkg or module.startswith(pkg + ".") for pkg in packages)
    if _DEBUG:
        print(f"{module} {val}", file=sys.stderr)
    return val


def install_filter(mode: str, packages: tuple[str, ...]) -> None:
    """Patch ``pydoc.classify_class_attrs`` to hide inherited members.

    Both ``TextDoc.docclass`` and ``HTMLDoc.docclass`` resolve
    ``classify_class_attrs`` from pydoc's globals at call time, so a single
    rebinding covers text, HTML and the ``-p``/``-b`` servers.
    """
    if mode == "all":
        return

    def classify_class_attrs(cls):
        attrs = _STOCK_CLASSIFY(cls)

        # Never filter the excluded classes themselves: `pydoc torch.nn.Module`
        # must still document torch.nn.Module.
        if mode == "internal" and not _is_internal(cls, packages):
            return attrs

        kept = []
        for entry in attrs:
            _name, _kind, home, _value = entry
            if home is cls:
                kept.append(entry)
            elif mode == "internal" and _is_internal(home, packages):
                kept.append(entry)
        return kept

    pydoc.classify_class_attrs = classify_class_attrs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pydoc_merlin",
        description="python -m pydoc, minus inherited third-party members.",
        add_help=False,
    )
    parser.add_argument(
        "--inherited",
        choices=("internal", "none", "all"),
        default="internal",
        help="which inherited members to keep (default: internal)",
    )
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        metavar="PKG",
        help="package treated as internal; repeatable (default: merlin)",
    )
    parser.add_argument(
        "--debug-classify",
        action="store_true",
        help="trace each class's module and internal/external verdict to stderr",
    )
    parser.add_argument(
        "--help",
        action="help",
        help="show this message and exit",
    )
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    global _DEBUG
    _DEBUG = args.debug_classify

    packages = tuple(args.packages or DEFAULT_PACKAGES)
    install_filter(args.inherited, packages)

    # Hand the remaining arguments to pydoc's own CLI so every flag it supports
    # (-w, -k, -p, -b, -n) keeps working.
    sys.argv = [sys.argv[0], *rest]
    try:
        pydoc.cli()
    except SystemExit as exc:  # pydoc.cli() exits on bad usage
        return exc.code if isinstance(exc.code, int) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
