#!/usr/bin/env python3
"""
Scan BT reference XML files to derive:
- allowlist of tags and attribute names
- blackboard variable examples (e.g. {goal}, {path})
- subtree definitions/references

Intended for research/evaluation tooling (no ROS2 runtime dependency).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Set, Tuple
from xml.etree import ElementTree as ET


BB_VAR_RE = re.compile(r"\{([^{}]+)\}")


@dataclass
class BbExample:
    file: str
    tag: str
    attr: str
    value: str


def iter_xml_files(
    root_dir: Path,
    *,
    include_generated: bool,
) -> Iterable[Path]:
    if include_generated:
        yield from sorted(root_dir.rglob("*.xml"))
        return
    for p in sorted(root_dir.rglob("*.xml")):
        if "__generated" in p.parts:
            continue
        yield p


def scan_xml_file(
    path: Path,
) -> Tuple[Dict[str, Set[str]], List[BbExample], Set[str], Set[str]]:
    """
    Returns:
      - tag_to_attrs: tag -> set(attr_names)
      - bb_examples: list of (file, tag, attr, value)
      - bt_definitions: BehaviorTree IDs defined in file
      - subtree_references: SubTree IDs referenced in file
    """
    tag_to_attrs: Dict[str, Set[str]] = defaultdict(set)
    bb_examples: List[BbExample] = []
    bt_definitions: Set[str] = set()
    subtree_references: Set[str] = set()

    tree = ET.parse(str(path))
    root = tree.getroot()

    for el in root.iter():
        tag_to_attrs[el.tag].update(el.attrib.keys())

        if el.tag == "BehaviorTree":
            bt_id = el.get("ID")
            if bt_id:
                bt_definitions.add(bt_id)
        if el.tag == "SubTree":
            st_id = el.get("ID")
            if st_id:
                subtree_references.add(st_id)

        for attr, value in el.attrib.items():
            if not value:
                continue
            if "{" not in value:
                continue
            if not BB_VAR_RE.search(value):
                continue
            bb_examples.append(
                BbExample(
                    file=str(path.as_posix()),
                    tag=el.tag,
                    attr=attr,
                    value=value,
                )
            )

    return tag_to_attrs, bb_examples, bt_definitions, subtree_references


def build_report(
    xml_dir: Path,
    *,
    include_generated: bool,
    max_examples_per_var: int,
) -> Dict[str, Any]:
    tag_attrs: DefaultDict[str, Set[str]] = defaultdict(set)
    tag_counts: DefaultDict[str, int] = defaultdict(int)

    bb_var_counts: DefaultDict[str, int] = defaultdict(int)
    bb_var_examples: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)

    all_bt_defs: Set[str] = set()
    all_st_refs: Set[str] = set()

    files = list(iter_xml_files(xml_dir, include_generated=include_generated))
    for f in files:
        t2a, bb_examples, defs, refs = scan_xml_file(f)
        for tag, attrs in t2a.items():
            tag_attrs[tag].update(attrs)
            tag_counts[tag] += 1
        for ex in bb_examples:
            for var in BB_VAR_RE.findall(ex.value):
                bb_var_counts[var] += 1
                if len(bb_var_examples[var]) < max_examples_per_var:
                    bb_var_examples[var].append(
                        {
                            "file": ex.file,
                            "tag": ex.tag,
                            "attr": ex.attr,
                            "value": ex.value,
                        }
                    )
        all_bt_defs.update(defs)
        all_st_refs.update(refs)

    missing_subtrees = sorted(all_st_refs - all_bt_defs)

    tags_out = {
        tag: {
            "attrs": sorted(attrs),
            "files_count": tag_counts[tag],
        }
        for tag, attrs in sorted(tag_attrs.items())
    }

    bb_out = {
        var: {"count": bb_var_counts[var], "examples": bb_var_examples[var]}
        for var in sorted(bb_var_counts.keys())
    }

    return {
        "xml_dir": str(xml_dir.as_posix()),
        "include_generated": include_generated,
        "files_scanned": [str(p.as_posix()) for p in files],
        "tags": tags_out,
        "blackboard_vars": bb_out,
        "subtrees": {
            "definitions": sorted(all_bt_defs),
            "references": sorted(all_st_refs),
            "missing_definitions": missing_subtrees,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan BT XML reference files and output tag/attr allowlist and "
            "blackboard examples."
        )
    )
    p.add_argument(
        "--xml-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "behavior_trees"),
        help=(
            "Directory containing BT XML files "
            "(default: BT_Navigator/behavior_trees)."
        ),
    )
    p.add_argument(
        "--include-generated",
        action="store_true",
        help="Also scan BT_Navigator/behavior_trees/__generated/*.xml.",
    )
    p.add_argument(
        "--max-examples-per-var",
        type=int,
        default=5,
        help="Max examples stored per blackboard variable.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="-",
        help="Output JSON path (default: '-' for stdout).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xml_dir = Path(args.xml_dir).resolve()
    report = build_report(
        xml_dir,
        include_generated=bool(args.include_generated),
        max_examples_per_var=args.max_examples_per_var,
    )

    out_text = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
    )
    if args.output == "-":
        print(out_text)
        return 0
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
