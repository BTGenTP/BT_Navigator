#!/usr/bin/env python3
# flake8: noqa
"""
Static post-generation validator for Nav2 / BehaviorTree.CPP XML.

Checks (v1):
- XML well-formed + expected root structure
- tag/attribute allowlist (catalog + learned from reference BTs)
- required ports for catalog-defined skills
- subtree definitions + cycle detection
- basic blackboard usage checks ({var} references)
- basic control-flow structural checks (empty control nodes, suspicious repeat)


"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET


BB_VAR_RE = re.compile(r"\{([^{}]+)\}")


# Minimal known blackboard directions for common Nav2 nodes (heuristic).
# Values: "in" / "out" / "inout".
KNOWN_PORT_DIRECTIONS: Dict[str, Dict[str, str]] = {
    "ComputePathToPose": {"goal": "in", "path": "out"},
    "ComputePathThroughPoses": {"goals": "in", "path": "out"},
    "FollowPath": {"path": "in"},
    "GoalUpdater": {"input_goal": "in", "output_goal": "out"},
    "TruncatePath": {"input_path": "in", "output_path": "out"},
    "RemovePassedGoals": {"input_goals": "in", "output_goals": "inout"},
}


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    file: Optional[str] = None
    xpath: Optional[str] = None
    tag: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"level": self.level, "code": self.code, "message": self.message}
        if self.file:
            out["file"] = self.file
        if self.xpath:
            out["xpath"] = self.xpath
        if self.tag:
            out["tag"] = self.tag
        return out


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_checklist(report: Dict[str, Any]) -> None:
    """
    Print a human-readable checklist to stderr.

    This keeps stdout clean for JSON output when --output is '-'.
    """
    import sys

    issues = report.get("issues") or []
    if not isinstance(issues, list):
        issues = []

    def has(code: str, *, level: Optional[str] = None) -> bool:
        for it in issues:
            if not isinstance(it, dict):
                continue
            if it.get("code") != code:
                continue
            if level is None or it.get("level") == level:
                return True
        return False

    def line(ok: bool, label: str) -> str:
        return f"{'✅' if ok else '❌'} {label}"

    checks = [
        (not has("xml_parse", level="error"), "XML parse (well-formed)"),
        (not has("root_tag", level="error"), "Root tag == <root>"),
        (not has("root_main_tree", level="error"), "root@main_tree_to_execute present"),
        (not has("missing_main_tree_def", level="error"), "MainTree definition exists"),
        (not has("bt_missing_id", level="error"), "All <BehaviorTree> have ID"),
        (not has("bt_duplicate_id", level="error"), "BehaviorTree IDs unique"),
        (not has("subtree_missing_definition", level="error"), "All SubTree IDs defined"),
        (not has("subtree_cycle", level="error"), "No SubTree cycles"),
        (not has("tag_not_allowed", level="error"), "All tags allowed (catalog + refs)"),
        (not has("missing_required_attr", level="error"), "Required attributes present"),
        (not has("unknown_attr", level="error"), "No unknown attributes (strict mode)"),
        (not has("empty_control_node", level="error"), "No empty control nodes"),
        (
            not has("blackboard_unproduced", level="error"),
            "Blackboard vars consistent (strict mode)",
        ),
    ]

    sys.stderr.write("\n[validate_bt_xml] Checklist\n")
    for ok, label in checks:
        sys.stderr.write(f"- {line(bool(ok), label)}\n")


def _catalog_allowlist(catalog: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Returns:
      - allowed_tags
      - allowed_attrs_by_tag
      - required_attrs_by_tag (only for catalog atomic skills, using "optional" convention)
    """
    allowed_tags: Set[str] = set()
    allowed_attrs_by_tag: Dict[str, Set[str]] = {}
    required_attrs_by_tag: Dict[str, Set[str]] = {}

    # Control nodes
    for item in catalog.get("control_nodes_allowed", []) or []:
        tag = item.get("bt_tag")
        attrs = item.get("attributes") or []
        if isinstance(tag, str) and tag:
            allowed_tags.add(tag)
            allowed_attrs_by_tag.setdefault(tag, set()).update({a for a in attrs if isinstance(a, str)})

    # Atomic skills
    for item in catalog.get("atomic_skills", []) or []:
        tag = item.get("bt_tag")
        input_ports = item.get("input_ports", {}) or {}
        output_ports = item.get("output_ports", {}) or {}
        if not (isinstance(tag, str) and tag):
            continue

        allowed_tags.add(tag)
        attrs: Set[str] = set()
        req: Set[str] = set()

        for k, v in dict(input_ports).items():
            if not isinstance(k, str):
                continue
            if k in ("ID", "__shared_blackboard"):
                continue
            attrs.add(k)
            is_optional = isinstance(v, str) and ("optional" in v.lower())
            if not is_optional:
                req.add(k)

        for k in dict(output_ports).keys():
            if isinstance(k, str) and k:
                attrs.add(k)

        allowed_attrs_by_tag.setdefault(tag, set()).update(attrs)
        if req:
            required_attrs_by_tag.setdefault(tag, set()).update(req)

    # Base BT tags always allowed
    for base in ("root", "BehaviorTree", "SubTree"):
        allowed_tags.add(base)
        allowed_attrs_by_tag.setdefault(base, set())

    # Known required attrs for base tags
    required_attrs_by_tag.setdefault("root", set()).add("main_tree_to_execute")
    required_attrs_by_tag.setdefault("BehaviorTree", set()).add("ID")

    # SubTree commonly uses these attrs
    allowed_attrs_by_tag.setdefault("SubTree", set()).update({"ID", "__shared_blackboard"})
    required_attrs_by_tag.setdefault("SubTree", set()).add("ID")

    return allowed_tags, allowed_attrs_by_tag, required_attrs_by_tag


def _scan_reference_allowlist(reference_dir: Path) -> Tuple[Set[str], Dict[str, Set[str]], Set[str]]:
    """
    Learns an allowlist from reference BT XMLs.
    Returns:
      - tags
      - attrs_by_tag
      - blackboard_vars
    """
    tags: Set[str] = set()
    attrs_by_tag: Dict[str, Set[str]] = {}
    bb_vars: Set[str] = set()

    for xml_path in sorted(reference_dir.rglob("*.xml")):
        if "__generated" in xml_path.parts:
            continue
        try:
            tree = ET.parse(str(xml_path))
        except Exception:
            # Reference trees should parse, but don't hard-fail the validator setup.
            continue
        root = tree.getroot()
        for el in root.iter():
            tags.add(el.tag)
            attrs_by_tag.setdefault(el.tag, set()).update(el.attrib.keys())
            for v in el.attrib.values():
                for var in BB_VAR_RE.findall(v or ""):
                    bb_vars.add(var)

    return tags, attrs_by_tag, bb_vars


def _xpath_of(el: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> str:
    parts: List[str] = []
    cur: Optional[ET.Element] = el
    while cur is not None:
        p = parent_map.get(cur)
        if p is None:
            parts.append(f"/{cur.tag}")
            break
        # 1-based index among siblings with same tag (XPath-ish)
        same = [c for c in list(p) if c.tag == cur.tag]
        idx = same.index(cur) + 1
        parts.append(f"/{cur.tag}[{idx}]")
        cur = p
    return "".join(reversed(parts))


def _build_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    parent: Dict[ET.Element, ET.Element] = {}
    for p in root.iter():
        for c in list(p):
            parent[c] = p
    return parent


def _collect_bt_definitions(root: ET.Element) -> Dict[str, ET.Element]:
    out: Dict[str, ET.Element] = {}
    for bt in root.findall("BehaviorTree"):
        bt_id = bt.get("ID")
        if bt_id:
            out[bt_id] = bt
    return out


def _collect_subtree_refs(root: ET.Element) -> Set[str]:
    refs: Set[str] = set()
    for st in root.iter("SubTree"):
        sid = st.get("ID")
        if sid:
            refs.add(sid)
    return refs


def _detect_cycles(bt_defs: Dict[str, ET.Element]) -> List[List[str]]:
    """
    Detect cycles in subtree references between BehaviorTree IDs.
    """
    graph: Dict[str, Set[str]] = {}
    for bt_id, bt_el in bt_defs.items():
        graph[bt_id] = set()
        for st in bt_el.iter("SubTree"):
            sid = st.get("ID")
            if sid:
                graph[bt_id].add(sid)

    visited: Set[str] = set()
    stack: Set[str] = set()
    path: List[str] = []
    cycles: List[List[str]] = []

    def dfs(n: str) -> None:
        visited.add(n)
        stack.add(n)
        path.append(n)
        for m in graph.get(n, set()):
            if m not in bt_defs:
                continue
            if m not in visited:
                dfs(m)
            elif m in stack:
                # cycle: m ... n -> m
                if m in path:
                    i = path.index(m)
                    cycles.append(path[i:] + [m])
        path.pop()
        stack.remove(n)

    for node in graph.keys():
        if node not in visited:
            dfs(node)

    return cycles


def _validate_tree(
    *,
    xml_path: Path,
    reference_dir: Optional[Path],
    catalog_path: Path,
    strict_attrs: bool,
    strict_blackboard: bool,
) -> Dict[str, Any]:
    issues: List[Issue] = []

    # Setup allowlists
    catalog = _load_json(catalog_path)
    cat_tags, cat_attrs_by_tag, required_attrs_by_tag = _catalog_allowlist(catalog)

    ref_tags: Set[str] = set()
    ref_attrs_by_tag: Dict[str, Set[str]] = {}
    ref_bb_vars: Set[str] = set()
    if reference_dir and reference_dir.exists():
        ref_tags, ref_attrs_by_tag, ref_bb_vars = _scan_reference_allowlist(reference_dir)

    allowed_tags = cat_tags | ref_tags
    allowed_attrs_by_tag: Dict[str, Set[str]] = {}
    for tag in allowed_tags:
        allowed_attrs_by_tag[tag] = set()
        allowed_attrs_by_tag[tag].update(cat_attrs_by_tag.get(tag, set()))
        allowed_attrs_by_tag[tag].update(ref_attrs_by_tag.get(tag, set()))

    # Parse XML
    try:
        tree = ET.parse(str(xml_path))
    except Exception as exc:
        issues.append(
            Issue(level="error", code="xml_parse", message=f"XML parse error: {exc}", file=str(xml_path))
        )
        return {"ok": False, "issues": [i.as_dict() for i in issues]}

    root = tree.getroot()
    parent_map = _build_parent_map(root)

    # Root checks
    if root.tag != "root":
        issues.append(
            Issue(
                level="error",
                code="root_tag",
                message=f"Expected root tag <root>, got <{root.tag}>",
                file=str(xml_path),
                xpath=_xpath_of(root, parent_map),
                tag=root.tag,
            )
        )
    if not root.get("main_tree_to_execute"):
        issues.append(
            Issue(
                level="error",
                code="root_main_tree",
                message="Missing attribute root@main_tree_to_execute",
                file=str(xml_path),
                xpath=_xpath_of(root, parent_map),
                tag=root.tag,
            )
        )

    bt_defs = _collect_bt_definitions(root)
    main_id = root.get("main_tree_to_execute") or ""
    if main_id and main_id not in bt_defs:
        issues.append(
            Issue(
                level="error",
                code="missing_main_tree_def",
                message=f"main_tree_to_execute='{main_id}' has no <BehaviorTree ID='{main_id}'> definition.",
                file=str(xml_path),
                xpath=_xpath_of(root, parent_map),
            )
        )

    # Uniqueness of BehaviorTree IDs
    seen_bt_ids: Set[str] = set()
    for bt in root.findall("BehaviorTree"):
        bt_id = bt.get("ID")
        if not bt_id:
            issues.append(
                Issue(
                    level="error",
                    code="bt_missing_id",
                    message="BehaviorTree missing required attribute ID.",
                    file=str(xml_path),
                    xpath=_xpath_of(bt, parent_map),
                    tag="BehaviorTree",
                )
            )
            continue
        if bt_id in seen_bt_ids:
            issues.append(
                Issue(
                    level="error",
                    code="bt_duplicate_id",
                    message=f"Duplicate BehaviorTree ID: {bt_id}",
                    file=str(xml_path),
                    xpath=_xpath_of(bt, parent_map),
                    tag="BehaviorTree",
                )
            )
        seen_bt_ids.add(bt_id)

    # SubTree definitions
    subtree_refs = _collect_subtree_refs(root)
    missing_defs = sorted([sid for sid in subtree_refs if sid not in bt_defs])
    for sid in missing_defs:
        issues.append(
            Issue(
                level="error",
                code="subtree_missing_definition",
                message=f"SubTree reference ID='{sid}' has no matching <BehaviorTree ID='{sid}'> definition.",
                file=str(xml_path),
            )
        )

    # Cycle detection
    cycles = _detect_cycles(bt_defs)
    for cyc in cycles:
        issues.append(
            Issue(
                level="error",
                code="subtree_cycle",
                message=f"SubTree cycle detected: {' -> '.join(cyc)}",
                file=str(xml_path),
            )
        )

    # Tag/attribute checks + blackboard checks
    produced_bb: Set[str] = set()
    consumed_bb: Set[str] = set()
    all_bb_vars: Set[str] = set()

    for el in root.iter():
        xp = _xpath_of(el, parent_map)

        # Tag allowlist
        if el.tag not in allowed_tags:
            issues.append(
                Issue(
                    level="error",
                    code="tag_not_allowed",
                    message=f"Tag <{el.tag}> is not in allowlist (catalog + reference BTs).",
                    file=str(xml_path),
                    xpath=xp,
                    tag=el.tag,
                )
            )
            continue

        # Required attributes
        req = required_attrs_by_tag.get(el.tag, set())
        for r in sorted(req):
            if r not in el.attrib:
                issues.append(
                    Issue(
                        level="error",
                        code="missing_required_attr",
                        message=f"Missing required attribute '{r}' on <{el.tag}>.",
                        file=str(xml_path),
                        xpath=xp,
                        tag=el.tag,
                    )
                )

        # Unknown attributes
        allowed_attrs = allowed_attrs_by_tag.get(el.tag, set())
        unknown = sorted([a for a in el.attrib.keys() if a not in allowed_attrs])
        if unknown:
            level = "error" if strict_attrs else "warning"
            issues.append(
                Issue(
                    level=level,
                    code="unknown_attr",
                    message=f"Unknown attribute(s) on <{el.tag}>: {', '.join(unknown)}",
                    file=str(xml_path),
                    xpath=xp,
                    tag=el.tag,
                )
            )

        # Blackboard refs
        for attr, value in el.attrib.items():
            for var in BB_VAR_RE.findall(value or ""):
                all_bb_vars.add(var)
                dir_map = KNOWN_PORT_DIRECTIONS.get(el.tag, {})
                direction = dir_map.get(attr, "in")  # default conservative: treat as input
                if direction in ("out", "inout"):
                    produced_bb.add(var)
                if direction in ("in", "inout"):
                    consumed_bb.add(var)

    # Control-node structural checks (very basic)
    CONTROL_NODES = {
        "Sequence",
        "Fallback",
        "ReactiveSequence",
        "ReactiveFallback",
        "RoundRobin",
        "PipelineSequence",
        "RateController",
        "DistanceController",
        "SpeedController",
        "KeepRunningUntilFailure",
        "Repeat",
        "Inverter",
        "RecoveryNode",
    }
    for el in root.iter():
        if el.tag not in CONTROL_NODES:
            continue
        xp = _xpath_of(el, parent_map)
        if len(list(el)) == 0:
            issues.append(
                Issue(
                    level="error",
                    code="empty_control_node",
                    message=f"Control node <{el.tag}> has no children.",
                    file=str(xml_path),
                    xpath=xp,
                    tag=el.tag,
                )
            )
        if el.tag == "Repeat" and "num_cycles" not in el.attrib:
            issues.append(
                Issue(
                    level="warning",
                    code="repeat_unbounded",
                    message="Repeat without num_cycles may be non-terminating (warning).",
                    file=str(xml_path),
                    xpath=xp,
                    tag="Repeat",
                )
            )

    # Blackboard consistency (heuristic)
    missing_bb = sorted(consumed_bb - produced_bb)
    if missing_bb:
        level = "error" if strict_blackboard else "warning"
        issues.append(
            Issue(
                level=level,
                code="blackboard_unproduced",
                message=(
                    "Blackboard variable(s) consumed but never produced (heuristic): "
                    + ", ".join(missing_bb)
                ),
                file=str(xml_path),
            )
        )

    # Informational: compare against reference vars to spot novel vars
    novel_bb = sorted(all_bb_vars - ref_bb_vars) if ref_bb_vars else []
    if novel_bb:
        issues.append(
            Issue(
                level="warning",
                code="blackboard_novel_vars",
                message="Blackboard variable(s) not seen in reference BTs: " + ", ".join(novel_bb),
                file=str(xml_path),
            )
        )

    ok = not any(i.level == "error" for i in issues)
    return {
        "ok": ok,
        "file": str(xml_path),
        "summary": {
            "issues_total": len(issues),
            "errors": sum(1 for i in issues if i.level == "error"),
            "warnings": sum(1 for i in issues if i.level == "warning"),
        },
        "issues": [i.as_dict() for i in issues],
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a Nav2 BehaviorTree.CPP XML file (static checks).")
    p.add_argument("xml", nargs="?", type=str, help="Path to the BT XML to validate.")
    p.add_argument("--xml-path", type=str, default=None, help="Alternative to positional xml path.")
    p.add_argument(
        "--catalog",
        type=str,
        default=str(Path(__file__).resolve().parent / "bt_nodes_catalog.json"),
        help="Path to bt_nodes_catalog.json (default: BT_Navigator/script/bt_nodes_catalog.json).",
    )
    p.add_argument(
        "--reference-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "behavior_trees"),
        help="Directory of reference BT XMLs used to learn allowlist/blackboard vars.",
    )
    p.add_argument("--no-reference-scan", action="store_true", help="Disable reference BT scanning.")
    p.add_argument("--strict-attrs", action="store_true", help="Unknown attributes are errors (default: warnings).")
    p.add_argument(
        "--strict-blackboard",
        action="store_true",
        help="Blackboard 'consumed but not produced' becomes an error (default: warning).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report to stdout (default).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="-",
        help="Output report path (default: '-' for stdout).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    xml_arg = args.xml or args.xml_path
    if not xml_arg:
        raise SystemExit("Missing BT XML path. Provide positional <xml> or --xml-path <xml>.")
    xml_path = Path(xml_arg).resolve()
    catalog_path = Path(args.catalog).resolve()
    reference_dir = None if args.no_reference_scan else Path(args.reference_dir).resolve()

    report = _validate_tree(
        xml_path=xml_path,
        reference_dir=reference_dir,
        catalog_path=catalog_path,
        strict_attrs=bool(args.strict_attrs),
        strict_blackboard=bool(args.strict_blackboard),
    )

    _print_checklist(report)

    out_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(out_text)
    else:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")

    if report.get("ok"):
        print("BT XML is valid ! ✅")
        return 0
    else:
        print("BT XML is invalid ❌.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


