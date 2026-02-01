#!/usr/bin/env python3

"""
Générateur de Behavior Tree pour le projet BT_Navigator.

Principe :
- Reçoit un prompt en langage naturel décrivant la mission du TurtleBot.
- Appelle un LLM (Mistral via API OpenAI-compatible) pour générer UNIQUEMENT
  une structure JSON (liste d'étapes).
- Construit un XML `turtlebot_mission.xml` STRICTEMENT compatible avec les
  nœuds BT autorisés (catalogue JSON).

Sortie par défaut :
  BT_Navigator/behavior_trees/__generated/turtlebot_mission.xml

Variables d'environnement :
- LLM_API_KEY   : clé API (obligatoire si --api-key non fourni)
- LLM_MODEL     : nom du modèle (ex : mistral-large-latest)
- LLM_API_BASE  : URL de base compatible OpenAI (par défaut: https://api.mistral.ai/v1)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree.ElementTree import Comment, Element, ElementTree, SubElement

import datetime

from prompt_guardrails import SafetyReport, analyze_prompt_heuristic, call_moderation_llm

try:  # Support d'un fichier .env facultatif
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


DEFAULT_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
DEFAULT_API_BASE = os.getenv("LLM_API_BASE", "https://api.mistral.ai/v1")
DEFAULT_MODERATION_MODEL = os.getenv("LLM_MODERATION_MODEL", "mistral-large-latest")
DEFAULT_MODERATION_API_BASE = os.getenv("LLM_MODERATION_API_BASE", "https://api.mistral.ai/v1")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # nav4rails/ (assumption based on layout)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "BT_Navigator"
    / "behavior_trees"
    / "__generated"
    / "turtlebot_mission_generated_"
    f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.xml"
)

CATALOG_PATH = SCRIPT_DIR / "bt_nodes_catalog.json"
NAV_SUBTREE_SOURCE_XML = (
    REPO_ROOT / "BT_Navigator" / "behavior_trees" / "navigate_then_spin.xml"
)


@dataclass(frozen=True)
class MissionStep:
    skill: str
    params: Dict[str, Any]
    comment: Optional[str] = None


def _is_interactive_stdin() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise RuntimeError(
            "Entrée utilisateur indisponible (EOF). "
            "Utilisez --mode fail-fast ou --mode auto-rewrite pour exécuter sans interaction."
        ) from exc


def _normalize_prompt(text: str) -> str:
    # Normalisation minimale: whitespace + trim
    out = " ".join((text or "").strip().split())
    return out


def _auto_rewrite_prompt(text: str, *, allowed_skills: List[str], max_chars: int) -> str:
    """
    Réécriture heuristique: réduire le bruit et forcer une formulation "mission → étapes".
    """
    raw = (text or "").strip()
    # Supprime blocs multi-lignes "bruit" (logs) par heuristique simple
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    kept: List[str] = []
    for ln in lines:
        if len(ln) > 240:
            continue
        if ln.lower().startswith(("traceback", "error:", "warning:", "debug:", "info:")):
            continue
        kept.append(ln)
        if len(kept) >= 30:
            break
    compact = " ".join(kept) if kept else _normalize_prompt(raw)

    hint = (
        "Mission TurtleBot (décris uniquement des actions concrètes). "
        "Contraintes: renvoyer une liste d'étapes parmi les skills autorisés. "
        f"Skills autorisés: {', '.join(allowed_skills)}. "
        "Unités: Wait=secondes, Spin=radians, BackUp=meters+m/s, DriveOnHeading=meters+m/s+seconds. "
        "Réponds sans contexte inutile."
    )
    rewritten = f"{hint}\nMission: {compact}"
    rewritten = rewritten.strip()
    if len(rewritten) > max_chars:
        rewritten = rewritten[: max(0, max_chars - 3)].rstrip() + "..."
    return rewritten


def _extract_missing_port(err: Exception) -> Optional[Dict[str, str]]:
    """
    Tente d'extraire {skill, port} depuis les messages d'erreur de _parse_steps().
    """
    msg = str(err)
    # Exemple: "Étape 'Wait': port requis manquant: wait_duration"
    import re

    m = re.search(r"Étape\s+'([^']+)':\s*port requis manquant:\s*([A-Za-z0-9_]+)", msg)
    if not m:
        return None
    return {"skill": m.group(1), "port": m.group(2)}


def _apply_guardrails(
    *,
    text: str,
    guardrails_mode: str,
    moderation_api_key: Optional[str],
    moderation_api_base: str,
    moderation_model: str,
) -> SafetyReport:
    if guardrails_mode == "off":
        return SafetyReport(ok=True, blocked=False, findings=[])

    heur = analyze_prompt_heuristic(text)
    if guardrails_mode == "heuristic":
        return heur

    # hybrid: heuristique + modération optionnelle
    if heur.blocked:
        return heur
    if moderation_api_key and moderation_api_base and moderation_model:
        mod = call_moderation_llm(
            text=text,
            api_key=moderation_api_key,
            api_base=moderation_api_base,
            model=moderation_model,
        )
        # fusion des findings
        if mod.findings:
            merged = list(heur.findings) + list(mod.findings)
            blocked = bool(heur.blocked or mod.blocked)
            return SafetyReport(ok=not blocked, blocked=blocked, findings=merged, redacted_text=heur.redacted_text)
    return heur


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _load_catalog(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Impossible de lire le catalogue: {path}: {exc}") from exc


def _allowed_skills(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in catalog.get("atomic_skills", []):
        sid = item.get("id")
        if isinstance(sid, str) and sid:
            out[sid] = item
    return out


def _parse_steps(raw: str, allowed: Dict[str, Dict[str, Any]]) -> List[MissionStep]:
    cleaned = _strip_markdown_fences(raw)
    try:
        data = json.loads(cleaned)
    except Exception as exc:
        raise RuntimeError(
            f"Réponse LLM non JSON.\nErreur: {exc}\nContenu brut:\n{raw}"
        ) from exc

    if not isinstance(data, list) or not data:
        raise ValueError("La sortie LLM doit être une liste JSON non-vide d'étapes.")

    steps: List[MissionStep] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Étape #{idx} invalide (doit être un objet JSON).")

        skill = item.get("skill")
        if skill not in allowed:
            allowed_list = ", ".join(sorted(allowed.keys()))
            raise ValueError(f"Skill '{skill}' non autorisé. Autorisés: {allowed_list}")

        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"Étape '{skill}': 'params' doit être un objet JSON.")

        # Validation sur la présence des ports requis.
        # Convention: si la description du port contient "optional",
        # alors le port n'est pas requis.
        input_ports = allowed[skill].get("input_ports", {}) or {}
        for port_name, port_desc in input_ports.items():
            if port_name in ("ID", "__shared_blackboard"):
                continue
            is_optional = isinstance(port_desc, str) and ("optional" in port_desc.lower())
            if (not is_optional) and (port_name not in params):
                raise ValueError(f"Étape '{skill}': port requis manquant: {port_name}")

        comment = item.get("comment")
        if comment is not None and not isinstance(comment, str):
            raise ValueError(f"Étape '{skill}': 'comment' doit être une chaîne.")

        steps.append(MissionStep(skill=skill, params=params, comment=comment))

    return steps


def _indent_xml(elem: Element, level: int = 0) -> None:
    indent_str = "  "
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():  # type: ignore[name-defined]
            child.tail = i  # type: ignore[name-defined]
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _add_nav_subtree_definition(root: Element) -> None:
    """
    Ajoute la définition <BehaviorTree ID="NavigateToPoseWithReplanningAndRecovery">...
    en la copiant depuis BT_Navigator/behavior_trees/navigate_then_spin.xml
    (qui la contient déjà).
    """
    if not NAV_SUBTREE_SOURCE_XML.exists():
        raise RuntimeError(f"Fichier source subtree introuvable: {NAV_SUBTREE_SOURCE_XML}")

    import xml.etree.ElementTree as ET

    src_tree = ET.parse(str(NAV_SUBTREE_SOURCE_XML))
    src_root = src_tree.getroot()
    found = None
    for bt in src_root.findall("BehaviorTree"):
        if bt.get("ID") == "NavigateToPoseWithReplanningAndRecovery":
            found = bt
            break

    if found is None:
        raise RuntimeError(
            "Subtree NavigateToPoseWithReplanningAndRecovery introuvable "
            f"dans {NAV_SUBTREE_SOURCE_XML}"
        )

    # Deep copy sans dépendances externes
    copied = ET.fromstring(ET.tostring(found, encoding="utf-8"))
    root.append(copied)


def build_bt_xml(steps: List[MissionStep], catalog: Dict[str, Any]) -> ElementTree:
    allowed = _allowed_skills(catalog)
    tag_by_skill = {sid: allowed[sid]["bt_tag"] for sid in allowed.keys()}

    root = Element("root", main_tree_to_execute="MainTree")
    bt = SubElement(root, "BehaviorTree", ID="MainTree")
    seq = SubElement(bt, "Sequence", name="TurtlebotMission")

    needs_nav_subtree = False

    for step in steps:
        if step.comment:
            seq.append(Comment(f" {step.comment} "))

        bt_tag = tag_by_skill[step.skill]

        if step.skill == "NavigateToGoalWithReplanningAndRecovery":
            # Matche le pattern existant: <SubTree ID="NavigateToPoseWithReplanningAndRecovery" __shared_blackboard="true"/>
            needs_nav_subtree = True
            SubElement(
                seq,
                "SubTree",
                ID="NavigateToPoseWithReplanningAndRecovery",
                __shared_blackboard="true",
            )
            continue

        # Default: node = <Tag ...attrs.../>
        attrs: Dict[str, str] = {}
        for k, v in step.params.items():
            if isinstance(v, bool):
                attrs[k] = "true" if v else "false"
            else:
                attrs[k] = str(v)
        SubElement(seq, bt_tag, **attrs)

    if needs_nav_subtree:
        _add_nav_subtree_definition(root)

    _indent_xml(root)
    return ElementTree(root)


def call_llm_for_steps(
    natural_language_prompt: str,
    api_key: str,
    model: str,
    api_base: str,
    catalog: Dict[str, Any],
) -> List[MissionStep]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Le package 'openai' est requis.\n"
            "Installez-le par exemple avec : pip install openai"
        ) from exc

    allowed = _allowed_skills(catalog)
    allowed_list = sorted(allowed.keys())

    system_prompt = (
        "Tu es un assistant spécialisé Nav2 / BehaviorTree.CPP.\n"
        "Ta tâche: convertir un objectif de mission en une liste JSON d'étapes.\n"
        "\n"
        "Règles STRICTES:\n"
        "- La sortie doit être UNIQUEMENT un JSON valide (aucun texte autour).\n"
        "- Le JSON est une liste d'objets: {\"skill\": <string>, \"params\": <object>, \"comment\": <string optional>}.\n"
        f"- skill doit être UNIQUEMENT parmi: {', '.join(allowed_list)}.\n"
        "- Ne renvoie jamais un skill hors-liste.\n"
        "- Ne renvoie jamais de structure XML.\n"
        "- Utilise des nombres (float/int) pour les paramètres numériques.\n"
        "\n"
        "Catalogue (résumé des skills):\n"
        + "\n".join(
            [
                f"- {sid}: {allowed[sid].get('semantic_description','')}"
                for sid in allowed_list
            ]
        )
        + "\n\n"
        "Exemple:\n"
        "[\n"
        "  {\"skill\": \"Wait\", \"params\": {\"wait_duration\": 2.0}, \"comment\": \"Stabilisation\"},\n"
        "  {\"skill\": \"Spin\", \"params\": {\"spin_dist\": 1.57}, \"comment\": \"Tourner à gauche\"}\n"
        "]\n"
    )

    client = OpenAI(api_key=api_key, base_url=api_base)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Objectif de mission (langage naturel):\n"
                    f"{natural_language_prompt}\n\n"
                    "Renvoie UNIQUEMENT le JSON d'étapes."
                ),
            },
        ],
    )

    content = completion.choices[0].message.content or ""
    return _parse_steps(content.strip(), allowed)


def _write_file(output_path: Path, xml_tree: ElementTree, header: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(header, encoding="utf-8")
    with open(tmp, "ab") as f:
        xml_tree.write(f, encoding="utf-8", xml_declaration=False)
    tmp.replace(output_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Génère un BT Nav2 (XML) à partir d'un prompt (LLM→JSON→XML)."
    )
    p.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Mission en langage naturel. Si omis, lit depuis stdin.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Chemin du XML de sortie (défaut: {DEFAULT_OUTPUT}).",
    )
    p.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    p.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Affiche le XML sur stdout.")
    p.add_argument(
        "--mode",
        choices=["interactive", "fail-fast", "auto-rewrite"],
        default="interactive",
        help="Politique face aux prompts trop longs/ambigus: interactive (défaut), fail-fast, auto-rewrite.",
    )
    p.add_argument(
        "--max-prompt-chars",
        type=int,
        default=2000,
        help="Budget max (caractères) appliqué au prompt utilisateur avant appel LLM (défaut: 2000).",
    )
    p.add_argument(
        "--max-clarify-rounds",
        type=int,
        default=3,
        help="Nombre max d'itérations de clarification en mode interactive (défaut: 3).",
    )
    p.add_argument(
        "--guardrails",
        choices=["off", "heuristic", "hybrid"],
        default="hybrid",
        help="Guardrails: off, heuristic (regex/règles), hybrid (heuristic + modération optionnelle).",
    )
    p.add_argument("--moderation-api-key", type=str, default=None, help="Clé API modération (optionnel).")
    p.add_argument(
        "--moderation-api-base",
        type=str,
        default=DEFAULT_MODERATION_API_BASE,
        help="Base URL OpenAI-compatible pour modération (optionnel).",
    )
    p.add_argument(
        "--moderation-model",
        type=str,
        default=DEFAULT_MODERATION_MODEL,
        help="Nom du modèle de modération (optionnel).",
    )
    p.add_argument(
        "--validate-xml",
        action="store_true",
        help="Valide le BT XML généré via validate_bt_xml.py avant d'écrire le fichier.",
    )
    p.add_argument(
        "--strict-validate-attrs",
        action="store_true",
        help="Avec --validate-xml: attributs inconnus = erreur.",
    )
    p.add_argument(
        "--strict-validate-blackboard",
        action="store_true",
        help="Avec --validate-xml: blackboard incohérent = erreur.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.prompt:
        prompt = args.prompt.strip()
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Erreur: aucun prompt fourni (--prompt ou stdin).", file=sys.stderr)
        return 1

    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        print(
            "Erreur: aucune clé API LLM fournie (--api-key ou LLM_API_KEY).",
            file=sys.stderr,
        )
        return 1

    catalog = _load_catalog(CATALOG_PATH)

    try:
        steps = call_llm_for_steps(
            natural_language_prompt=prompt,
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            catalog=catalog,
        )
    except Exception as exc:
        print(f"Erreur appel LLM: {exc}", file=sys.stderr)
        return 1

    try:
        xml_tree = build_bt_xml(steps, catalog=catalog)
    except Exception as exc:
        print(f"Erreur génération XML: {exc}", file=sys.stderr)
        return 1

    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"<!-- Generated on {ts} -->\n"
        f"<!-- Prompt: {prompt} -->\n"
        f"<!-- Model: {args.model} -->\n"
        f"<!-- API base: {args.api_base} -->\n"
    )

    if args.dry_run:
        import io

        buf = io.BytesIO()
        buf.write(header.encode("utf-8"))
        xml_tree.write(buf, encoding="utf-8", xml_declaration=False)
        print(buf.getvalue().decode("utf-8"))
        return 0

    out = Path(args.output).resolve()
    try:
        _write_file(out, xml_tree, header)
    except Exception as exc:
        print(f"Erreur écriture fichier: {exc}", file=sys.stderr)
        return 1

    print(f"Behavior Tree généré: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


