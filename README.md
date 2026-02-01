### BT_Navigator

Ce dossier contient un pipeline **LLM → JSON strict → Behavior Tree XML Nav2** et un script de lancement **Gazebo TurtleBot3 + Nav2** qui exécute le Behavior Tree généré.

L’objectif est de recevoir une mission en langage naturel, de la convertir via un LLM (Mistral) en étapes structurées **contraintes**, puis de garantir que le XML produit est **compatible** avec les nœuds BT disponibles côté Nav2.

---

### Structure du dossier

- `behavior_trees/`: exemples de BT Nav2 (XML)
- `behavior_trees/__generated/`: **sortie** des BT générés (XML)
- `params/nav2_params.yaml`: paramètres Nav2 (dont `bt_navigator.plugin_lib_names`)
- `maps/`: maps de simulation
- `script/`:
  - `bt_nodes_catalog.json`: **catalogue** des compétences/nœuds autorisés (ports + description)
  - `gen_turtlebot_mission_bt.py`: générateur **prompt → LLM → JSON → XML**
  - `validate_bt_xml.py`: validateur statique **post-génération** (XML/structure/ports/blackboard)
  - `run_turtlebot3_nav2_generated_bt.sh`: pipeline Gazebo + Nav2 utilisant le BT généré

---

### Pré‑requis ROS / Navigation

Environnement cible: **ROS 2 Humble** (Linux/WSL2 recommandé).

À installer/configurer:
- **ROS 2 Humble**
- **Nav2** (`nav2_bringup`, `nav2_bt_navigator`, etc.)
- **TurtleBot3**:
  - `turtlebot3_gazebo`
  - modèles TB3 (ex: `burger`)
- Un workspace ROS 2 “principal” (par défaut `ROS_MAIN_WS=$HOME/ros2_ws`) déjà compilé avec les dépendances.

Le script `BT_Navigator/script/run_turtlebot3_nav2_generated_bt.sh` source:
- `/opt/ros/humble/setup.bash`
- puis optionnellement `$ROS_MAIN_WS/install/setup.bash`

---

### Pré‑requis LLM (Mistral)

Le générateur (`gen_turtlebot_mission_bt.py`) appelle un endpoint **OpenAI-compatible** via le client Python `openai`.

À prévoir:
- une clé API: `LLM_API_KEY`
- un modèle: `LLM_MODEL` (défaut: `mistral-large-latest`)
- une base URL: `LLM_API_BASE` (défaut: `https://api.mistral.ai/v1`)

Dépendances Python:
- `openai` (obligatoire)
- `python-dotenv` (optionnel, pour charger un fichier `.env`)

---

### Concept: “JSON contraint” puis XML garanti

Le LLM n’a pas le droit de produire du XML.

Il doit produire **uniquement** une liste JSON d’étapes de la forme:

```json
[
  { "skill": "Wait", "params": { "wait_duration": 2.0 }, "comment": "Stabilisation" },
  { "skill": "Spin", "params": { "spin_dist": 1.57 }, "comment": "Tourner ~90°" }
]
```

Le script:
- refuse tout `skill` non listé dans `script/bt_nodes_catalog.json`
- vérifie les ports requis (`optional` dans le catalogue = non requis)
- génère ensuite un XML Nav2 lisible et compatible

---

### Générer un Behavior Tree (XML)

Depuis la racine du repo:

```bash
python3 BT_Navigator/script/gen_turtlebot_mission_bt.py --prompt "Attends 2s, tourne de 90°, attends 1s"
```

Sortie par défaut (timestampée):
- `BT_Navigator/behavior_trees/__generated/turtlebot_mission_generated_YYYYMMDDhhmmss.xml`

Options utiles:
- `--output <path>`: choisir le chemin de sortie
- `--dry-run`: affiche sur stdout sans écrire de fichier
- `--model`, `--api-base`, `--api-key`: override des paramètres LLM

---

### Lancer la simulation Gazebo + Nav2 + exécution du BT généré

Script principal:
- `BT_Navigator/script/run_turtlebot3_nav2_generated_bt.sh`

Exemple:

```bash
bash BT_Navigator/script/run_turtlebot3_nav2_generated_bt.sh "Attends 2s, tourne de 90°"
```

Ce script:
- génère un BT XML via `gen_turtlebot_mission_bt.py`
- crée un `params_file` temporaire (`/tmp/nav2_params_generated_bt.yaml`) où:
  - `bt_navigator.default_nav_to_pose_bt_xml` pointe vers le XML généré
- lance ensuite:
  - Gazebo TB3 world
  - `nav2_bringup localization_launch.py`
  - publie `/initialpose` (AMCL)
  - `nav2_bringup navigation_launch.py`
  - (optionnel) RViz

#### Options du script de lancement

- `--auto`: ignore la confirmation “Valider le BT ?”
- `--no-rviz`: ne lance pas RViz
- `--send-goal`: envoie automatiquement un goal Nav2 (déclenche l’exécution du BT)
- `--validate-bt`: valide le BT XML avant simulation (par défaut: activé)
- `--no-validate-bt`: désactive la validation statique avant simulation
- `--strict-validate-attrs`: mode strict (attributs inconnus = erreur)
- `--initial-pose "x,y,yaw"`: pose initiale AMCL (map), ex: `"0.0,0.0,0.0"`
- `--goal-pose "x,y,yaw"`: goal `/navigate_to_pose`, ex: `"2.0,1.0,0.0"`

Note:
- Sans RViz, le script active automatiquement `SEND_GOAL=1` (sinon la mission ne démarre pas).

#### Variables d’environnement utiles

- `ROS_MAIN_WS`: workspace ROS principal (défaut: `$HOME/ros2_ws`)
- `MAP_FILE`: map utilisée (défaut: `BT_Navigator/maps/exploration_map.yaml`)
- `NAV2_PARAMS_SRC`: yaml nav2 source (défaut: `BT_Navigator/params/nav2_params.yaml`)
- `GENERATED_BT_XML`: chemin du XML de mission (défaut: `BT_Navigator/behavior_trees/__generated/turtlebot_mission.xml`)

---

### Flux complet (du prompt à l’exécution dans Gazebo)

- **1) Prompt utilisateur** (langage naturel)  
  Exemple: “Aller à un goal, attendre, tourner…”

- **2) Appel LLM (Mistral)**  
  Le système prompt force une sortie **JSON uniquement** et limite `skill` à la liste du catalogue.

- **3) Validation JSON**  
  Le script vérifie:
  - structure (liste d’objets)
  - `skill` autorisés
  - paramètres requis présents

- **4) Génération XML Nav2**  
  Le script produit un XML `BehaviorTree.CPP` composé uniquement de tags connus (ex: `Wait`, `Spin`, `SubTree NavigateToPoseWithReplanningAndRecovery`, etc.).

- **5) Lancement Nav2**  
  Le launcher remappe `bt_navigator.default_nav_to_pose_bt_xml` vers le BT généré via un params yaml temporaire.

- **6) Déclenchement mission**  
  Le BT “default_nav_to_pose” s’exécute quand un goal `NavigateToPose` est envoyé:
  - via RViz (Nav2 Goal), ou
  - via `--send-goal` (script)

---

### Troubleshooting rapide

- **Le robot ne bouge pas**:
  - vérifier qu’une pose initiale AMCL a bien été publiée (`/initialpose`)
  - vérifier qu’un goal a bien été envoyé (RViz ou `--send-goal`)

- **Erreur “Réponse LLM non JSON”**:
  - le LLM n’a pas respecté la contrainte; réduire la complexité du prompt et relancer

- **Skill non autorisé**:
  - ajouter la compétence dans `script/bt_nodes_catalog.json` + implémenter sa traduction XML si nécessaire

---

### Validation post-génération (recommandé avant simulation)

Le script de lancement exécute désormais une validation statique du BT avant Gazebo/Nav2.

Validation manuelle:

```bash
python3 BT_Navigator/script/validate_bt_xml.py BT_Navigator/behavior_trees/__generated/turtlebot_mission.xml
```

Un exemple volontairement invalide est fourni pour déclencher des issues:
- `BT_Navigator/behavior_trees/__generated/invalid_bt_for_validator_demo.xml`


