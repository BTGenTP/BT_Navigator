#!/usr/bin/env bash
set -eo pipefail

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*"; }
err() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }

AUTO_MODE=0
PROMPT=""
NO_RVIZ=0
SEND_GOAL=0
VALIDATE_BT=1
STRICT_VALIDATE_ATTRS=0
INITIAL_POSE="0.0,0.0,0.0" # x,y,yaw(rad) in map
GOAL_POSE="0.0,0.0,0.0"    # x,y,yaw(rad) in map

while [ "$#" -gt 0 ]; do
  case "$1" in
    --auto) AUTO_MODE=1; shift ;;
    --no-rviz) NO_RVIZ=1; shift ;;
    --send-goal) SEND_GOAL=1; shift ;;
    --validate-bt) VALIDATE_BT=1; shift ;;
    --no-validate-bt) VALIDATE_BT=0; shift ;;
    --strict-validate-attrs) STRICT_VALIDATE_ATTRS=1; shift ;;
    --initial-pose) INITIAL_POSE="${2:-}"; shift 2 ;;
    --goal-pose) GOAL_POSE="${2:-}"; shift 2 ;;
    *) PROMPT="$*"; break ;;
  esac
done

if [ -z "$PROMPT" ]; then
  read -rp "Description de la mission (prompt LLM) : " PROMPT
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BT_NAV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ROS_MAIN_WS="${ROS_MAIN_WS:-$HOME/ros2_ws}"
MAP_FILE="${MAP_FILE:-$BT_NAV_DIR/maps/exploration_map.yaml}"
NAV2_PARAMS_SRC="${NAV2_PARAMS_SRC:-$BT_NAV_DIR/params/nav2_params.yaml}"
GENERATED_BT_XML="${GENERATED_BT_XML:-$BT_NAV_DIR/behavior_trees/__generated/turtlebot_mission.xml}"

TMP_PARAMS="/tmp/nav2_params_generated_bt.yaml"

if [ "$NO_RVIZ" -eq 1 ] && [ "$SEND_GOAL" -eq 0 ]; then
  # Sans RViz, personne ne publie l'initial pose ni n'envoie de goal :
  # on active donc le mode automatique d'exécution de mission.
  SEND_GOAL=1
fi

cleanup() {
  if [ -n "${PIDS:-}" ]; then
    log "Arrêt des processes: $PIDS"
    # shellcheck disable=SC2086
    kill $PIDS >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

yaw_to_quat() {
  # args: yaw(rad) -> prints: "<qz> <qw>"
  python3 - <<'PY'
import math, sys
yaw = float(sys.argv[1])
qz = math.sin(yaw/2.0)
qw = math.cos(yaw/2.0)
print(f"{qz} {qw}")
PY
}

publish_initial_pose() {
  local pose_csv="$1"
  IFS=',' read -r IX IY IYAW <<< "$pose_csv"
  if [ -z "$IX" ] || [ -z "$IY" ] || [ -z "$IYAW" ]; then
    err "--initial-pose invalide. Attendu: \"x,y,yaw\" (ex: \"0.0,0.0,0.0\")"
    return 1
  fi
  read -r IQZ IQW <<< "$(python3 -c "import math; yaw=float('$IYAW'); print(math.sin(yaw/2.0), math.cos(yaw/2.0))")"
  log "Publication /initialpose (map): x=$IX y=$IY yaw=$IYAW"
  ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "{header: {frame_id: map}, pose: {pose: {position: {x: $IX, y: $IY, z: 0.0}, orientation: {z: $IQZ, w: $IQW}}, covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0685]}}"
}

send_nav_goal() {
  local pose_csv="$1"
  IFS=',' read -r GX GY GYAW <<< "$pose_csv"
  if [ -z "$GX" ] || [ -z "$GY" ] || [ -z "$GYAW" ]; then
    err "--goal-pose invalide. Attendu: \"x,y,yaw\" (ex: \"0.0,0.0,0.0\")"
    return 1
  fi
  read -r GQZ GQW <<< "$(python3 -c "import math; yaw=float('$GYAW'); print(math.sin(yaw/2.0), math.cos(yaw/2.0))")"
  log "Envoi d'un goal Nav2 (/navigate_to_pose) pour déclencher l'exécution du BT..."
  ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    "{pose: {header: {frame_id: map}, pose: {position: {x: $GX, y: $GY, z: 0.0}, orientation: {z: $GQZ, w: $GQW}}}}" &
  PIDS="$PIDS $!"
}

log "Sourcing ROS 2 Humble..."
source /opt/ros/humble/setup.bash

if [ -f "$ROS_MAIN_WS/install/setup.bash" ]; then
  log "Sourcing workspace principal: $ROS_MAIN_WS"
  # shellcheck disable=SC1091
  source "$ROS_MAIN_WS/install/setup.bash"
else
  log "Workspace principal non trouvé (install/setup.bash absent): $ROS_MAIN_WS"
fi

if [ ! -f "$NAV2_PARAMS_SRC" ]; then
  err "Params Nav2 introuvables: $NAV2_PARAMS_SRC"
  exit 1
fi

if [ ! -f "$MAP_FILE" ]; then
  err "Map introuvable: $MAP_FILE"
  exit 1
fi

log "Génération du Behavior Tree via LLM → $GENERATED_BT_XML"
python3 "$SCRIPT_DIR/gen_turtlebot_mission_bt.py" --prompt "$PROMPT" --output "$GENERATED_BT_XML"

log "Aperçu du BT généré:"
echo "------------------------------------------------------------"
cat "$GENERATED_BT_XML"
echo
echo "------------------------------------------------------------"

if [ "$VALIDATE_BT" -eq 1 ]; then
  VALIDATION_REPORT="/tmp/validate_bt_report.json"
  log "Validation statique du BT (pré-simulation) → $VALIDATION_REPORT"
  VALIDATE_ARGS=(--output "$VALIDATION_REPORT")
  if [ "$STRICT_VALIDATE_ATTRS" -eq 1 ]; then
    VALIDATE_ARGS+=(--strict-attrs)
  fi
  if ! python3 "$SCRIPT_DIR/validate_bt_xml.py" "$GENERATED_BT_XML" "${VALIDATE_ARGS[@]}"; then
    err "BT invalide: simulation annulée. Voir: $VALIDATION_REPORT"
    exit 1
  fi
  log "BT validé ✔️"
else
  log "--no-validate-bt: validation ignorée."
fi

if [ "$AUTO_MODE" -eq 0 ]; then
  while true; do
    read -r -p "Valider ce Behavior Tree pour Nav2 et lancer Gazebo/Nav2 ? [O/n] " REPLY
    ANSWER="${REPLY,,}"
    case "$ANSWER" in
      ""|"o"|"oui"|"y"|"yes") break ;;
      "n"|"non"|"no") log "Annulé."; exit 0 ;;
      *) echo "Réponse non reconnue. Merci de répondre par Oui ou Non." ;;
    esac
  done
else
  log "Mode automatique (--auto): validation ignorée."
fi

log "Création d'un params_file Nav2 temporaire avec default_nav_to_pose_bt_xml=$GENERATED_BT_XML"
cp "$NAV2_PARAMS_SRC" "$TMP_PARAMS"

# Remplace (ou ajoute si absent) le paramètre bt_navigator.default_nav_to_pose_bt_xml
if grep -qE "^[[:space:]]*default_nav_to_pose_bt_xml:" "$TMP_PARAMS"; then
  # note: | comme séparateur pour faciliter les chemins
  sed -i "s|^[[:space:]]*default_nav_to_pose_bt_xml:.*$|    default_nav_to_pose_bt_xml: $GENERATED_BT_XML|g" "$TMP_PARAMS"
else
  err "Paramètre default_nav_to_pose_bt_xml non trouvé dans $TMP_PARAMS (format inattendu)."
  exit 1
fi

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
log "TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL"

log "Lancement Gazebo (turtlebot3_world)..."
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
PIDS="$!"
sleep 2

log "Lancement Nav2 localization..."
ros2 launch nav2_bringup localization_launch.py use_sim_time:=True map:="$MAP_FILE" &
PIDS="$PIDS $!"
sleep 3

# AMCL a besoin d'une pose initiale pour publier map->odom.
publish_initial_pose "$INITIAL_POSE" || exit 1
sleep 2

log "Lancement Nav2 navigation (params_file=$TMP_PARAMS)..."
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True params_file:="$TMP_PARAMS" &
PIDS="$PIDS $!"

if [ "$SEND_GOAL" -eq 1 ]; then
  # Le BT "default_nav_to_pose_bt_xml" n'est exécuté que lorsqu'un goal NavigateToPose est envoyé.
  sleep 3
  send_nav_goal "$GOAL_POSE" || exit 1
fi

if [ "$NO_RVIZ" -eq 0 ]; then
  log "Lancement RViz..."
  ros2 launch nav2_bringup rviz_launch.py
else
  log "--no-rviz: RViz non lancé. Attente (Ctrl+C pour quitter)."
  wait
fi


