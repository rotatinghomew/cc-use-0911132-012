set -eu
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
export PYTHONDONTWRITEBYTECODE=1
bash scripts/demo.sh
python3 -m unittest discover -s tests -p test_demand_planning.py -q
python3 -m emergency_resource_dispatch.control.command fixtures/demo.json --database "$workspace/analysis.sqlite3" > "$workspace/baseline.json"
python3 -m emergency_resource_dispatch.demand_planning --input fixtures/analysis/input.json --database "$workspace/analysis.sqlite3" --output "$workspace/first.json"
python3 -m emergency_resource_dispatch.demand_planning --input fixtures/analysis/input.json --database "$workspace/analysis.sqlite3" --output "$workspace/replay.json"
python3 scripts/check_analysis.py "$workspace/first.json" "$workspace/replay.json"
python3 -m emergency_resource_dispatch.demand_planning --input fixtures/analysis/capacity_drop_illustration.json --database "$workspace/illustration.sqlite3" --output "$workspace/illustration.json"
python3 -m json.tool "$workspace/illustration.json" > /dev/null
python3 -m json.tool "$workspace/first.json"
