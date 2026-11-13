set -eu
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s tests/resource_schedules -p 'test_*.py' -q
python3 -m unittest discover -s tests -p test_analysis_inputs.py -v
PYTHONPYCACHEPREFIX="$workspace/cache" python3 -m compileall -q emergency_resource_dispatch tests
python3 -m emergency_resource_dispatch.control.command fixtures/demo.json --database "$workspace/operations.sqlite3" > "$workspace/operations.json"
python3 -m json.tool "$workspace/operations.json"
