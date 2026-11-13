测试：python3 -m unittest discover -s tests/resource_schedules -p 'test_*.py'
构建：python3 -m compileall -q emergency_resource_dispatch tests/resource_schedules
验收：python3 -m emergency_resource_dispatch.control.command fixtures/demo.json --database /tmp/emergency_resource_dispatch-acceptance.sqlite3
