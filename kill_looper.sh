ps -aux | grep 'python3 loop_test.py' | awk '{print $2}' | xargs kill
ps -aux | grep 'python3 Tasker.py' | awk '{print $2}' | xargs kill