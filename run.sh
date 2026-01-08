echo "RUNNING SETUP"
sh ./setup.sh > setup.log 2>&1
echo "RUNNING TASKS"
sh ./inference.sh > inference.log 2>&1
echo "EVERYTHING DONE"
