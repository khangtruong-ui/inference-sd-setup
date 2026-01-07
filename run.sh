echo "RUNNING SETUP"
sh ./setup.sh > setup.log
echo "RUNNING TASKS"
sh ./inference.sh > inference.log
echo "EVERYTHING DONE"
