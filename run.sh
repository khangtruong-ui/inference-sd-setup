echo "RUNNING SETUP"
sh ./setup.sh > setup.log 2>&1
echo "RUNNING TASKS"
sh ./inference.sh 2>&1 | grep -v "deprecated" > inference.log 
echo "EVERYTHING DONE"
