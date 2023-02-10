ps -aux | grep train_City | awk '{print $2}' | xargs kill -9
