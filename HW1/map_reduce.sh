hdfs dfs -rm -r /output
chmod a+x /mapper_stat.py
chmod a+x /reducer_stat.py
yarn jar /opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar \
     -file /mapper_stat.py \
     -file /reducer_stat.py \
     -mapper "./mapper_stat.py" \
     -reducer "./reducer_stat.py" \
     -input /AB_NYC_2019.csv \
     -output /output \
     -numReduceTasks 1