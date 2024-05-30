for j in {'L','R'}
do
    for i in {0..5}
    do
        python scripts/row_pose.py scripts/config.yaml row$j-$i.mp4 --out-preds results/row$j-$i.csv --out-video results/row$j-$i.mp4
    done
done