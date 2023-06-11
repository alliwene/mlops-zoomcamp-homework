while getopts s: flag
do
  case "${flag}" in
    s) script=${OPTARG};;
  esac
done

python $script \
  --wandb_project mlops-zoomcamp-experiment-tracking \
  --wandb_entity alliwene \
  --data_artifact "alliwene/mlops-zoomcamp-experiment-tracking/NYC-Taxi:v0"