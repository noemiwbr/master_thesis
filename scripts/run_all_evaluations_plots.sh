#!/bin/bash

datasets=("nowcast" "forecast")
splits=("train" "validation" "test")
models=("BNN" "QRNN" "median" "shifted")
zones=("NO1" "NO2" "NO3" "NO4")

for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    for model in "${models[@]}"; do
      for zone in "${zones[@]}"; do
        if [[ "$dataset" == "nowcast" && "$split" == "test" ]]; then
            continue
        fi
        if [[ "$dataset" == "forecast" && "$split" == "train" ]]; then
            continue
        fi

        echo "Running: $dataset | $split | $model | $zone"

        python run_evaluation.py \
          --dataset "$dataset" \
          --split "$split" \
          --model "$model" \
          --bidding_zone "$zone"

        python run_plots.py \
          --dataset "$dataset" \
          --split "$split" \
          --model "$model" \
          --bidding_zone "$zone"

      done
    done
  done
done