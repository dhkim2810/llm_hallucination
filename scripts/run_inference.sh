for model in phi3 llama3
do
	for dataset in sciq scienceqa
	do
		echo "Running inference for model: $model, dataset: $dataset"
		python inference.py --model_name $model --dataset $dataset --batch_size 64
		python inference.py --model_name $model --dataset $dataset --batch_size 64 --use_hint
	done
done