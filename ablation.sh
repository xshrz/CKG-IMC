python main.py --save_path ./experiments/without_kg -adv --device cuda:1
python main.py --save_path ./experiments/without_pna -adv --device cuda:0 --pna_imc_warm_up_epochs 1000
python main.py --save_path ./experiments/without_imc -adv --device cuda:1
