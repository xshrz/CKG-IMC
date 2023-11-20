# python main.py --save_path ./experiments/change_layers --pna_n_layer 1 --device cuda:1 
# python main.py --save_path ./experiments/change_layers --pna_n_layer 2 --device cuda:1 
# python main.py --save_path ./experiments/change_layers --pna_n_layer 3 --device cuda:1 
# python main.py --save_path ./experiments/change_layers --pna_n_layer 4 --device cuda:1 
# python main.py --save_path ./experiments/change_layers --pna_n_layer 5 --device cuda:1 
# python main.py --save_path ./experiments/change_layers --pna_n_layer 6 --device cuda:1 
# python main.py --save_path ./experiments/change_layers --pna_n_layer 7 --device cuda:1 
# python main.py --save_path ./experiments/all_fold -adv --device cuda:0
# python main.py --save_path ./experiments/without_pna -adv --device cuda:0
# python main.py --save_path ./experiments/test -adv --device cuda:0
# python main.py --save_path ./experiments/test -adv --device cuda:0 --ccs_threshold 0.7 --pps_threshold 0.5 --kg_hidden_dim 2048 --autoencoder_hidden_dim 2048
# python main.py --save_path ./experiments/test1 -adv --device cuda:1 --ccs_threshold 0.6 --pps_threshold 0.5 --kg_hidden_dim 2048 --autoencoder_hidden_dim 1500
# python main.py --save_path ./experiments/test1 -adv --device cuda:1 --ccs_threshold 0 --pps_threshold 0 --kg_hidden_dim 2048 --autoencoder_hidden_dim 1500 --pna_hidden_dim 25 --pna_edge_dim 25
# python main.py --save_path ./experiments/test -adv --device cuda:0 --pna_imc_warm_up_epochs 200 --pna_imc_patience 3 --ccs_threshold 0.4 --pps_threshold 0.3 
# python main.py --save_path ./experiments/without_se -adv --device cuda:1 --pna_imc_warm_up_epochs 200 --pna_imc_patience 40 --pna_imc_max_epochs 4000
python main.py --save_path ./experiments/kg_sample --kg_negative_sample_size 100 -adv --device cuda:1 --pna_imc_warm_up_epochs 200 --pna_imc_patience 40 --pna_imc_max_epochs 4000
python main.py --save_path ./experiments/kg_sample --kg_negative_sample_size 10 -adv --device cuda:0 --pna_imc_warm_up_epochs 200 --pna_imc_patience 40 --pna_imc_max_epochs 4000

python main.py --save_path ./experiments/kg_sample --kg_negative_sample_size 10 -adv --device cuda:0 --ccs_threshold 0.65 --pps_threshold 0.45 # kg 0.88 aupr 0.89765

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_imc_warm_up_epochs 200 --pna_imc_patience 5 --ccs_threshold 0.5 --pps_threshold 0.45 # aupr 0.897

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.7  --pps_threshold 0.5 # kg aupr 0.881 aupr 0.8952

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.65 --pps_threshold 0.45 --pna_ccs_threshold 0.6 --pna_pps_threshold 0.4 --kg_load_embedding # kg aupr  aupr 0.894

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.65  --pps_threshold 0.45 --pna_ccs_threshold 0.7 --pna_pps_threshold 0.5 --kg_load_embedding  # kg aupr 0. aupr 0.896

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.65  --pps_threshold 0.40 --pna_ccs_threshold 0.6 --pna_pps_threshold 0.4  # kg aupr 0.880 aupr 0.895

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.65  --pps_threshold 0.48 --pna_ccs_threshold 0.65 --pna_pps_threshold 0.48 --kg_load_embedding --pna_imc_warm_up_epochs 600 # kg aupr 0.8849 aupr 0.894

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.65  --pps_threshold 0.48 --pna_ccs_threshold 0.65 --pna_pps_threshold 0.45 --kg_load_embedding --pna_imc_warm_up_epochs 600 # kg aupr 0.8849 aupr 0.898

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.63  --pps_threshold 0.45 --pna_ccs_threshold 0.63 --pna_pps_threshold 0.45  # kg aupr 0.8809 aupr 0.896

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.48 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.48  # kg aupr 0.882 aupr 0.897

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.65  --pps_threshold 0.48 --pna_ccs_threshold 0.65 --pna_pps_threshold 0.45 --kg_load_embedding --pna_imc_warm_up_epochs 600 # kg cat kg aupr 0.8 aupr 0.8995

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.48 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.48 --kg_load_embedding --pna_imc_warm_up_epochs 600  # cat kg aupr 0.882 aupr 0.8980

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.65  --pps_threshold 0.48 --pna_ccs_threshold 0.62 --pna_pps_threshold 0.45 --kg_load_embedding --pna_imc_warm_up_epochs 600 # kg cat kg aupr 0.8 aupr 0.8970

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.65  --pps_threshold 0.48 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45 --kg_load_embedding --pna_imc_warm_up_epochs 600 # kg cat kg aupr 0.8 aupr 0.8995

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47  --pna_imc_warm_up_epochs 600  # cat kg aupr 0.879 aupr 0.900

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.65  --pps_threshold 0.48 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_load_embedding --pna_imc_warm_up_epochs 600 # kg cat kg aupr 0.8 aupr 0.8999

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.46 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.46  --pna_imc_warm_up_epochs 600  # cat kg aupr 0.879 aupr 0.898

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_load_embedding --pna_imc_warm_up_epochs 600  # no cat kg aupr 0.879 aupr 0.892

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.67  --pps_threshold 0.47 --pna_ccs_threshold 0.67 --pna_pps_threshold 0.47  --pna_imc_warm_up_epochs 600  # cat kg aupr 0.880 aupr 0.8970

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_hidden_dim 768 --pna_imc_warm_up_epochs 600  # cat kg aupr 0.884 aupr 0.8995

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_hidden_dim 768 --kg_load_embedding --pna_imc_warm_up_epochs 500  # cat kg aupr 0.884 aupr 0.89

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_hidden_dim 1400 --pna_imc_warm_up_epochs 500  # cat kg aupr 0.877 aupr 0.877

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47  --kg_load_embedding --pna_imc_warm_up_epochs 600  # cat normalzie activation kg aupr 0.879 aupr 0.8959

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_hidden_dim 2048 --pna_imc_warm_up_epochs 500  # cat normalzie activation kg aupr 0.877 aupr 0.894

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47  --kg_load_embedding --pna_imc_warm_up_epochs 600 --use_autoencoder  # cat kg aupr 0.879 aupr 0.900

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47  --kg_load_embedding --pna_imc_warm_up_epochs 600 --use_autoencoder --autoencoder_protein_hidden_dim 512 # cat kg aupr 0.879 aupr 0.898

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_hidden_dim 1200 --pna_imc_warm_up_epochs 500  # cat  kg aupr 0.875 aupr 0.9002

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47  --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat add kg aupr 0.879 aupr 0.8996

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47  --kg_load_embedding --pna_imc_warm_up_epochs 400 --use_autoencoder --autoencoder_train_epochs 50  # cat kg aupr 0.879 aupr 0.8990

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --ccs_threshold 0.68  --pps_threshold 0.47 --pna_ccs_threshold 0.68 --pna_pps_threshold 0.47 --kg_hidden_dim 1280 --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --kg_load_embedding --pna_imc_warm_up_epochs 400 --use_autoencoder --autoencoder_train_epochs 50 --autoencoder_compound_hidden_dim 768 --autoencoder_protein_hidden_dim 768 # cat kg aupr 0.879 aupr 0.8989

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.7 --pna_pps_threshold 0.5 --kg_hidden_dim 1024 --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.898

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --kg_load_embedding --use_autoencoder --autoencoder_train_epochs 50 --autoencoder_hidden_dim 768 # cat kg aupr 0.879 aupr 0.8998

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.65 --pna_pps_threshold 0.43 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9007

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.65 --pna_pps_threshold 0.43 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.8995

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.65 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9011

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.63 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9010

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9013
python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.44 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9008
python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.46 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9006
python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400 --use_autoencoder --autoencoder_train_epochs 10  # cat  kg aupr 0.875 aupr 0.90

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat without activation kg aupr 0.875 aupr 0.898

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400 --pna_n_layer 2 # cat layer2 kg aupr 0.875 aupr 0.891

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat batch norm kg aupr 0.875 aupr 0.899

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --kg_load_embedding --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45  # cat bn kg aupr 0.879 aupr 0.894

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --kg_load_embedding --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45  # cat normalize kg aupr 0.879 aupr 0.9008

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45 --kg_hidden_dim 1024 --kg_load_embedding --pna_imc_warm_up_epochs 400  # cat normalize non kg aupr 0.875 aupr 0.8

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --kg_load_embedding --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45  # cat normalize non kg aupr 0.879 aupr 0.893

python main.py --save_path ./experiments/kg_sample -adv --device cuda:0 --kg_load_embedding --pna_ccs_threshold 0.64 --pna_pps_threshold 0.45  # cat act kg aupr 0.879 aupr 0.893

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9013

python main.py --save_path ./experiments/sample2 -adv --device cuda:1 --pna_imc_warm_up_epochs 400  # cat  kg aupr 0.875 aupr 0.9013

python main.py --save_path ./experiments/test -adv --device cuda:0 --pna_hidden_dim 1024 
python main.py --save_path ./experiments/test -adv --device cuda:0 --pna_hidden_dim 1024 --pna_edge_dim 1024
python main.py --save_path ./experiments/test -adv --device cuda:0 --pna_hidden_dim 1500 --pna_edge_dim 1024
python main.py --save_path ./experiments/test -adv --device cuda:0 --pna_hidden_dim 1500
python main.py --save_path ./experiments/no_app_tau -adv --device cuda:1 --do_predict

python main.py --save_path ./experiments/pred -adv --device cuda:0 --do_predict

python main.py --save_path ./experiments/top50 -adv --device cuda:0 --do_predict

python main.py --save_path ./experiments/without_imc_right -adv --device cuda:1
