python trainval_net.py  --dataset table --net res101 --s 1 --epochs 30 --bs 1 --nw 1  --lr 0.0001 --lr_decay_step 1000   --cuda --o adam

python trainval_accuracy_net.py  --dataset table --net res101 --test_csv hdfc_tata_bajaj.csv --epochs 100 --lr 0.001 --lr_decay_step 25 --bs 4 --cuda --pre True --checksession 1 --checkepoch 12 --checkpoint 325 --test_epochs 5
