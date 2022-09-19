# Experiments

## SIR example
### Training 
```commandline
python train_SIR.py --device cpu --AIS True --bound_surjection True --num_flows 20 --model_num 100 --iteration 126 --temp 2000 --eval_every 25 --lr 4e-4
python train_SIR.py --device cpu --AIS False --bound_surjection False --num_flows 20 --model_num 100 --iteration 126 --temp 2000 --eval_every 25 --lr 4e-4
```

### Evaluating
To evaluate ATVI and VI models, 
```commandline
python Evaluate.py --dataset SIR --setting BTAT --resume False --num_flows 20 --device cpu 
python Evaluate.py --dataset SIR --setting BFAF --bound_surjection False --AIS False  --resume False --num_flows 20 --device cpu 
```
To evaluate ABC samples, 
```commandline
python Evaluate_ABC_SIR.py 
```

## SEIR example
### Training 
To train SEIR example, 
```commandline
 python train_SEIR.py --model_num 1 --lr 8e-4  --batch_size 8 --eval_every 25 --dataset SEIR --smoothing 1400 --bound_surjection True --AIS True --iteration 451 --temp 10 --smoothing2 300 --eval_size 1024 --num_flows 20
 python train_SEIR.py --model_num 1 --lr 8e-4  --batch_size 8 --eval_every 25 --dataset SEIR --smoothing 1400 --bound_surjection False --AIS False --iteration 451 --smoothing2 300 --eval_size 1024 --num_flows 20
```

### Evaluating
To evaluate ATVI and VI models, 
```commandline
python Evaluate.py --dataset SEIR --setting BTAT --resume False --num_flows 20 
python Evaluate.py --dataset SEIR --setting BFAF --resume False --num_flows 20 --bound_surjection False 
```
To evaluate ABC-SMC-MNN samples, 
```commandline
python Evaluate_ABC_SEIR.py 
```

## MSIR example
### Training 
To train MSIR example, 
```commandline
python train.py --eval_every 25 --lr 6e-3 --iteration 401 --num_flows 20 --bound_surjection True --AIS True --resume False --model_num 50 --batch_size 8
python train.py --eval_every 50 --lr 6e-3 --iteration 301 --num_flows 20 --bound_surjection False --AIS False --resume False --model_num 50 --batch_size 16
```

### Evaluating
To evaluate ATVI and VI models, 
```commandline
python Evaluate.py --dataset MSIR --setting BTAT --resume False 
python Evaluate.py --dataset MSIR --setting BFAF --resume False --bound_surjection False --AIS False --num_flows 20 --eval_size 500

```
To evaulate MCMC smaples, 
```commandline
python Evaluate_MCMC.py 
```

