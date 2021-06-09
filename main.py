from train import  train
from helpers import *

# Creates the complete folder architecture
create_folders_structure()

#problems = ['PIMA', 'CaliforniaHousing', 'FashionMNIST']
#optimizers = ['SGD', 'Adagrad', 'Adam']

problems = ['PIMA', 'CaliforniaHousing', 'FashionMNIST']
optimizers = ['SGD', 'Adagrad', 'Adam']

for prob in problems:
    dataset, folds = load_data(prob, k_folds=10)
    print(f'Problem {prob}')
    print('--------------------------------')
    for opt in optimizers:
        print(f'Optimizer {opt}')
        print('--------------------------------')
        train(dataset, folds, prob, opt)



