from postprocess import generate_plots
from train import  train
from postprocess import *
from helpers import *

# Creates the complete folder architecture
create_folders_structure()

#problems = ['PIMA', 'CaliforniaHousing', 'FashionMNIST']
#optimizers = ['SGD', 'Adagrad', 'Adam']

problems = ['PIMA', 'CaliforniaHousing', 'FashionMNIST']
optimizers = ['SGD', 'Adagrad', 'Adam']

for prob in problems:
    print(f'Problem {prob}')
    print('--------------------------------')
    for opt in optimizers:
        dataset, folds = load_data(prob, k_folds=10) ### Have to shift this here because folds don't run otherwise
        print(f'Optimizer {opt}')
        print('--------------------------------')
        train(dataset, folds, prob, opt)

