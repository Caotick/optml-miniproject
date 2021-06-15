import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def generate_plots(nb_epochs, nb_fold, problems, optimizers):
    """
    Generates the plots after having run all data
    """

    current = os.getcwd()
    data_path = current + "/data/test_run"
    graph_path = current + "/data/graph"

    results = defaultdict(dict)

    for prob in problems:
        fig, axs = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(6, 15))

        train_losses = np.zeros((nb_fold, nb_epochs))
        val_losses = np.zeros((nb_fold, nb_epochs))
        accuracies = np.zeros((nb_fold, nb_epochs))

        for ax, optimizer in enumerate(optimizers):
            current_prob_path = data_path + f"/{prob}/{optimizer}"
            for i in range(nb_fold):
                with open(current_prob_path + f'/{i}.pkl', 'rb') as f:
                    x = pickle.load(f)
                    train_losses[i] = x['train_losses']
                    val_losses[i] = x['val_losses']
                    if prob != 'CaliforniaHousing':
                        accuracies[i] = x['accuracies']

            results[optimizer]['train_losses_mean'] = np.mean(train_losses, axis=0)
            results[optimizer]['val_losses_mean'] = np.mean(val_losses, axis=0)
            results[optimizer]['accuracies_mean'] = np.mean(accuracies, axis=0)



            results[optimizer]['train_losses_std'] = np.std(train_losses, axis=0)
            results[optimizer]['val_losses_std'] = np.std(val_losses, axis=0)
            results[optimizer]['accuracies_std'] = np.std(accuracies, axis=0)

            axs[ax].errorbar(x=range(1, nb_epochs + 1), y=results[optimizer]['train_losses_mean'],
                             yerr=results[optimizer]['train_losses_std'],
                             label='Train Loss')

            axs[ax].errorbar(x=range(1, nb_epochs + 1), y=results[optimizer]['val_losses_mean'],
                             yerr=results[optimizer]['val_losses_std'],
                             label='Val Loss')

            axs[ax].set_title(optimizer + f" ({prob})")
            axs[ax].set_ylabel('Loss')

            if (prob != 'CaliforniaHousing'):
                ax2 = axs[ax].twinx()
                color = 'tab:red'
                ax2.set_ylabel('accuracy', color=color)
                plt.errorbar(x=range(1, nb_epochs + 1),
                             y=results[optimizer]['accuracies_mean'],
                             yerr=results[optimizer]['accuracies_std'],
                             label='Accuracy', color=color, alpha=0.5)
                ax2.set_ylim([0.45, 0.95])
                ax2.tick_params(axis='y', labelcolor=color)
            if (ax == 1):
                if (prob != 'CaliforniaHousing'):
                    ax2.legend(loc='upper left')
                axs[ax].legend(loc='upper right')

            plt.savefig(graph_path + f"/{prob}.png")


def print_best_loss_and_acc(nb_epochs, nb_fold, problems, optimizers):
    current = os.getcwd()
    data_path = current + "/data/test_run"
    graph_path = current + "/data/graph"

    results = defaultdict(dict)

    for prob in problems:
        print(f"=========== Problem : {prob} ===========\n")

        train_losses = np.zeros((nb_fold, nb_epochs))
        val_losses = np.zeros((nb_fold, nb_epochs))
        accuracies = np.zeros((nb_fold, nb_epochs))

        for optimizer in optimizers:
            current_prob_path = data_path + f"/{prob}/{optimizer}"
            for i in range(nb_fold):
                with open(current_prob_path + f'/{i}.pkl', 'rb') as f:
                    x = pickle.load(f)
                    train_losses[i] = x['train_losses']
                    val_losses[i] = x['val_losses']
                    if prob != 'CaliforniaHousing':
                        accuracies[i] = x['accuracies']

            results[optimizer]['train_losses_mean'] = np.mean(train_losses, axis=0)
            results[optimizer]['val_losses_mean'] = np.mean(val_losses, axis=0)
            results[optimizer]['accuracies_mean'] = np.mean(accuracies, axis=0)


            min_val_loss_epoch = np.argmin(results[optimizer]['val_losses_mean'])

            if prob != 'CaliforniaHousing':
                print(f"====== {optimizer} ====== \n"
                      f" Best Mean Val loss : {results[optimizer]['val_losses_mean'][min_val_loss_epoch]}\n"
                      f" Reached on epoch: {min_val_loss_epoch} \n"
                      f" Mean Train loss on best val loss : {results[optimizer]['train_losses_mean'][min_val_loss_epoch]} \n"
                      f" Mean Accuracy on best val loss: {results[optimizer]['accuracies_mean'][min_val_loss_epoch]} \n")
            else:
                print(f" Best validation loss with optimizer {optimizer}: \n"
                      f" Reached on epoch: {min_val_loss_epoch} \n"
                      f" Mean Train loss on best val loss: {results[optimizer]['train_losses_mean'][min_val_loss_epoch]} \n"
                      f" Mean Val loss : {results[optimizer]['val_losses_mean'][min_val_loss_epoch]}\n")
