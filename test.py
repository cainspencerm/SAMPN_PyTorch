import models, utils, dataset as ds  # local imports

import numpy as np
import torch
from torch.utils import data
import argparse


def main():
    parser = argparse.ArgumentParser(description='Test a model on a dataset.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument('--dataset', type=str, default='lipophilicity', help='Dataset type (lipophilicity or solubility).')
    parser.add_argument('--model', type=str, default='QSAR', help='Model type (QSAR or QSARPlus).')
    parser.add_argument('--weights', type=str, default='./checkpoint/lipophilicity/model_state_dict.pt', help='Path to the model weights.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations to run.')
    args = parser.parse_args()

    # Prepare dataset and dataloaders.
    if args.dataset == 'lipophilicity':
        data_path = './data/LogP_moleculenet.csv'  # Lipophilicity
    elif args.dataset == 'solubility':
        data_path = './data/water_solubilityOCD.csv'  # Aqueous Solubility
    else:
        raise ValueError('Invalid dataset type: {}'.format(args.dataset))

    test_set = ds.DGLDataset(data_path, split='test', shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=16, shuffle=True)

    # Initialize the metrics.
    metric_funcs = list(utils.metrics.values())

    # Create the model.
    if args.model == 'QSAR':
        model = models.QSAR()
    elif args.model == 'QSARPlus':
        model = models.QSARPlus()
    else:
        raise ValueError('Invalid model type: {}'.format(args.model))

    # Test the best model.
    model.load_state_dict(torch.load(args.weights))

    model.eval()

    scores = []
    for i in range(args.iterations):
        scores_i = np.zeros(len(metric_funcs))

        with torch.no_grad():
            for smiles, labels in test_loader:

                preds = model(smiles).squeeze()

                # Calculate metrics.
                for idx, metric_func in enumerate(metric_funcs):
                    scores_i[idx] += metric_func(preds, labels)

        scores_i /= len(test_loader)
        scores.append(scores_i)

        test_set.reshuffle()

    """
    average_scores shape: (iterations, len(metric_funcs))
             | rmse | mse | mae | r2 | pearson |
    ---------+------+-----+-----+----+---------+
    iter 1   |      |     |     |    |         |
    ---------+------+-----+-----+----+---------+
    iter 2   |      |     |     |    |         |
    ---------+------+-----+-----+----+---------+
    ...      |      |     |     |    |         |
    ---------+------+-----+-----+----+---------+
    iter n   |      |     |     |    |         |
    ---------+------+-----+-----+----+---------+
    """
    scores = np.array(scores) / args.iterations
    average_scores = np.mean(scores, axis=0)
    stddev_scores = np.std(scores, axis=0)
    for metric_name, average_score, stddev_score in zip(utils.metrics.keys(), average_scores, stddev_scores):
        print('{:<10} {:.4f} +/- {:.4f}'.format(metric_name, average_score, stddev_score))


if __name__ == '__main__':
    main()