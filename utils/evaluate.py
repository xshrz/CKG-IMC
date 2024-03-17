import logging
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

@torch.no_grad()
def evaluate_kge_model(model, test_dataloader, fold, step, args):
    """
    Evaluate a knowledge graph embedding (KGE) model on the given test triples.

    Args:
        model: The trained KGE model.
        test_triples (torch.Tensor): Test triples containing (head, tail, label).
        fold (int): The fold index for cross-validation.
        step (int): The training step at which the evaluation is performed.
        args: Arguments containing information about the model and dataset.

    Returns:
        result (dict): Evaluation results including AUC, AUPR, F1 score, precision, recall, and accuracy.
    """
    logging.info(f'Evaluate KGE Model at step {step}')
    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    for sample,label in test_dataloader:
        sample,label = sample.to(args.device),label.flatten().detach().cpu().numpy()
        score = model(sample).flatten().detach().cpu().numpy()
        y_pred = np.concatenate([y_pred, score])
        y_true = np.concatenate([y_true, label])
    return evaluate(fold, step, y_pred, y_true)

@torch.no_grad()
def evaluate_pna_imc_model(model, test_triples, fold, epoch):
    """
    Evaluate PNA-IMC model.

    Args:
        model: The trained PNA-IMC model.
        test_triples (torch.Tensor): Test triples containing (head, tail, label).
        fold (int): The fold index for cross-validation.
        epoch (int): The epoch at which the evaluation is performed.

    Returns:
        result (dict): Evaluation results including AUC, AUPR, F1 score, precision, recall, and accuracy.
    """
    logging.info(f'Evaluate PNA-IMC model at epoch {epoch}')
    model.eval()

    # Perform model inference
    loss, CPI_reconstruct = model()
    y_pred = CPI_reconstruct[(test_triples[:, 0], test_triples[:, 1])].cpu().numpy()
    y_true = test_triples[:, 2]

    # Evaluate and return results
    result = evaluate(fold, epoch, y_pred, y_true)
    logging.info("loss: %f" % loss.item())
    return result

def evaluate(fold, epoch, y_pred, y_true):
    """
    Evaluate model predictions against ground truth labels.

    Args:
        fold (int): The fold index for cross-validation.
        epoch (int): The epoch at which the evaluation is performed.
        y_pred (numpy.ndarray): Predicted scores.
        y_true (numpy.ndarray): Ground truth labels.

    Returns:
        result (dict): Evaluation results including AUC, AUPR, F1 score, precision, recall, and accuracy.
    """
    threshold = 0.5

    # Calculate evaluation metrics
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Display and return results
    result = {
        'fold': fold + 1,
        'epoch': epoch,
        'AUC': auc,
        "AUPR": aupr,
        'F1 score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
    }
    for key, value in result.items():
        logging.info("%s:%s %s" % (key, ' ' * (10 - len(key)), value))
    return result
