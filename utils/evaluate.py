import logging
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

@torch.no_grad()
def evaluate_kge_model(model, test_triples, fold, step, args):
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
    logging.info(f'evaluate_kge_model at step {step}')
    model.eval()

    # Extract embeddings for head, relation, and tail
    head = model.entity_embedding[test_triples[:, 0]]
    relation = model.relation_embedding[0].repeat(head.size(0), 1)
    tail = model.entity_embedding[test_triples[:, 1] + args.n_compound]

    # Split complex embeddings into real and imaginary parts
    re_head, im_head = torch.chunk(head, 2, dim=1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=1)

    # Calculate the score for each triple
    re_score = re_relation * re_tail + im_relation * im_tail
    im_score = re_relation * im_tail - im_relation * re_tail
    score = re_head * re_score + im_head * im_score
    score = score.sum(dim=1)
    y_pred = score.cpu().numpy()

    # Evaluate and return results
    return evaluate(fold, step, y_pred, test_triples[:, 2])

@torch.no_grad()
def evaluate_pna_imc_model(model, test_triples, fold, epoch):
    """
    Evaluate a Predictive Network Alignment with Implicit Multi-channel (PNA-IMC) model.

    Args:
        model: The trained PNA-IMC model.
        test_triples (torch.Tensor): Test triples containing (head, tail, label).
        fold (int): The fold index for cross-validation.
        epoch (int): The epoch at which the evaluation is performed.

    Returns:
        result (dict): Evaluation results including AUC, AUPR, F1 score, precision, recall, and accuracy.
    """
    logging.info(f'evaluate_pna_imc_model at epoch {epoch}')
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
