from utils import *
from batchscores import *
from sklearn.metrics import roc_auc_score

def get_epoch_performances_baseline_CAE(valid_loader, test_loader, device, net, normal_cls):
    scores_valid, scores_labels_valid = DeepNormalityHyperspheres_BatchScores_baseline_CAE(valid_loader, device, net)
    y_valid_ad = convert_labels(scores_labels_valid, normal_cls)
    epoch_valid_AUC = roc_auc_score(y_valid_ad, scores_valid)
    scores_test, scores_labels_test = DeepNormalityHyperspheres_BatchScores_baseline_CAE(test_loader, device, net)
    y_test_ad = convert_labels(scores_labels_test, normal_cls)
    epoch_test_AUC = roc_auc_score(y_test_ad, scores_test)
    return epoch_valid_AUC, epoch_test_AUC, scores_test, scores_labels_test

def get_epoch_performances_baseline(valid_loader, test_loader, device, net, hypersphere_center, normal_cls):
    scores_valid, scores_labels_valid, scores_per_center_valid = DeepNormalityHyperspheres_BatchScores_baseline(valid_loader, device, net, hypersphere_center)
    y_valid_ad = convert_labels(scores_labels_valid, normal_cls)
    epoch_valid_AUC = roc_auc_score(y_valid_ad, scores_valid)
    scores_test, scores_labels_test, scores_per_center_test = DeepNormalityHyperspheres_BatchScores_baseline(test_loader, device, net, hypersphere_center)
    y_test_ad = convert_labels(scores_labels_test, normal_cls)
    epoch_test_AUC = roc_auc_score(y_test_ad, scores_test)
    return epoch_valid_AUC, epoch_test_AUC, scores_test, scores_labels_test, scores_per_center_test

def get_epoch_performances_DMSVDD(valid_loader, test_loader, device, net, hyperspheres_center, radius, normal_cls):
    scores_valid, scores_labels_valid, scores_per_center_valid = DeepNormalityHyperspheres_BatchScores_DMSVDD(valid_loader, device, net, hyperspheres_center, radius)
    y_valid_ad = convert_labels(scores_labels_valid, normal_cls)
    epoch_valid_AUC = roc_auc_score(y_valid_ad, scores_valid)
    scores_test, scores_labels_test, scores_per_center_test = DeepNormalityHyperspheres_BatchScores_DMSVDD(test_loader, device, net, hyperspheres_center, radius)
    y_test_ad = convert_labels(scores_labels_test, normal_cls)
    epoch_test_AUC = roc_auc_score(y_test_ad, scores_test)
    return epoch_valid_AUC, epoch_test_AUC, scores_test, scores_labels_test, scores_per_center_test