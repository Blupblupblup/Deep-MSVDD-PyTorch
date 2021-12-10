import torch

def DeepNormalityHyperspheres_BatchScores_baseline_CAE(test_loader, device, net):

    scores = []
    scores_labels = []
    net.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            reconstruction_error = torch.sum((outputs - inputs) ** 2, dim=tuple((range(1, outputs.dim()))))

            scores += reconstruction_error.cpu().tolist()
            scores_labels += labels.cpu().tolist()

    return scores, scores_labels

def DeepNormalityHyperspheres_BatchScores_baseline(test_loader, device, net, hypersphere_center):

    scores = []
    scores_labels = []
    net.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            dist_to_centers = torch.sum((outputs - hypersphere_center) ** 2, dim=1)
            try:
                scores_per_center = torch.cat([scores_per_center, dist_to_centers], dim=0)
            except UnboundLocalError:
                scores_per_center = dist_to_centers
            batch_scores = dist_to_centers

            scores += batch_scores.cpu().tolist()
            scores_labels += labels.cpu().tolist()

    return scores, scores_labels, scores_per_center

def DeepNormalityHyperspheres_BatchScores_DMSVDD(test_loader, device, net, hyperspheres_center, radius):
    scores = []
    scores_labels = []
    net.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            inputs, labels = data.to(device), targets.to(device)
            outputs = net(inputs)
            dist_to_centers = torch.sum((outputs.unsqueeze(1).repeat(1, hyperspheres_center.size()[0],1) - hyperspheres_center.unsqueeze(0).repeat(outputs.size()[0], 1, 1)) ** 2, dim=2)
            try:
                scores_per_center = torch.cat([scores_per_center, dist_to_centers], dim=0)
            except UnboundLocalError: # if scores_per_center does not exist yet, create it
                scores_per_center = dist_to_centers

            batch_scores, min_dist_idx = torch.min(dist_to_centers, dim=1)
            batch_scores -= radius[min_dist_idx] ** 2

            scores += batch_scores.cpu().tolist()
            scores_labels += labels.cpu().tolist()

    return scores, scores_labels, scores_per_center