import torch
import utils.solvers
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F

def sample_minimal_sets(opt, log_probs):

    probs = torch.softmax(log_probs, dim=1) 
    probs = probs.squeeze(-1)
    B, N, Mo = probs.size()
    M = opt.instances #24

    choice_weights = probs[..., :M].view(B, 1, 1, N, M).expand(B, opt.samplecount, opt.hypotheses, N, M) 
    choice_weights = choice_weights.transpose(-1, -2).contiguous().view(-1, N) 
    
    choice_batched = torch.multinomial(choice_weights, opt.mss, replacement=True)
    
    choices = choice_batched.view(B, opt.samplecount, opt.hypotheses, M, opt.mss) 

    return choices 

def generate_hypotheses(opt, X, choices):
    
    B, N, C = X.size()  
    _, K, S, M, mss = choices.size() 

    X_e = X.view(B, 1, 1, 1, N, C).expand(B, K, S, M, N, C) 
    choices_e = choices.view(B, K, S, M, mss, 1).expand(B, K, S, M, mss, C) 

    X_samples = torch.gather(X_e, -2, choices_e) 
 
    hypotheses = utils.solvers.minimal_solver[opt.problem](X_samples) 

    return hypotheses 


def sample_hypotheses(opt, mode, hypotheses, weighted_inlier_counts, inlier_scores, residuals):
    B, K, S, M = weighted_inlier_counts.size() 

    softmax_input = opt.softmax_alpha * weighted_inlier_counts 
    
    hyp_selection_weights = torch.softmax(softmax_input, dim = 2) 
    log_p_h_S = torch.nn.functional.log_softmax(softmax_input, dim=2)  

    choice_weights = hyp_selection_weights.transpose(-1, -2).contiguous().view(-1, S) 

    if opt.hypsamples > 0 and mode == "train":
        choice_batched = torch.multinomial(choice_weights, opt.hypsamples, replacement=True) 
        choices = choice_batched.view(B, K, M, opt.hypsamples)
        H = opt.hypsamples 
    else:
        choice_batched = torch.argmax(choice_weights, dim=-1) 
        choices = choice_batched.view(B, K, M, 1)
        H = 1

    hyp_choices_e = choices.view(B, K, 1, M, H) 
    log_p_e = log_p_h_S.view(B, K, S, M, 1).expand(B, K, S, M, H) 
    selected_log_p = torch.gather(log_p_e, 2, hyp_choices_e).squeeze(2)  
    log_p_M_S = selected_log_p.sum(2) 

    B, K, S, M, D = hypotheses.size() 
    B, K, M, H = choices.size()
    B, K, S, M, N = inlier_scores.size()
    hypotheses_e = hypotheses.view(B, K, S, M, 1, D).expand(B, K, S, M, H, D)
    inlier_scores_e = inlier_scores.view(B, K, S, M, 1, N).expand(B, K, S, M, H, N)
    residuals_e = residuals.view(B, K, S, M, 1, N).expand(B, K, S, M, H, N) 
    hyp_choices_e = choices.view(B, K, 1, M, H, 1).expand(B, K, 1, M, H, D)

    selected_hypotheses = torch.gather(hypotheses_e, 2, hyp_choices_e).squeeze(2) 
    hyp_choices_e = choices.view(B, K, 1, M, H, 1).expand(B, K, 1, M, H, N)
    selected_inlier_scores = torch.gather(inlier_scores_e, 2, hyp_choices_e).squeeze(2)  
    selected_residuals = torch.gather(residuals_e, 2, hyp_choices_e).squeeze(2) 

    return log_p_M_S, selected_inlier_scores, selected_hypotheses, selected_residuals 

def graph_based_sampling(weights, graph_adj_matrix, num_samples):
    
    device = weights.device
    
    degree_matrix = torch.diag(torch.sum(graph_adj_matrix, dim=-1))  
    normalized_adj = torch.matmul(torch.inverse(degree_matrix + 1e-5), graph_adj_matrix)  

    normalized_adj = normalized_adj.to(device)
    
    propagated_weights = torch.matmul(normalized_adj, weights.unsqueeze(-1)).squeeze(-1)  

    probabilities = propagated_weights / torch.sum(propagated_weights)
    
    samples = torch.multinomial(probabilities, num_samples, replacement=True)
    
    return samples


def create_geometric_adjacency_matrix(hypotheses, threshold=0.1):
    
    B, K, S, M = hypotheses.size()
    adjacency_matrix = torch.zeros(S, S)

    for i in range(S):
        for j in range(S):
            similarity = torch.norm(hypotheses[:, :, i, :] - hypotheses[:, :, j, :], dim=-1).mean()
            if similarity < threshold:
                adjacency_matrix[i, j] = 1

    return adjacency_matrix

def create_weighted_adjacency_matrix(weighted_inlier_counts, alpha=0.5):
    
    B, K, S, M = weighted_inlier_counts.size()
    adjacency_matrix = torch.zeros(S, S)

    for i in range(S):
        for j in range(S):
            weight_diff = torch.abs(weighted_inlier_counts[..., i, :] - weighted_inlier_counts[..., j, :]).mean()
            if weight_diff < alpha:
                adjacency_matrix[i, j] = 1

    return adjacency_matrix

def create_sparse_adjacency_matrix(S, sparsity=0.1):
    
    adjacency_matrix = (torch.rand(S, S) < sparsity).float()
    adjacency_matrix = torch.triu(adjacency_matrix, diagonal=1)  
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T  
    return adjacency_matrix

def create_weighted_adjacency_matrix3(weighted_inlier_counts, alpha=0.5):
    
    B, K, S, M = weighted_inlier_counts.size()
    
    weight_diff = torch.abs(weighted_inlier_counts.unsqueeze(2) - weighted_inlier_counts.unsqueeze(3)).mean(dim=-1)
    adjacency_matrix = (weight_diff < alpha).float()

    return adjacency_matrix

def create_weighted_adjacency_matrix2(weighted_inlier_counts, alpha=0.5):
    
    aggregated_weights = weighted_inlier_counts.mean(dim=(0, 1))  

    weight_diff = torch.abs(aggregated_weights.unsqueeze(1) - aggregated_weights.unsqueeze(0)).mean(dim=-1)  # [S, S]

    adjacency_matrix = (weight_diff < alpha).float()  # [S, S]

    return adjacency_matrix
