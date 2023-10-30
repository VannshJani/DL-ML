import torch
from astra.torch.al.aquisition.base import EnsembleAcquisition



    
# maximum mean standard deviation aquisition function
class Mean_std(EnsembleAcquisition):
    def acquire_scores(self, logits: torch.Tensor) -> torch.Tensor:
        # Mean-STD acquisition function
        # (n_nets, pool_dim, n_classes) shape
        n_models = logits.shape[0]
        pool_num = logits.shape[1]
        n_classes = logits.shape[2]
        scores = torch.zeros(pool_num)
        for idx in range(pool_num):
            model_predictions = logits[:, idx, :] # (n_nets, n_classes) shape
            standard_deviations = []
            for class_idx in range(n_classes):
                model_predictions_per_class = model_predictions[:, class_idx] # (n_nets) shape
                e2 = torch.mean(model_predictions_per_class**2)
                e1_2 = torch.mean(model_predictions_per_class)**2
                std = torch.sqrt(e2 - e1_2)
                standard_deviations.append(std)
            standard_deviations = torch.tensor(standard_deviations)
            mean_std = torch.mean(standard_deviations)
            scores[idx] = mean_std

        return scores