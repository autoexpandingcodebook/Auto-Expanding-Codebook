import torch
import torch.distributed as dist
from torchmetrics import Metric
from einops import pack, rearrange

class CodeBookMetric(Metric):
    def __init__(self, num_embeddings, group=2, use_loss=False):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.index_count = None
        self.history_index_count = None
        self.last_history_index_count = None

        self.use_loss = use_loss
        self.group = group
        if use_loss:
            self.register_buffer("index_loss", torch.zeros(group, num_embeddings))


    def update(self, indices, loss=None):
        # indices, ps = pack([indices], "g *")
        indices, ps = pack([indices], "* g")
        indices = rearrange(indices, "n g -> g n")
        g, _ = indices.shape

        index_counts = []
        for group in range(g):
            index_counts +=[
                torch.bincount(
                    indices[group].view(-1).long(), minlength=self.num_embeddings
                )
            ]

        if self.index_count is None:
            self.index_count = index_counts
        else:
            for group in range(g):
                self.index_count[group] = self.index_count[group] + index_counts[group]
        
        if loss is not None and self.use_loss:
            loss, ps = pack([loss], "* g")
            loss = rearrange(loss, "n g -> g n")
            g, _ = indices.shape
            device = indices.device
            self.index_loss = self.index_loss.to(device)
            for group in range(g):
                self.index_loss[group].scatter_add_(0, indices[group], loss[group])

    def compute_independent_codebook(self):
        def compute_group_metrics(index_count):
            # Compute probabilities for each group
            group_probs = index_count / torch.sum(index_count)

            # Group perplexity
            group_perplexity = torch.exp(
                -torch.sum(group_probs * torch.log(group_probs + 1e-10))
            ).item()

            # Group used codebook percentage
            group_used_codebook = (
                torch.count_nonzero(group_probs).item() / group_probs.numel()
            )

            return group_probs, group_perplexity, group_used_codebook

        # Compute metrics for each group
        total_perplexity = 0
        total_used_codebook = 1
        for group in range(len(self.index_count)):
            _, group_perplexity, group_used_codebook = compute_group_metrics(self.index_count[group])
            total_perplexity += group_perplexity
            total_used_codebook *= group_used_codebook

        # Aggregate metrics across groups
        total_perplexity /= len(self.index_count)  # Average perplexity across groups
        total_used_codebook = total_used_codebook * 100  # Convert to percentage

        return total_perplexity, total_used_codebook

    def compute(self):
        # Aggregate counts across all groups
        total_index_count = torch.zeros_like(self.index_count[0])
        for group in range(len(self.index_count)):
            total_index_count += self.index_count[group]

        # Compute shared probabilities for the codebook
        shared_probs = total_index_count / torch.sum(total_index_count)

        # Compute shared perplexity
        shared_perplexity = torch.exp(
            -torch.sum(shared_probs * torch.log(shared_probs + 1e-10))
        ).item()

        # Compute shared used codebook percentage
        shared_used_codebook = (
            torch.count_nonzero(total_index_count).item() / total_index_count.numel()
        ) * 100

        return shared_perplexity, shared_used_codebook
    
    def reset(self):
        self.index_count = None
        if self.use_loss:
            self.index_loss = torch.zeros(self.group, self.num_embeddings)

    def reset_last_history_index_count(self):
        if self.history_index_count is None:
            return
        group_nums = len(self.history_index_count)
        for group in range(group_nums):
            sum_history = torch.sum(self.history_index_count[group] != 0).item()
            sum_last = torch.sum(self.last_history_index_count[group] != 0).item()
            sum_change_zero = torch.sum((self.last_history_index_count[group] == 0) & (self.history_index_count[group] != 0)).item()
            print(f"Groupz:{group} sum_history:{sum_history} sum_last:{sum_last} sum_change_zero:{sum_change_zero}")

        for group in range(group_nums):
            self.history_index_count[group] = self.last_history_index_count[group].clone()

        self.last_history_index_count = None


    def reduce(self):
        for group in range(len(self.index_count)):
            tensor = self.index_count[group]
            dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            self.index_count[group] = tensor
        
            if self.use_loss:
                loss = self.index_loss[group]
                dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
                self.index_loss[group] = loss

        if self.history_index_count is None:
            self.history_index_count = []
            for group in range(len(self.index_count)):
                self.history_index_count.append(self.index_count[group].clone())
        else:
            for group in range(len(self.index_count)):
                self.history_index_count[group] += self.index_count[group]

        if self.last_history_index_count is None:
            self.last_history_index_count = []
            for group in range(len(self.index_count)):
                self.last_history_index_count.append(self.index_count[group].clone())
        else:
            for group in range(len(self.index_count)):
                self.last_history_index_count[group] += self.index_count[group]

    def get_result(self):
        self.reduce()
        perplexity, used_codebook = self.compute()
        prod_perplexity, prod_used_codebook = self.compute_independent_codebook()

        output = {
            "perplexity": perplexity,
            "used_codebook": used_codebook,
            "prod_perplexity": prod_perplexity,
            "prod_used_codebook": prod_used_codebook,
        }

        return output