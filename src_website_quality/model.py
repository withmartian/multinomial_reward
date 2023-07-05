import torch
from torch import nn

class Model(nn.Module):
  def __init__(self, transformer, processor, max_ranks_per_batch=27):
    super().__init__()
    self.config = transformer.config
    self.transformer = transformer
    self.v_head = nn.Linear(transformer.config.hidden_size, 1).to(self.transformer.device) # bias=False ?
    # fixme: convert v_head to torch_dtype? CLIP probs uses the default torch.float32
    self.processor = processor
    self.max_ranks_per_batch = max_ranks_per_batch


  def forward(self, pixel_values=None, labels=None, counts=None, **kwargs):
    transformer_outputs = self.transformer(
        pixel_values=pixel_values,
        output_hidden_states=True,
        **kwargs
    )

    hidden_states = transformer_outputs.last_hidden_state
    rewards = self.v_head(hidden_states).squeeze(-1) # total length x 50
    bs = len(counts) # number of websites

    token_size = rewards.shape[1] # 50 for pretrained CLIP

    # getting end_scores
    ptr = 0
    ranked_rewards = []
    for _ , slice_count_lst in counts:
      website_rewards = [] # contains each year of the website's rewards
      for i in range(len(slice_count_lst)):
        # averaging hidden states for slices
        website_rewards.append(rewards[ptr : ptr + slice_count_lst[i]].mean(dim=0, keepdim=True)) #torch.Size(1,50)
        ptr += slice_count_lst[i]
      ranked_rewards.append(torch.cat(website_rewards))

    # set score to -inf when the website has been padded to max_ranks
    # end_scores = [[float('-inf')]*bs for _ in range(self.max_ranks_per_batch)]
    # for i in range(len(ranked_rewards)):
    #   for j in range(len(ranked_rewards[i])):
    #     end_scores[j][i] = torch.mean(ranked_rewards[i][j])
    
    end_scores = [list() for _ in range(self.max_ranks_per_batch)]
    for i in range(len(ranked_rewards)):
      for j in range(len(ranked_rewards[i])):
        end_scores[j].append(torch.mean(ranked_rewards[i][j]))

    loss = torch.tensor(0.0, device=self.transformer.device, requires_grad=True)
    effective_batch_size = bs

    # getting ranked_rewards
    for i in range(bs):
      if len(ranked_rewards[i]) > 1:
        pairwise_rewards = torch.stack(
            [self.get_pairwise_reward(ranked_rewards[i][j], ranked_rewards[i][k])
            for (j,k) in all_pairs(len(ranked_rewards[i]))])

        mean = torch.mean(pairwise_rewards[~torch.any(pairwise_rewards.isnan())])
        if not mean.isnan():
          loss = loss + mean
        else:
          print("pairwise_rewards when isnan: ", pairwise_rewards)
          print("__ Inputs to pairwise_reward __")
          # print("all_pairs(len(unpadded)): ", all_pairs(len(unpadded)))
          # print([(ranked[j][i], ranked[k][i], ranked_rewards[j][i], ranked_rewards[k][i])
          #         for (j, k) in all_pairs(len(unpadded))])
          effective_batch_size -= 1
      else:
        effective_batch_size -= 1

    loss = loss / effective_batch_size if effective_batch_size > 0 else loss
    effective_batch_size = bs

    return {
            "end_scores": end_scores,
            "loss": loss
        }

  def get_pairwise_reward(self, reward1, reward2):
    return -torch.log(torch.sigmoid(reward1 - reward2)).mean()


def all_pairs(n):
  return [(i, j) for i in range(n) for j in range(i+1, n)]

