from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, transformer, tokenizer, max_ranks_per_batch=2):
        super().__init__()
        self.config = transformer.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.config.torch_dtype = next(transformer.parameters()).dtype
        self.v_head = self.v_head.to(dtype=self.config.torch_dtype)
        self.tokenizer = tokenizer
        self.PAD_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.max_ranks_per_batch = max_ranks_per_batch


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,  # Trainer passes in labels, but it's not a kwarg for hf transformers, so we consume it here
            **kwargs
    ):
        attention_mask, inputs_embeds = self.convert_tensors(attention_mask, inputs_embeds)

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = input_ids.shape[0] // self.max_ranks_per_batch

        ranked = [input_ids[i:i + bs] for i in range(0, len(input_ids), bs)]
        ranked_rewards = [rewards[i:i + bs] for i in range(0, len(rewards), bs)]

        end_scores = [list() for _ in range(self.max_ranks_per_batch)]

        for i in range(bs):
            for j in range(self.max_ranks_per_batch):
                end_scores[j].append(ranked_rewards[j][i][self.get_start_of_padding(ranked[j][i]) - 1])

        loss = torch.tensor(0.0, device=self.transformer.device, requires_grad=True)
        effective_batch_size = bs

        for i in range(bs):
            unpadded = [rank[i] for rank in ranked if
                        not torch.equal(rank[i], torch.tensor(self.PAD_ID).repeat(rank[i].shape).to(self.transformer.device))]
            # If we don't have enough for a proper comparison, how should it be factored into the loss?
            if len(unpadded) > 1:
                pairwise_rewards = torch.stack(
                    [self.get_pairwise_reward(ranked[j][i], ranked[k][i], ranked_rewards[j][i], ranked_rewards[k][i])
                     for (j, k) in all_pairs(len(unpadded))])

                mean = torch.mean(pairwise_rewards[~torch.any(pairwise_rewards.isnan())])
                if not mean.isnan():
                    loss = loss + mean
                else:
                    print("pairwise_rewards when isnan: ", pairwise_rewards)
                    print("__ Inputs to pairwise_reward __")
                    print("all_pairs(len(unpadded)): ", all_pairs(len(unpadded)))
                    print([(ranked[j][i], ranked[k][i], ranked_rewards[j][i], ranked_rewards[k][i])
                           for (j, k) in all_pairs(len(unpadded))])
                    effective_batch_size -= 1
            else:
                effective_batch_size -= 1

        if not (effective_batch_size > 0):
            print("effective_batch_size: ", effective_batch_size)

        loss = loss / effective_batch_size if effective_batch_size > 0 else loss

        return {
            "end_scores": end_scores,
            "loss": loss
        }

    def convert_tensors(self, *tensors):
        return [t.to(dtype=self.config.torch_dtype) if t is not None else None for t in tensors]

    def get_pairwise_reward(self, input1, input2, reward1, reward2):
        end_of_prompt = self.get_end_of_prompt(input1, input2)
        start_of_padding = max(self.get_start_of_padding(input1), self.get_start_of_padding(input2))
        return -torch.log(
            torch.sigmoid(reward1[end_of_prompt:start_of_padding] - reward2[end_of_prompt:start_of_padding])).mean()

    def get_end_of_prompt(self, input1, input2):
        inds = (input1 != input2).nonzero()
        return inds[0] if len(inds) > 0 else len(input1) - 1

    def get_start_of_padding(self, input1):
        inds = (input1 == self.PAD_ID).nonzero()
        return inds[0].item() if len(inds) > 0 else len(input1)

    def loss(self, outputs, batch):
        return outputs["loss"]


def all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]
