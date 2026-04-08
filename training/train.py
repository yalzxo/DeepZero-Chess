import torch
import torch.nn.functional as F


def train(model, replay_buffer, epochs=1, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        batch = replay_buffer.sample(64)

        total_loss = 0

        for state, policy, value in batch:
            state = torch.tensor(state).unsqueeze(0).float()
            target_policy = torch.tensor(policy).unsqueeze(0).float()
            target_value = torch.tensor([[value]]).float()

            pred_policy, pred_value = model(state)

            policy_loss = -torch.sum(
                target_policy *
                torch.log_softmax(pred_policy, dim=1)
            )

            value_loss = F.mse_loss(pred_value, target_value)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch:", epoch, "Loss:", total_loss)