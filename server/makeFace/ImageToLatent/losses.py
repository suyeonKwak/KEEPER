import torch


class LatentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = L1Loss()
        self.log_cosh_loss = LogCoshLoss()
        self.l2_loss = torch.nn.MSELoss()

    def forward(
        self, real_features, generated_features, average_dlatents=None, dlatents=None
    ):
        loss = 0
        loss += 1 * self.l2_loss(real_features, generated_features)

        # Dlatent Loss - Forces latents to stay near the space the model uses for faces.
        if average_dlatents is not None and dlatents is not None:
            loss += 1 * 512 * self.l1_loss(average_dlatents, dlatents)

        return loss


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        loss = true - pred
        return torch.mean(torch.log(torch.cosh(loss + 1e-12)))


class L1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        return torch.mean(torch.abs(true - pred))
