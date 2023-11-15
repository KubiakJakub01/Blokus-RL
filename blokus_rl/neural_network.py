"""Module with neural network wrapper for MCTS."""
import torch
import torch.nn.functional as F

from .colossumrl import ColosseumBlokusGameWrapper
from .hparams import MCTSHparams
from .utils import LOG_INFO, AverageMeter, to_device
from .models import get_model


class BlokusNNetWrapper:
    def __init__(
        self,
        game: ColosseumBlokusGameWrapper,
        hparams: MCTSHparams,
        device: str = "cpu",
        model_type: str | None = None,
    ):
        self.game = game
        self.hparams = hparams
        self.device = device
        self.model_type = model_type or hparams.model_type
        self.model = get_model(self.model_type)(game, hparams)
        self.model.to(self.device)
        self.elo = 1000
        self.latest_loss = 0
        self.mean_loss = AverageMeter()
        if len(list(self.model.parameters())) > 0:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

    def train_step(self, batch):
        """Train the model for one step.

        Args:
            batch: A batch of data.

        Returns:
            The loss."""

        # Set model to training mode
        self.model.train()

        # Move data to device
        batch = to_device(batch, self.device)

        # Get data from batch
        obs = batch["observation"]
        masks = batch["mask"]
        p_gt = batch["prob"]
        v_gt = batch["score"]

        # Forward pass
        p_pred, v_pred = self.model(obs)

        # Compute loss
        loss = self.compute_loss(masks, (p_pred, v_pred), (p_gt, v_gt))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track loss
        self.latest_loss = loss.item()
        self.mean_loss.update(self.latest_loss)

        # Return loss
        return loss.item()

    @torch.inference_mode()
    def predict(self, x, mask):
        """Predict the policy and value for a given state.

        Args:
            x: The state.
            mask: The mask of valid actions.

        Returns:
            The policy and value."""
        self.model.eval()
        x = torch.from_numpy(x).float().to(self.device).unsqueeze(0)
        mask = torch.from_numpy(mask).bool().to(self.device)
        p_logits, v = self.model(x)
        # EXP because log softmax
        p = self.get_valid_dist(mask, p_logits[0]).cpu().numpy().squeeze()
        return p, v.cpu().numpy().squeeze()

    @torch.inference_mode()
    def eval_fn(self, batch):
        self.model.eval()
        batch = to_device(batch, self.device)
        # Get data from batch
        obs = batch["observation"]
        masks = batch["mask"]
        p_gt = batch["prob"]
        v_gt = batch["score"]

        # Forward pass
        p_pred, v_pred = self.model(obs)

        # Compute loss
        loss = self.compute_loss(masks, (p_pred, v_pred), (p_gt, v_gt))

        return loss

    def compute_loss(self, masks, prediction, target):
        """Compute the loss.

        Args:
            masks: The mask of valid actions.
            prediction: The prediction.
            target: The target.

        Returns:
            The loss."""
        p_pred, v_pred = prediction
        p_gt, v_gt = target
        v_loss = (v_pred.squeeze() - v_gt).pow(2).mean()  # Mean squared error
        p_loss = 0
        for mask, gt, logits in zip(masks, p_gt, p_pred):
            pred = self.get_valid_dist(mask, logits, log_softmax=True)
            pred = F.pad(pred, (0, gt.shape[0] - pred.shape[0]), value=0)
            p_loss += -torch.sum(gt * pred)
        p_loss /= masks.size(0)
        return p_loss + v_loss

    def get_valid_dist(self, mask, logits, log_softmax=False):
        """Get the valid distribution.

        Args:
            mask: The mask of valid actions.
            logits: The logits.
            log_softmax: Whether to return the log softmax.

        Returns:
            The valid distribution."""
        selection = torch.masked_select(logits, mask)
        dist = torch.nn.functional.log_softmax(selection, dim=-1)
        if log_softmax:
            return dist
        return torch.exp(dist)

    def save_checkpoint(self, filename: str = "checkpoint.pth.tar"):
        """Save the model."""
        model_path = self.hparams.checkpoint_dir / filename
        LOG_INFO("Saving checkpoint to: %s", model_path)
        torch.save(
            {
                "nnet": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "mean_loss": self.mean_loss.avg,
                "latest_loss": self.latest_loss,
                "elo": self.elo,
            },
            model_path,
        )

    def load_checkpoint(self, filename: str = "checkpoint.pth.tar"):
        """Load the model."""
        model_path = self.hparams.checkpoint_dir / filename
        LOG_INFO("Loading model from: %s", str(model_path))
        assert model_path.exists(), f"Model path doesn't exist {model_path}"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["nnet"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.mean_loss.update(checkpoint["mean_loss"])
        self.latest_loss = checkpoint["latest_loss"]
        self.elo = checkpoint["elo"]
