"""Policy model.

Fully-connected neural network with residual connections.
The models represent the agent's policy and map states
to actions.

"""
import functools
import torch
import torch.nn as nn

# 設定設備：如果有 CUDA 可用則使用 GPU，否則使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 將 eval 裝飾器從 utils.py 移至此處
def eval_decorator(function: callable) -> callable:
    """Evaluation decorator for class methods.

    Wraps function that calls a PyTorch module and ensures
    that inference is performed in evaluation model. Returns
    back to training mode after inference.

    Args:
        function: A callable.

    Returns:
        Decorated function.
    """

    @functools.wraps(function)
    def eval_wrapper(self, *args, **kwargs):
        self.eval()
        out = function(self, *args, **kwargs)
        self.train()
        return out

    return eval_wrapper


class ResidualBlock(nn.Module):
    """Simple MLP-block."""

    def __init__(self, in_features: int, out_features: int, args) -> None:
        """Initializes residual MLP-block."""
        super().__init__()

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.GELU(),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.Dropout(p=args.dropout_probability),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp_block(x)


class Model(nn.Module):
    def __init__(self, args) -> None:
        """Initializes the model."""
        super().__init__()

        field_size = args.field_size
        dims_state = field_size**2
        num_actions = field_size**2
        hidden_features = args.num_hidden_units
        num_layers = args.num_layers
        prob_dropout = args.dropout_probability

        input_layer = [
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=dims_state, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=prob_dropout),
        ]

        hidden_layers = [
            ResidualBlock(in_features=hidden_features, out_features=hidden_features, args=args)
            for _ in range(num_layers)
        ]

        output_layer = [
            nn.Linear(in_features=hidden_features, out_features=num_actions),
            nn.Softmax(dim=-1) if args.algorithm == "policy_gradient" else nn.Identity(),
        ]

        self.model = nn.Sequential(*input_layer, *hidden_layers, *output_layer)
        # 將模型移至指定設備
        self.to(device)

    @eval_decorator
    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> int:
        """Predicts action for given state.

        Args:
            state: Tensor representing state of playing field.

        Returns:
            The action represented by an integer.
        """
        # 確保輸入張量在正確的設備上
        state = state.to(device)
        prediction = self(state)
        action = torch.argmax(prediction, dim=-1).item()
        return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 確保輸入張量在正確的設備上
        x = x.to(device)
        x = self.model(x)
        return x
