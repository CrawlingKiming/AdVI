"""Implementations of autoregressive transforms."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from .diagnostic import tile
from survae import utils
from survae import transforms
#from survae.transforms import made as made_module
from survae.transforms import splines

def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(self,
                 in_degrees,
                 out_features,
                 autoregressive_features,
                 random_mask,
                 is_output,
                 bias=True,
                 zero_initialization = True):
        super().__init__(
            in_features=len(in_degrees),
            out_features=out_features,
            bias=bias)
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

        if zero_initialization:
            #init_weights(self.linear_layers)
            #for layers in self.linear_layers:
            init.zeros_(self.weight)
            init.zeros_(self.bias)

    @classmethod
    def _get_mask_and_degrees(cls,
                              in_degrees,
                              out_features,
                              autoregressive_features,
                              random_mask,
                              is_output):
        if is_output:
            out_degrees = tile(
                _get_input_degrees(autoregressive_features),
                out_features // autoregressive_features
            )
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long)
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.
    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        if random_mask:
            raise ValueError('Masked residual block can\'t be used with random masks.')
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError('In a masked residual block, the output degrees can\'t be'
                               ' less than the corresponding input degrees.')

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            #init_weights(self.linear_layers)
            for layers in self.linear_layers:
                init.zeros_(layers.weight)
                init.zeros_(layers.bias)
            # init.uniform_(self.linear_layers[-1].weight, a=-1e-6, b=1e-6)
            # init.uniform_(self.linear_layers[-1].bias, a=-1e-6, b=1e-6)
            # nn.init.zeros_(self.linear_layers[-1].weight)
            # nn.init.zeros_(self.linear_layers[-2].weight)
            # nn.init.zeros_(self.linear_layers[-1].bias)
    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat((temps, self.context_layer(context)), dim=1),
                dim=1
            )
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE.
    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 output_multiplier=1,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        if use_residual_blocks and random_mask:
            raise ValueError('Residual blocks can\'t be used with random masks.')
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True
        )

    def forward(self, inputs, context=None):
        outputs = self.initial_layer(inputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, context)
        outputs = self.final_layer(outputs)
        return outputs

class AutoregressiveTransform(transforms.Transform):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(AutoregressiveTransform, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(inputs, autoregressive_params)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()

class MaskedPiecewiseRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_bins=10,
                 tails=True, #None
                 tail_bound=2., #1.
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE
                 ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size,
            features,
            self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins:]

        if hasattr(self.autoregressive_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == 'linear':
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

