"""
Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

extention of RLlib's MARWIL for RNN policy based on ray 2.5.1
1. Adds RNN states into SampleBatch as postprocess.
"""
import logging
from typing import Any, Dict, List, Optional, Type, Union
import numpy as np

from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import (
    ValueNetworkMixin,
    compute_gradients,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, get_variable
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.utils.typing import (
    LocalOptimizer,
    ModelGradients,
    TensorType,
)
from ray.rllib.algorithms.marwil.marwil_tf_policy import MARWILLoss, PostprocessAdvantages

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


class PostprocessRNNStates(PostprocessAdvantages):
    """Add seq_lens and states into the offline trajectory data without RNN states.
    """

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[Any, SampleBatch]] = None,
        episode=None,
    ):
        if self.is_recurrent() and not "state_in_0" in sample_batch:
            count = sample_batch.count
            seq_lens = []
            while count > 0:
                seq_lens.append(min(count, self.max_seq_len))
                count -= self.max_seq_len
            sample_batch["seq_lens"] = np.array(seq_lens)
            sample_batch.max_seq_len = self.max_seq_len
            dummy=self._get_dummy_batch_from_view_requirements(len(seq_lens))
            num_states=len([k for k in dummy.keys() if k.startswith("state_in")])
            for i in range(num_states):
                sample_batch["state_in_{}".format(i)]=dummy["state_in_{}".format(i)]

        return super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )

# We need this builder function because we want to share the same
# custom logics between TF1 dynamic and TF2 eager policies.
def get_rnn_marwil_tf_policy(name: str, base: type) -> type:
    """Construct a RNNMARWILTFPolicy inheriting either dynamic or eager base policies.

    Args:
        base: Base class for this policy. DynamicTFPolicyV2 or EagerTFPolicyV2.

    Returns:
        A TF Policy to be used with MAML.
    """

    class RNNMARWILTFPolicy(ValueNetworkMixin, PostprocessRNNStates, base):
        def __init__(
            self,
            observation_space,
            action_space,
            config,
            existing_model=None,
            existing_inputs=None,
        ):
            # First thing first, enable eager execution if necessary.
            base.enable_eager_execution_if_necessary()

            # Initialize base class.
            base.__init__(
                self,
                observation_space,
                action_space,
                config,
                existing_inputs=existing_inputs,
                existing_model=existing_model,
            )

            ValueNetworkMixin.__init__(self, config)
            PostprocessRNNStates.__init__(self)

            # Not needed for pure BC.
            if config["beta"] != 0.0:
                # Set up a tf-var for the moving avg (do this here to make it work
                # with eager mode); "c^2" in the paper.
                self._moving_average_sqd_adv_norm = get_variable(
                    config["moving_average_sqd_adv_norm_start"],
                    framework="tf",
                    tf_name="moving_average_of_advantage_norm",
                    trainable=False,
                )

            # Note: this is a bit ugly, but loss and optimizer initialization must
            # happen after all the MixIns are initialized.
            self.maybe_initialize_optimizer_and_loss()

        @override(base)
        def loss(
            self,
            model: Union[ModelV2, "tf.keras.Model"],
            dist_class: Type[TFActionDistribution],
            train_batch: SampleBatch,
        ) -> Union[TensorType, List[TensorType]]:
            model_out, _ = model(train_batch)
            action_dist = dist_class(model_out, model)
            value_estimates = model.value_function()

            self._marwil_loss = MARWILLoss(
                self,
                value_estimates,
                action_dist,
                train_batch,
                self.config["vf_coeff"],
                self.config["beta"],
            )

            return self._marwil_loss.total_loss

        @override(base)
        def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
            stats = {
                "policy_loss": self._marwil_loss.p_loss,
                "total_loss": self._marwil_loss.total_loss,
            }
            if self.config["beta"] != 0.0:
                stats["moving_average_sqd_adv_norm"] = self._moving_average_sqd_adv_norm
                stats["vf_explained_var"] = self._marwil_loss.explained_variance
                stats["vf_loss"] = self._marwil_loss.v_loss

            return stats

        @override(base)
        def compute_gradients_fn(
            self, optimizer: LocalOptimizer, loss: TensorType
        ) -> ModelGradients:
            return compute_gradients(self, optimizer, loss)

    RNNMARWILTFPolicy.__name__ = name
    RNNMARWILTFPolicy.__qualname__ = name

    return RNNMARWILTFPolicy


RNNMARWILTF1Policy = get_rnn_marwil_tf_policy("RNNMARWILTF1Policy", DynamicTFPolicyV2)
RNNMARWILTF2Policy = get_rnn_marwil_tf_policy("RNNMARWILTF2Policy", EagerTFPolicyV2)
