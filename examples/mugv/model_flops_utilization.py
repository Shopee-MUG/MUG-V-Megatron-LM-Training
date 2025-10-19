import megatron.training.training
from megatron.training.utils import print_rank_0


# TODO: update mugdit num floating point operations with online average computation,
# as computation is variable across iterations.
class FlopsCounter:

    def __init__(self):
        self.cnt = 0
        self.avg_num_flops = 0

    def mugdit_num_floating_point_operations(self, args, batch_size):
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # MoE.
        num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
        gated_linear_multiplier = 3 / 2 if args.swiglu else 1

        # The 12x term below comes from the following factors; for more details, see
        # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
        #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
        #       in MLP layer).
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        # - 4x: Recompute granularity is enabled, so each GEMM in the model needs to be performed 4 times.
        expansion_factor = 3 * 2 * 2 if args.recompute_granularity is None else 4 * 2 * 2

        cur_batch_shape = getattr(args, 'cur_batch_shape', 1) # for dynamic batch shape
        cur_nano_batchsize = cur_batch_shape[0]
        t, h, w = cur_batch_shape[2:]
        cur_seq_length = t * h * w  / 1 / 2 / 2
        cur_context_length = 300
        cur_flops = (
            expansion_factor
            * batch_size * cur_nano_batchsize
            * cur_seq_length
            * args.num_layers
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                (
                    (
                        1
                        + (args.num_query_groups / args.num_attention_heads)
                        + (cur_seq_length / args.hidden_size)
                    ) * query_projection_to_hidden_size_ratio
                )
                # Cross attention.
                + (
                    (
                        1
                        + (cur_context_length / cur_seq_length) * (args.num_query_groups / args.num_attention_heads)
                        + (cur_context_length / args.hidden_size)
                    ) * query_projection_to_hidden_size_ratio
                )
                # MLP.
                + (
                    (args.ffn_hidden_size / args.hidden_size)
                    * num_experts_routed_to
                    * gated_linear_multiplier
                )
            )
        )

        self.cnt += 1
        self.avg_num_flops += (cur_flops - self.avg_num_flops) / self.cnt
        return self.avg_num_flops


flops_counter = FlopsCounter()


def register_flops_hook_for_logging():
    print_rank_0("- [Hook] Registering Flops Counter hook")
    megatron.training.training.num_floating_point_operations = flops_counter.mugdit_num_floating_point_operations
