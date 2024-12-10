from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()


if __name__ == "__main__":
    main()

# Example run:
# python -m debugpy --wait-for-client --listen 5678 ./vidur/main.py --replica_config_device a100 --replica_config_model_name meta-llama/Meta-Llama-3-8B --cluster_config_num_replicas 2 --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 --request_generator_config_type synthetic --synthetic_request_generator_config_num_requests 512  --length_generator_config_type trace --trace_request_length_generator_config_max_tokens 16384 --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv --interval_generator_config_type poisson --poisson_request_interval_generator_config_qps 6.45 --replica_scheduler_config_type vllm --vllm_scheduler_config_max_tokens_in_batch 8192 --global_scheduler_config_type round_robin --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384 --metrics_config_wandb_project "vidur" --metrics_config_wandb_group "default" --metrics_config_wandb_run_name "vllm_trace_splitwiseconv"
# ython -m debugpy --wait-for-client --listen 5678 ./vidur/main.py --replica_config_device a100 --replica_config_model_name meta-llama/Meta-Llama-3-8B --cluster_config_num_replicas 2 --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 --request_generator_config_type synthetic --synthetic_request_generator_config_num_requests 512  --length_generator_config_type trace --trace_request_length_generator_config_max_tokens 16384 --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv --interval_generator_config_type poisson --poisson_request_interval_generator_config_qps 6.45 --replica_scheduler_config_type splitwise --splitwise_scheduler_config_max_tokens_in_batch 8192 --global_scheduler_config_type splitwise --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384 --metrics_config_wandb_project "vidur" --metrics_config_wandb_group "default" --metrics_config_wandb_run_name "splitwise_trace_splitwiseconv"
