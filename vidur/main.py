from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()
    print("Finished parsing config")

    set_seeds(config.seed)

    simulator = Simulator(config)
    print("Finished initializing simulator")
    simulator.run()
    print("Finished running simulator")


if __name__ == "__main__":
    """
    Example run commands below. For debug, use debugpy --wait-for-client --listen 5678 ./vidur/main.py instead of vidur.main.
    -VLLM:
        python -m vidur.main --replica_config_device a100 --replica_config_model_name meta-llama/Meta-Llama-3-8B --cluster_config_num_replicas 2 --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 --request_generator_config_type synthetic --synthetic_request_generator_config_num_requests 512  --length_generator_config_type trace --trace_request_length_generator_config_max_tokens 16384 --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv --interval_generator_config_type poisson --poisson_request_interval_generator_config_qps 6.45 --replica_scheduler_config_type vllm --vllm_scheduler_config_max_tokens_in_batch 8192 --global_scheduler_config_type lor --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384 --metrics_config_wandb_project "vidur" --metrics_config_wandb_group "default" --metrics_config_wandb_run_name "vllm_trace_splitwiseconv"
    -Splitwise:
        python -m vidur.main --replica_config_device a100 --replica_config_model_name meta-llama/Meta-Llama-3-8B --cluster_config_num_replicas 2 --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 --request_generator_config_type synthetic --synthetic_request_generator_config_num_requests 512  --length_generator_config_type trace --trace_request_length_generator_config_max_tokens 16384 --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv --interval_generator_config_type poisson --poisson_request_interval_generator_config_qps 6.45 --replica_scheduler_config_type splitwise --splitwise_scheduler_config_max_tokens_in_batch 8192 --global_scheduler_config_type splitwise --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384 --metrics_config_wandb_project "vidur" --metrics_config_wandb_group "default" --metrics_config_wandb_run_name "splitwise_trace_splitwiseconv"

    To generate simulation results for a combination of configuration parameters, edit the config.yml file in vidur/config_optimizer/config_explorer/config/config.yml and run the command: 
        python -m vidur.config_optimizer.config_explorer.main --output-dir <OUTPUT_DIR> --config-path vidur/config_optimizer/config_explorer/config/config.yml
    If using Windows, the file paths will have to be absolute, e.g.:
        python -m vidur.config_optimizer.config_explorer.main --output-dir C:/<PATH TO PROJECT>/<OUTPUT_DIR> --config-path C:/<PATH TO PROJECT>/vidur/config_optimizer/config_explorer/config/config.yml --cache-dir C:/<PATH TO PROJECT>/<CACHE_DIR>
    Set --max-iterations to reduce the time taken for the search.

    To generate an analysis of the simulation results, run the following command. <OUTPUT_DIR> will be the same folder as the one used in the previous command:
        python -m vidur.config_optimizer.analyzer.stats_extractor --sim-results-dir <OUTPUT_DIR> for Linux
        python -m vidur.config_optimizer.analyzer.stats_extractor --sim-results-dir C:/<PATH TO PROJECT>/<OUTPUT_DIR> for Windows

    To generate a dashboard for the analysis, run the following command:
        python -m streamlit run /config_optimizer/analyzer/dashboard/main.py -- --sim-results-dir <OUTPUT_DIR> for Linux
        python -m streamlit run C:/<PATH TO PROJECT>/config_optimizer/analyzer/dashboard/main.py -- --sim-results-dir C:/<PATH TO PROJECT>/<OUTPUT_DIR>/ for Windows
    """
    main()
