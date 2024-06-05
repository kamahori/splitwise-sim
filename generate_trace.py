import os

from collections import namedtuple
import json
import requests

import numpy as np
import pandas as pd

from scipy import stats


Distributions = namedtuple('Distributions', ['application_id',
                                             'request_type',
                                             'arrival_process',
                                             'batch_size',
                                             'prompt_size',
                                             'token_size'])
Distribution = namedtuple('Distribution', ['name', 'params'])


def generate_samples(distribution, params, size):
    """
    Generate random samples from the given distribution.
    """
    if distribution == "constant":
        return np.ones(size) * params["value"]
    elif distribution == "normal":
        return stats.norm(**params).rvs(size=size)
    elif distribution == "truncnorm":
        return stats.truncnorm(**params).rvs(size=size)
    elif distribution == "randint":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "uniform":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "exponential":
        return stats.expon(**params).rvs(size=size)
    elif distribution == "poisson":
        return stats.poisson(**params).rvs(size=size)
    elif distribution == "trace":
        df = pd.read_csv(params["filename"])
        return df[params["column"]].sample(size, replace=True).values
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def generate_trace(max_requests, distributions, end_time=None):
    """
    Generate a trace of requests based on the given distributions.
    """
    # Generate request IDs
    request_ids = np.arange(max_requests)

    # Generate the distributions
    arrival_timestamps = generate_samples(distributions.arrival_process.name,
                                          distributions.arrival_process.params,
                                          max_requests)
    arrival_timestamps = np.cumsum(arrival_timestamps)
    application_ids = generate_samples(distributions.application_id.name,
                                       distributions.application_id.params,
                                       max_requests)
    application_ids = map(int, application_ids)
    batch_sizes = generate_samples(distributions.batch_size.name,
                                   distributions.batch_size.params,
                                   max_requests)
    batch_sizes = map(int, batch_sizes)
    prompt_sizes = generate_samples(distributions.prompt_size.name,
                                    distributions.prompt_size.params,
                                    max_requests)
    prompt_sizes = map(int, prompt_sizes)
    token_sizes = generate_samples(distributions.token_size.name,
                                   distributions.token_size.params,
                                   max_requests)
    token_sizes = map(int, token_sizes)
    request_type_ids = generate_samples(distributions.request_type.name,
                                        distributions.request_type.params,
                                        max_requests)
    request_type_ids = map(int, request_type_ids)

    # Combine the arrays into a DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,
        "request_type": request_type_ids,
        "application_id": application_ids,
        "arrival_timestamp": arrival_timestamps,
        "batch_size": batch_sizes,
        "prompt_size": prompt_sizes,
        "token_size": token_sizes,
    })

    if end_time is not None:
        trace_df = trace_df[trace_df["arrival_timestamp"] < end_time]

    return trace_df


def get_exponential_scale(num_servers, utilization, request_duration):
    """
    assumes that request_duration is in seconds
    """
    interarrival_time = request_duration / (1.0 * utilization)
    exponential_scale = interarrival_time / num_servers
    return exponential_scale


def generate_trace_from_utilization(
    max_requests,
    end_time,
    num_servers,
    utilization,
    request_duration,
    pt_distributions_file):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    exponential_scale = get_exponential_scale(num_servers, utilization, request_duration)
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference
        arrival_process=Distribution("exponential", {"scale": exponential_scale}),
        prompt_size=Distribution("trace", {"filename": pt_distributions_file,
                                           "column": "ContextTokens"}),
        token_size=Distribution("trace", {"filename": pt_distributions_file,
                                          "column": "GeneratedTokens"}),
        batch_size=Distribution("constant", {"value": 1}),
    )

    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)
    return trace_df


def generate_trace_from_prompt_token_size_distributions(
    max_requests,
    end_time,
    request_rate,
    pt_distributions_filename):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference
        arrival_process=Distribution("exponential", {"scale": 1.0 / request_rate}),
        prompt_size=Distribution("trace", {"filename": pt_distributions_filename,
                                           "column": "ContextTokens"}),
        #prompt_size=Distribution("truncnorm", {"a": (prompt_min-prompt_mean)/prompt_std,
        #                                       "b": (prompt_max-prompt_mean)/prompt_std,
        #                                       "loc": prompt_mean,
        #                                       "scale": prompt_std}),
        token_size=Distribution("trace", {"filename": pt_distributions_filename,
                                          "column": "GeneratedTokens"}),
        #token_size=Distribution("truncnorm", {"a": (token_min-token_mean)/token_std,
        #                                      "b": (token_max-token_mean)/token_std,
        #                                      "loc": token_mean,
        #                                      "scale": token_std}),
        batch_size=Distribution("constant", {"value": 1}),
    )
    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)
    return trace_df


def generate_traces(max_requests,
                    end_time,
                    request_rates,
                    pt_distributions_file,
                    trace_filename_template):
    """
    Generate traces with prompt/token size distributions.
    """
    for request_rate in request_rates:
        trace_df = generate_trace_from_prompt_token_size_distributions(
            max_requests,
            end_time,
            request_rate,
            pt_distributions_file)
        trace_filename = trace_filename_template.format(request_rate)
        trace_df.to_csv(trace_filename, index=False)

def generate_moe_trace(max_requests, distributions, expert_popularity_filename, end_time=None):
    """
    Generate a trace of requests based on the given distributions.

    TODO (keisuke): this is generating expert selection per request, not per token.
    """
    # Generate request IDs
    request_ids = np.arange(max_requests)

    # Generate the distributions
    arrival_timestamps = generate_samples(distributions.arrival_process.name,
                                          distributions.arrival_process.params,
                                          max_requests)
    arrival_timestamps = np.cumsum(arrival_timestamps)
    application_ids = generate_samples(distributions.application_id.name,
                                       distributions.application_id.params,
                                       max_requests)
    application_ids = map(int, application_ids)
    batch_sizes = generate_samples(distributions.batch_size.name,
                                   distributions.batch_size.params,
                                   max_requests)
    batch_sizes = map(int, batch_sizes)
    prompt_sizes = generate_samples(distributions.prompt_size.name,
                                    distributions.prompt_size.params,
                                    max_requests)
    prompt_sizes = map(int, prompt_sizes)
    token_sizes = generate_samples(distributions.token_size.name,
                                   distributions.token_size.params,
                                   max_requests)
    token_sizes = map(int, token_sizes)
    request_type_ids = generate_samples(distributions.request_type.name,
                                        distributions.request_type.params,
                                        max_requests)
    request_type_ids = map(int, request_type_ids)

    # Load expert popularity data
    # TODO (keisuke): stop hard-coding the number of layers (32)
    # and the number of experts to be selected (2)
    expert_selection = np.zeros((32, max_requests, 2), dtype=np.uint8)
    with open(expert_popularity_filename, "r") as f:
        expert_popularity = json.load(f)
        expert_selection_prob = np.array(expert_popularity["expert_popularity"]) # (32, 8)
        # normalize
        expert_selection_prob = expert_selection_prob / expert_selection_prob.sum(axis=1, keepdims=True)
        for i in range(32):
            # generate top-2 selection based on the probability for `max_requests` instances
            expert_selection[i] = np.random.choice(8, size=(max_requests, 2), p=expert_selection_prob[i])

    # Combine the arrays into a DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,
        "request_type": request_type_ids,
        "application_id": application_ids,
        "arrival_timestamp": arrival_timestamps,
        "batch_size": batch_sizes,
        "prompt_size": prompt_sizes,
        "token_size": token_sizes,
    } | {
        f"expert_{i}_{j}": expert_selection[i, :, j] for i in range(32) for j in range(2)
    })

    if end_time is not None:
        trace_df = trace_df[trace_df["arrival_timestamp"] < end_time]

    return trace_df

def generate_trace_from_prompt_token_size_distributions_and_expert_popularity(
    max_requests,
    end_time,
    request_rate,
    pt_distributions_filename,
    expert_popularity_filename):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference
        arrival_process=Distribution("exponential", {"scale": 1.0 / request_rate}),
        prompt_size=Distribution("trace", {"filename": pt_distributions_filename,
                                           "column": "ContextTokens"}),
        #prompt_size=Distribution("truncnorm", {"a": (prompt_min-prompt_mean)/prompt_std,
        #                                       "b": (prompt_max-prompt_mean)/prompt_std,
        #                                       "loc": prompt_mean,
        #                                       "scale": prompt_std}),
        token_size=Distribution("trace", {"filename": pt_distributions_filename,
                                          "column": "GeneratedTokens"}),
        #token_size=Distribution("truncnorm", {"a": (token_min-token_mean)/token_std,
        #                                      "b": (token_max-token_mean)/token_std,
        #                                      "loc": token_mean,
        #                                      "scale": token_std}),
        batch_size=Distribution("constant", {"value": 1}),
    )
    trace_df = generate_moe_trace(max_requests,
                              distributions,
                              expert_popularity_filename,
                              end_time=end_time)
    return trace_df


def generate_moe_traces(max_requests,
                    end_time,
                    request_rates,
                    pt_distributions_file,
                    expert_popularity_file,
                    trace_filename_template):
    """
    Generate traces with prompt/token size distributions.
    """
    for request_rate in request_rates:
        trace_df = generate_trace_from_prompt_token_size_distributions_and_expert_popularity(
            max_requests,
            end_time,
            request_rate,
            pt_distributions_file,
            expert_popularity_file)
        trace_filename = trace_filename_template.format(request_rate)
        trace_df.to_csv(trace_filename, index=False)


def generate_code_traces(
    max_requests,
    end_time,
    request_rates,
    code_distributions_file,
    trace_filename_template="traces/rr_code_{}.csv"):
    """
    code traces distribution
    prompt_mean = 2048, prompt_std = 1973, prompt_min = 3, prompt_max = 7437
    token_mean = 28, token_std = 60, token_min = 6, token_max = 1899
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    code_distributions_file,
                    trace_filename_template)


def generate_conv_traces(
    max_requests,
    end_time,
    request_rates,
    conv_distributions_file,
    trace_filename_template="traces/rr_conv_{}.csv"):
    """
    conv traces distribution
    prompt_mean = 1155, prompt_std = 1109, prompt_min = 2, prompt_max = 14050
    token_mean = 211, token_std = 163, token_min = 7, token_max = 1000
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    conv_distributions_file,
                    trace_filename_template)

def generate_moe_code_traces(
    max_requests,
    end_time,
    request_rates,
    code_distributions_file,
    expert_popularity_file,
    trace_filename_template="traces/rr_moe_code_{}.csv"):
    """
    code traces distribution
    prompt_mean = 2048, prompt_std = 1973, prompt_min = 3, prompt_max = 7437
    token_mean = 28, token_std = 60, token_min = 6, token_max = 1899
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_moe_traces(max_requests,
                        end_time,
                        request_rates,
                        code_distributions_file,
                        expert_popularity_file,
                        trace_filename_template)


def generate_moe_conv_traces(
    max_requests,
    end_time,
    request_rates,
    conv_distributions_file,
    expert_popularity_file,
    trace_filename_template="traces/rr_moe_conv_{}.csv"):
    """
    conv traces distribution
    prompt_mean = 1155, prompt_std = 1109, prompt_min = 2, prompt_max = 14050
    token_mean = 211, token_std = 163, token_min = 7, token_max = 1000
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_moe_traces(max_requests,
                        end_time,
                        request_rates,
                        conv_distributions_file,
                        expert_popularity_file,
                        trace_filename_template)

def download_file(url, filename):
    """
    Download a file from the given URL.
    """
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)


def download_azure_llm_traces():
    """
    Download traces from the given URL.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    url_base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"

    if not os.path.exists("data/code_distributions.csv"):
        url = url_base + "AzureLLMInferenceTrace_code.csv"
        download_file(url, "data/code_distributions.csv")
        print("Downloaded code traces")

    if not os.path.exists("data/conv_distributions.csv"):
        url = url_base + "AzureLLMInferenceTrace_conv.csv"
        download_file(url, "data/conv_distributions.csv")
        print("Downloaded conv traces")

def download_fiddler_expert_traces():
    """
    Download traces from the given URL.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    url_base = "https://github.com/efeslab/fiddler/raw/popularity/benchmarks/results/popularity/mixtral_8x7b_instruct/"
    categories = [
        "college_computer_science",
        "college_medicine",
        "high_school_biology",
        "high_school_mathematics",
        "high_school_us_history",
        "international_law",
        "machine_learning",
    ]
    for category in categories:
        if not os.path.exists(f"data/expert_popularity_{category}.json"):
            url = os.path.join(url_base, f"expert_popularity_{category}.json")
            download_file(url, f"data/expert_popularity_{category}.json")
            print(f"Downloaded {category} expert popularity data")


if __name__ == "__main__":
    # download prompt and token size distributions
    download_azure_llm_traces()
    download_fiddler_expert_traces()

    # generate request traces
    generate_code_traces(
        max_requests=1000000,
        end_time=600,
        request_rates=list(range(30, 251, 10)),
        code_distributions_file="data/code_distributions.csv")
    print("Generated code traces")

    generate_conv_traces(
        max_requests=1000000,
        end_time=600,
        request_rates=list(range(30, 251, 10)),
        conv_distributions_file="data/conv_distributions.csv")
    print("Generated conv traces")

    # generate request traces for 2 min
    generate_code_traces(
        max_requests=1000000,
        end_time=120,
        request_rates=list(range(30, 101, 10)),
        code_distributions_file="data/code_distributions.csv",
        trace_filename_template="traces/rr_code_{}_2min.csv")
    print("Generated code 2min traces")

    generate_conv_traces(
        max_requests=1000000,
        end_time=120,
        request_rates=list(range(30, 101, 10)),
        conv_distributions_file="data/conv_distributions.csv",
        trace_filename_template="traces/rr_conv_{}_2min.csv")
    print("Generated conv 2min traces")

    # generate moe request traces
    generate_moe_code_traces(
        max_requests=1000000,
        end_time=600,
        request_rates=list(range(30, 251, 10)),
        code_distributions_file="data/code_distributions.csv",
        expert_popularity_file="data/expert_popularity_college_computer_science.json",
        trace_filename_template="traces/rr_moe_code_{}.csv")
    print("Generated moe code traces")

    generate_moe_conv_traces(
        max_requests=1000000,
        end_time=600,
        request_rates=list(range(30, 251, 10)),
        conv_distributions_file="data/conv_distributions.csv",
        expert_popularity_file="data/expert_popularity_college_computer_science.json",
        trace_filename_template="traces/rr_moe_conv_{}.csv")
    print("Generated moe conv traces")

    # generate moe request traces for 2 min
    generate_moe_code_traces(
        max_requests=1000000,
        end_time=120,
        request_rates=list(range(30, 101, 10)),
        code_distributions_file="data/code_distributions.csv",
        expert_popularity_file="data/expert_popularity_college_computer_science.json",
        trace_filename_template="traces/rr_moe_code_{}_2min.csv")
    print("Generated moe code 2min traces")

    generate_moe_conv_traces(
        max_requests=1000000,
        end_time=120,
        request_rates=list(range(30, 101, 10)),
        conv_distributions_file="data/conv_distributions.csv",
        expert_popularity_file="data/expert_popularity_college_computer_science.json",
        trace_filename_template="traces/rr_moe_conv_{}_2min.csv")
    print("Generated moe conv 2min traces")
