import logging
import random
import json 
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import count
import numpy as np 
import networkx as nx

from executor import Executor
from flow import Flow
from metrics import RequestMetrics, GenerativeLLMRequestMetrics, RequestSLO
from node import Node
from simulator import clock, schedule_event, cancel_event, reschedule_event
from task import Task, TaskType


class RequestState(IntEnum):
    """
    RequestState describes the states of a Request.
    """
    NONE = 0
    QUEUED_AT_ROUTER = 1
    QUEUED_AT_SCHEDULER = 2
    RUNNING_ON_EXECUTOR = 3
    COMPLETED_AT_SCHEDULER = 4
    COMPLETED_AT_ROUTER = 5
    ABORTED = 6


class RequestType(IntEnum):
    COMPUTE = 0 # Not implemented
    DNN = 1 # Not implemented
    GENERATIVE_LLM = 2


@dataclass(kw_only=True)
class Request():
    """
    Request is a DAG of Tasks and Flows targeting an Application.
    Requests must have a single root Node.
    """
    request_id: int
    node_id: count = field(default_factory=count)
    application_id: int
    request_type: RequestType
    batch_size: int = 1
    arrival_timestamp: float = 0.
    state: RequestState = field(default=RequestState.NONE)
    dag: nx.DiGraph = field(default_factory=nx.DiGraph)
    root_node: Node = None
    nodes: dict = field(default_factory=dict)
    metrics: RequestMetrics = field(default_factory=RequestMetrics)
    slo: RequestSLO = field(default_factory=RequestSLO)
    executor: Executor = None

    # TODO (keisuke): I want to stop hard coding this foolish thing
    expert_0_0: int = 0
    expert_0_1: int = 0
    expert_1_0: int = 0
    expert_1_1: int = 0
    expert_2_0: int = 0
    expert_2_1: int = 0
    expert_3_0: int = 0
    expert_3_1: int = 0
    expert_4_0: int = 0
    expert_4_1: int = 0
    expert_5_0: int = 0
    expert_5_1: int = 0
    expert_6_0: int = 0
    expert_6_1: int = 0
    expert_7_0: int = 0
    expert_7_1: int = 0
    expert_8_0: int = 0
    expert_8_1: int = 0
    expert_9_0: int = 0
    expert_9_1: int = 0
    expert_10_0: int = 0
    expert_10_1: int = 0
    expert_11_0: int = 0
    expert_11_1: int = 0
    expert_12_0: int = 0
    expert_12_1: int = 0
    expert_13_0: int = 0
    expert_13_1: int = 0
    expert_14_0: int = 0
    expert_14_1: int = 0
    expert_15_0: int = 0
    expert_15_1: int = 0
    expert_16_0: int = 0
    expert_16_1: int = 0
    expert_17_0: int = 0
    expert_17_1: int = 0
    expert_18_0: int = 0
    expert_18_1: int = 0
    expert_19_0: int = 0
    expert_19_1: int = 0
    expert_20_0: int = 0
    expert_20_1: int = 0
    expert_21_0: int = 0
    expert_21_1: int = 0
    expert_22_0: int = 0
    expert_22_1: int = 0
    expert_23_0: int = 0
    expert_23_1: int = 0
    expert_24_0: int = 0
    expert_24_1: int = 0
    expert_25_0: int = 0
    expert_25_1: int = 0
    expert_26_0: int = 0
    expert_26_1: int = 0
    expert_27_0: int = 0
    expert_27_1: int = 0
    expert_28_0: int = 0
    expert_28_1: int = 0
    expert_29_0: int = 0
    expert_29_1: int = 0
    expert_30_0: int = 0
    expert_30_1: int = 0
    expert_31_0: int = 0
    expert_31_1: int = 0

    def __post_init__(self):
        pass

    def __hash__(self):
        """
        NOTE: hash functions get overridden to None in child classes
        """
        return hash(self.request_id)

    def __eq__(self, other):
        return self.request_id == other.request_id

    def successors(self, node):
        """
        Returns the next Task or Flow to be executed after node.
        """
        return self.dag.successors(node)

    def predecessors(self, node):
        """
        Returns the previous Task or Flow to be executed before node.
        """
        return self.dag.predecessors(node)

    def get_node(self, node_id):
        """
        Returns the Node with node_id from the DAG.
        # NOTE: could alternatively store node_ids in DAG and node as attribute
        """
        return self.nodes[node_id]

    def get_node_metrics(self, node_id):
        """
        Returns the metrics of the Node with node_id.
        """
        node = self.get_node(node_id)
        if isinstance(node, Task):
            node_type = node.task_type.name
            runner = f"{node.instance.name}_{node.instance.instance_id}"
        elif isinstance(node, Flow):
            node_type = node.flow_type.name
            runner = node.link.name
        else:
            raise ValueError("Unsupported node type")
        data = {
            "request_id": self.request_id,
            "request_type": self.request_type,
            "node_id": node_id,
            "node_type": node_type,
            "runner": runner,
            "start_timestamp": node.metrics.start_timestamp,
            "completion_timestamp": node.metrics.completion_timestamp,
        }
        return data

    def get_all_node_metrics(self):
        data = []
        for node_id in self.nodes:
            data.append(self.get_node_metrics(node_id))
        return data

    def arrive_at_router(self):
        assert self.state == RequestState.NONE
        self.metrics.router_arrival_timestamp = clock()
        self.state = RequestState.QUEUED_AT_ROUTER

    def arrive_at_scheduler(self):
        """
        NOTE: we don't track routing overheads
        """
        assert self.state == RequestState.QUEUED_AT_ROUTER
        self.metrics.scheduler_arrival_timestamp = clock()
        self.metrics.router_queue_time = clock() - \
                                    self.metrics.router_arrival_timestamp
        self.state = RequestState.QUEUED_AT_SCHEDULER

    def run_on_executor(self):
        assert self.state == RequestState.QUEUED_AT_SCHEDULER
        self.metrics.executor_start_timestamp = clock()
        self.metrics.scheduler_queue_time = clock() - \
                                    self.metrics.scheduler_arrival_timestamp
        self.state = RequestState.RUNNING_ON_EXECUTOR

    def complete_at_scheduler(self):
        """
        NOTE: we don't track executor <--> scheduler communication overheads
        """
        assert self.state == RequestState.RUNNING_ON_EXECUTOR
        self.metrics.scheduler_completion_timestamp = clock()
        self.metrics.service_time += clock() - \
                                    self.metrics.executor_start_timestamp
        self.metrics.scheduler_response_time = clock() - \
                                    self.metrics.scheduler_arrival_timestamp
        self.state = RequestState.COMPLETED_AT_SCHEDULER

    def complete_at_router(self):
        """
        NOTE: we don't track scheduler <--> router communication overheads
        """
        assert self.state == RequestState.COMPLETED_AT_SCHEDULER
        self.metrics.router_completion_timestamp = clock()
        self.metrics.router_response_time = clock() - \
                                    self.metrics.router_arrival_timestamp
        self.state = RequestState.COMPLETED_AT_ROUTER

    def abort(self):
        if self.state == RequestState.QUEUED_AT_ROUTER:
            self.metrics.router_queue_time += clock() - \
                                    self.metrics.router_arrival_timestamp
        elif self.state == RequestState.QUEUED_AT_SCHEDULER:
            self.metrics.scheduler_queue_time += clock() - \
                                    self.metrics.scheduler_arrival_timestamp
        elif self.state == RequestState.RUNNING_ON_EXECUTOR:
            self.metrics.service_time += clock() - \
                                    self.metrics.executor_start_timestamp
        elif self.state == RequestState.COMPLETED_AT_SCHEDULER:
            pass
        self.state = RequestState.ABORTED

    def get_results(self):
        pass

    def create_task(self, task_type, **kwargs):
        """
        Creates a Task and adds it to the DAG.
        """
        task = Task.from_type(task_type=task_type,
                              node_id=next(self.node_id),
                              request=self,
                              **kwargs)
        self.dag.add_node(task)
        self.nodes[task.node_id] = task
        return task

    def create_flow(self, flow_type, **kwargs):
        """
        Creates a Flow and adds it to the DAG.
        """
        flow = Flow.from_type(flow_type=flow_type,
                              node_id=next(self.node_id),
                              request=self,
                              **kwargs)
        self.dag.add_node(flow)
        self.nodes[flow.node_id] = flow
        return flow

    def remove_node(self, node):
        """
        Removes a Node from the DAG.
        """
        self.dag.remove_node(node)
        del self.nodes[node.node_id]

    @classmethod
    def from_dict(cls, request_dict):
        """
        Returns a Request from a Pandas dictionary.
        """
        if request_dict["request_type"] == RequestType.GENERATIVE_LLM:
            request = GenerativeLLMRequest(**request_dict)
        else:
            raise ValueError(f"Unsupported request type: {request_dict['request_type']}")
        return request


@dataclass(kw_only=True)
class GenerativeLLMRequest(Request):
    """
    GenerativeLLMRequests are requests that generate tokens from a prompt.
    Prompt processing and token generation are represented as Tasks.
    KV-cache shipping is represented using Flows.
    NOTE: Assumes that KV-cache is uniformly split across all GPUs.
    NOTE: Multi-prompt chat conversations are not supported here.
    """
    max_seq_len: int = 0
    processed_tokens: int
    _processed_tokens: int = 0
    generated_tokens: int
    _generated_tokens: int = 0
    prompt_size: int = 0
    token_size: int = 0
    kv_cache_size: int = 0
    flow_node: Flow = None
    cost: float = 0.
    memory: float = 0.
    metrics: GenerativeLLMRequestMetrics = field(
        default_factory=GenerativeLLMRequestMetrics)

    def __post_init__(self):
        self.max_seq_len = self.prompt_size + self.token_size
        # create prompt and token tasks
        prompt_task = self.create_task(task_type=TaskType.PROMPT,
                                       prompt_size=self.prompt_size)
        token_task = self.create_task(task_type=TaskType.TOKEN,
                                      token_size=self.token_size - 1)
        # update DAG
        self.dag.add_edge(prompt_task, token_task)
        self.root_node = prompt_task

    def __hash__(self):
        return hash(self.request_id)

    @property
    def processed_tokens(self):
        """
        Returns the number of prompt tokens processed so far.
        """
        return self._processed_tokens

    @processed_tokens.setter
    def processed_tokens(self, processed_tokens):
        """
        Sets the number of prompt tokens processed so far.
        """
        if isinstance(processed_tokens, property):
            processed_tokens = 0
        if processed_tokens > self.prompt_size + self.token_size:
            print(processed_tokens, self.prompt_size + self.token_size)
            raise ValueError("Processed tokens limit exceeded")
        self._processed_tokens = processed_tokens

    @property
    def generated_tokens(self):
        """
        Returns the number of tokens generated so far.
        """
        return self._generated_tokens

    @generated_tokens.setter
    def generated_tokens(self, generated_tokens):
        """
        Sets the number of tokens generated so far.
        """
        if isinstance(generated_tokens, property):
            generated_tokens = 0
        if generated_tokens > self.max_seq_len:
            raise ValueError("Maximum sequence length exceeded")
        self._generated_tokens = generated_tokens


    def estimate_kv_cache_size(self, num_tokens=None, model=None):
        """
        Returns the KV-cache size after generating num_tokens
        Requires the Request root node to be allocated on an Instance.
        """
        if num_tokens is None:
            num_tokens = self.generated_tokens
        if model is None:
            model = self.root_node.instance.model
        return 2 * self.batch_size * num_tokens * model.architecture.hidden_size \
                * model.architecture.num_layers * model.size.dtype_size

    def get_nth_token_overhead(self):
        """
        Returns the overhead of generating the nth token.
        """
        return self.nodes[1].metrics.start_timestamp - self.nodes[0].metrics.completion_timestamp

@dataclass(kw_only=True)
class GenerativeMoERequest(Request):
    """
    GenerativeMoERequest are requests that generate tokens from a prompt.
    Attention and expert layers are disaggregated.
    Activation shipping is represented using Flows.
    NOTE: Assumes that KV-cache is uniformly split across all GPUs.
    NOTE: Multi-prompt chat conversations are not supported here.
    """
    max_seq_len: int = 0
    processed_tokens: int
    _processed_tokens: int = 0
    generated_tokens: int
    _generated_tokens: int = 0
    prompt_size: int = 0
    token_size: int = 0
    kv_cache_size: int = 0
    n_layers: int = 0 # TODO (keisuke): Number of layers in the MoE model
    n_experts: int = 8 # Number of experts in the MoE model
    flow_node: Flow = None
    cost: float = 0.
    memory: float = 0.
    metrics: GenerativeLLMRequestMetrics = field(
        default_factory=GenerativeLLMRequestMetrics)

    def __post_init__(self):
        self.max_seq_len = self.prompt_size + self.token_size
        # create prompt and token tasks
        # TODO (keisuke): determine number of tokens for each based on expert popularity info
        # attention_task = self.create_task(task_type=TaskType.ATTENTION,
        #                                prompt_size=self.prompt_size)
        # expert_task = self.create_task(task_type=TaskType.EXPERT,
        #                               token_size=self.token_size - 1)
        # # update DAG
        # self.dag.add_edge(attention_task, expert_task)
        # self.root_node = attention_task

        # TODO (keisuke): now selecting the expert based on the popularity info here, 
        # but ideally we want the simulator to be deterministic
        np.random.seed(0)
        expert_popularity_filename = "./data/expert_popularity_college_computer_science.json"
        with open(expert_popularity_filename, "r") as f:
            expert_popularity = json.load(f)
            expert_selection_prob = np.array(expert_popularity["expert_popularity"]) # (32, 8)
            # normalize
            expert_selection_prob = expert_selection_prob / expert_selection_prob.sum(axis=1, keepdims=True)
        
        # we will generate the DAG involving all attention and expert layers for both prompt and token phases
        # first do prompt phase
        expert_tasks = []
        # TODO: double check the size argument to task
        for i in range(self.n_layers):
            attention_task = self.create_task(task_type=TaskType.ATTENTION,
                                           prompt_size=self.prompt_size,
                                           layer_id=i,
                                           is_prompt=True
                                           )
            if i == 0:
                self.root_node = attention_task
            # TODO (keisuke): determine number of tokens for each based on expert popularity info, for now just equal split to 8 experts
            expert_selection_prompt = np.random.choice(8, size=(self.prompt_size, 2), p=expert_selection_prob[i])
            # count the appearance of each value (0-7) in the `expert_selection_prompt`
            expert_selection_prompt_count = np.zeros(8)
            for j in range(8):
                expert_selection_prompt_count[j] = np.sum(expert_selection_prompt == j)

            for j in range(self.n_experts):
                expert_task = self.create_task(task_type=TaskType.EXPERT,
                                              prompt_size=expert_selection_prompt_count[j],
                                              layer_id=i,
                                              expert_id=j,
                                              is_prompt=True
                                              )
                self.dag.add_edge(attention_task, expert_task)
                expert_tasks.append(expert_task)
        # then do token phase
        for token_id in range(self.token_size):
            for layer_id in range(self.n_layers):
                attention_task = self.create_task(task_type=TaskType.ATTENTION,
                                               token_size=1,
                                               layer_id=layer_id,
                                               is_prompt=False
                                               )
                if len(expert_tasks) != 0:
                    for expert_task in expert_tasks:
                        self.dag.add_edge(expert_task, attention_task)
                
                # draw two random numbers for expert ids
                expert_tasks = np.random.choice(8, size=(1, 2), p=expert_selection_prob[i])[0]
                expert_id_1 = expert_tasks[0]
                expert_id_2 = expert_tasks[1]
                expert_task_1 = self.create_task(task_type=TaskType.EXPERT,
                                                token_size=1,
                                                layer_id=layer_id,
                                                expert_id=expert_id_1,
                                                is_prompt=False
                                                )
                expert_task_2 = self.create_task(task_type=TaskType.EXPERT,
                                                token_size=1,
                                                layer_id=layer_id,
                                                expert_id=expert_id_2,
                                                is_prompt=False
                                                )
                self.dag.add_edge(attention_task, expert_task_1)
                self.dag.add_edge(attention_task, expert_task_2)
                
    def __hash__(self):
        return hash(self.request_id)

    @property
    def processed_tokens(self):
        """
        Returns the number of prompt tokens processed so far.
        """
        return self._processed_tokens

    @processed_tokens.setter
    def processed_tokens(self, processed_tokens):
        """
        Sets the number of prompt tokens processed so far.
        """
        if isinstance(processed_tokens, property):
            processed_tokens = 0
        if processed_tokens > self.prompt_size + self.token_size:
            print(processed_tokens, self.prompt_size + self.token_size)
            raise ValueError("Processed tokens limit exceeded")
        self._processed_tokens = processed_tokens

    @property
    def generated_tokens(self):
        """
        Returns the number of tokens generated so far.
        """
        return self._generated_tokens

    @generated_tokens.setter
    def generated_tokens(self, generated_tokens):
        """
        Sets the number of tokens generated so far.
        """
        if isinstance(generated_tokens, property):
            generated_tokens = 0
        if generated_tokens > self.max_seq_len:
            raise ValueError("Maximum sequence length exceeded")
        self._generated_tokens = generated_tokens


    def estimate_kv_cache_size(self, num_tokens=None, model=None):
        """
        Returns the KV-cache size after generating num_tokens
        Requires the Request root node to be allocated on an Instance.
        """
        if num_tokens is None:
            num_tokens = self.generated_tokens
        if model is None:
            model = self.root_node.instance.model
        return 2 * self.batch_size * num_tokens * model.architecture.hidden_size \
                * model.architecture.num_layers * model.size.dtype_size

    def get_nth_token_overhead(self):
        """
        Returns the overhead of generating the nth token.
        """
        return self.nodes[1].metrics.start_timestamp - self.nodes[0].metrics.completion_timestamp



if __name__ == "__main__":
    pass
