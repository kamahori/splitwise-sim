"""
Utility functions to initialize the Cluster with a starting state.
"""

import logging

from model import ModelParallelism
from simulator import clock, schedule_event, cancel_event, reschedule_event


def load_start_state(start_state_cfg, **kwargs):
    """
    Load the start state configuration and initialize the cluster.
    """
    state_type = start_state_cfg.state_type
    if state_type == "unallocated":
        pass
    elif state_type == "orca":
        uniform(start_state_cfg, **kwargs)
    elif state_type == "baseline":
        uniform(start_state_cfg, **kwargs)
    elif "splitwise" in state_type:
        splitwise(start_state_cfg, **kwargs)
    elif "moe" in state_type:
        moe(start_state_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown start state type: {state_type}")


def uniform(start_state_cfg, cluster, applications, **kwargs):
    """
    Initialize all servers with a single instance of the application.
    """
    application = applications[start_state_cfg.application_id]
    allocator = application.allocator
    servers = cluster.servers

    instance_cfg = start_state_cfg.instance
    parallelism = ModelParallelism(pipeline_parallelism=instance_cfg.pipeline_parallelism,
                                   tensor_parallelism=instance_cfg.tensor_parallelism)

    for sku_name in servers:
        for server in servers[sku_name]:
            allocator.start_spin_up_instance(instance_cfg=instance_cfg,
                                             processors=server.processors,
                                             parallelism=parallelism,
                                             pre_start=True)


def splitwise(start_state_cfg, cluster, applications, **kwargs):
    """
    Initialize all servers with a single instance of the application.
    Separate prompt and token instances with different kinds of parallelism.
    TODO: use preferences and constraints within scheduler instead
    """
    application = applications[start_state_cfg.application_id]
    allocator = application.allocator
    servers = cluster.servers

    prompt_cfg = start_state_cfg.prompt
    token_cfg = start_state_cfg.token
    prompt_parallelism = ModelParallelism(pipeline_parallelism=prompt_cfg.pipeline_parallelism,
                                          tensor_parallelism=prompt_cfg.tensor_parallelism)
    token_parallelism = ModelParallelism(pipeline_parallelism=token_cfg.pipeline_parallelism,
                                         tensor_parallelism=token_cfg.tensor_parallelism)

    split_type = start_state_cfg.split_type

    if split_type == "homogeneous":
        n_prompts = prompt_cfg.num_instances
        n_tokens = token_cfg.num_instances
        # allocate n_prompt instance of prompt
        all_servers = [server for sku_name in servers for server in servers[sku_name]]
        for server in all_servers[:n_prompts]:
            for proc_id in range(0, len(server.processors), prompt_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                 processors=server.processors[proc_id:proc_id+prompt_parallelism.tensor_parallelism],
                                                 parallelism=prompt_parallelism,
                                                 pre_start=True,
                                                 tag="prompt")
        for server in all_servers[n_prompts:n_prompts+n_tokens]:
            for proc_id in range(0, len(server.processors), token_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                 processors=server.processors[proc_id:proc_id+token_parallelism.tensor_parallelism],
                                                 parallelism=token_parallelism,
                                                 pre_start=True,
                                                 tag="token")

    if split_type == "heterogeneous":
        prompt_instances = prompt_cfg.instance_names
        token_instances = token_cfg.instance_names
        for sku_name in servers:
            for server in servers[sku_name]:
                if sku_name in prompt_instances:
                    # allocate as many prompt instances as possible
                    for proc_id in range(0, len(server.processors), prompt_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                         processors=server.processors[proc_id:proc_id+prompt_parallelism.tensor_parallelism],
                                                         parallelism=prompt_parallelism,
                                                         pre_start=True,
                                                         tag="prompt")
                elif sku_name in token_instances:
                    # allocate as many token instances as possible
                    for proc_id in range(0, len(server.processors), token_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                         processors=server.processors[proc_id:proc_id+token_parallelism.tensor_parallelism],
                                                         parallelism=token_parallelism,
                                                         pre_start=True,
                                                         tag="token")
                else:
                    raise ValueError(f"Unsupported sku_name: {sku_name}")


def moe(start_state_cfg, cluster, applications, **kwargs):
    application = applications[start_state_cfg.application_id]
    allocator = application.allocator
    servers = cluster.servers

    attention_cfg = start_state_cfg.attention
    expert_cfg = start_state_cfg.expert
    attention_parallelism = ModelParallelism(pipeline_parallelism=attention_cfg.pipeline_parallelism,
                                             tensor_parallelism=attention_cfg.tensor_parallelism)
    expert_parallelism = ModelParallelism(pipeline_parallelism=expert_cfg.pipeline_parallelism,
                                          tensor_parallelism=expert_cfg.tensor_parallelism)

    split_type = start_state_cfg.split_type

    if split_type == "homogeneous":
        n_attention = attention_cfg.num_instances
        n_expert = expert_cfg.num_instances

        all_servers = [server for sku_name in servers for server in servers[sku_name]]
        # for attention, we will assign one attention instance for one server
        for server in all_servers[:n_attention]:
            for proc_id in range(0, len(server.processors), attention_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=attention_cfg,
                                                 processors=server.processors[proc_id:proc_id+attention_parallelism.tensor_parallelism],
                                                 parallelism=attention_parallelism,
                                                 pre_start=True,
                                                 tag="attention")
        # we assign 256 experts to the number of instances n_expert
        # currently we do random assignment
        expert_instance_assignment = []
        # TODO: change 256 to number of experts
        expert_per_instance = 256 // n_expert
        # currently support instance num divisible by 256
        assert(256 % n_expert == 0)
        def flat_to_2d(index):
            if 0 <= index <= 255:
                a = index // 8
                b = index % 8
                return [a, b]
        for i in range(n_expert):
            expert_info = {}
            for j in range(expert_per_instance):
                layer_id, expert_id = flat_to_2d(i*expert_per_instance + j)
                if layer_id not in expert_info:
                    expert_info[layer_id] = []
                expert_info[layer_id].append(expert_id)
            expert_instance_assignment.append(expert_info)
            
        for i, server in enumerate(all_servers[n_attention:n_attention+n_expert]):
            for proc_id in range(0, len(server.processors), expert_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=expert_cfg,
                                                 processors=server.processors[proc_id:proc_id+expert_parallelism.tensor_parallelism],
                                                 parallelism=expert_parallelism,
                                                 pre_start=True,
                                                 expert_info=expert_instance_assignment[i],
                                                 tag="expert")

    if split_type == "heterogeneous":
        attention_instances = attention_cfg.instance_names
        expert_instances = expert_cfg.instance_names
        for sku_name in servers:
            for server in servers[sku_name]:
                if sku_name in attention_instances:
                    for proc_id in range(0, len(server.processors), attention_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=attention_cfg,
                                                        processors=server.processors[proc_id:proc_id+attention_parallelism.tensor_parallelism],
                                                        parallelism=attention_parallelism,
                                                        pre_start=True,
                                                        tag="attention")
                elif sku_name in expert_instances:
                    for proc_id in range(0, len(server.processors), expert_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=expert_cfg,
                                                        processors=server.processors[proc_id:proc_id+expert_parallelism.tensor_parallelism],
                                                        parallelism=expert_parallelism,
                                                        pre_start=True,
                                                        tag="expert")
                else:
                    raise ValueError(f"Unsupported sku_name: {sku_name}")
