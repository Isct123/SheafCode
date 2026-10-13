import inspect
from argparse import Namespace, ArgumentParser
from typing import Union


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'off'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y', 'on'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def config_dict_from_args(args):
       
                                                                                                                  

                            
                          
       
    keys_to_remove = {'hpc_exp_number', 'trials', 'optimize_parallel', 'optimize_parallel_gpu',
                      'optimize_parallel_cpu', 'generate_trials', 'optimize_trials_parallel_gpu'}
    hparams = {key: v for key, v in args.__dict__.items() if key not in keys_to_remove}
    return hparams


def update_from_config(args: Namespace, config: dict):
    assert set(config.keys()) <= set(vars(args)), f'{set(config.keys()).difference(vars(args))} not in args.'
    args.__dict__.update(config)
    return args


def parse_by_group(parser):
       
                                                                              
                                                             

                          
                             
            
       
    assert isinstance(parser, ArgumentParser)
    args = parser.parse_args()

                                                                                       
    pos_group, optional_group = parser._action_groups[0], parser._action_groups[1]
    args_dict = args._get_kwargs()
    pos_optional_arg_names = [arg.dest for arg in pos_group._group_actions] + [arg.dest for arg in
                                                                               optional_group._group_actions]
    pos_optional_args = {name: value for name, value in args_dict if name in pos_optional_arg_names}
    other_group_args = dict()

                                                                            
    if len(parser._action_groups) > 2:
        for group in parser._action_groups[2:]:
            group_arg_names = [arg.dest for arg in group._group_actions]
            other_group_args[group.title] = Namespace(
                **{name: value for name, value in args_dict if name in group_arg_names})

                                                            
    combined_args = pos_optional_args
    combined_args.update(other_group_args)
    return Namespace(flat=args, **combined_args)


def filter_args(args: Union[Namespace, dict], target_cls, return_dict=False):
    argspec = inspect.getfullargspec(target_cls.__init__)
    target_args = argspec.args
    if isinstance(args, Namespace):
        args = vars(args)
    filtered_args = {k: args[k] for k in target_args if k in args}
    if return_dict:
        return filtered_args
    return Namespace(**filtered_args)


def filter_function_args(args: Union[Namespace, dict], function, return_dict=False):
    argspec = inspect.getfullargspec(function)
    target_args = argspec.args
    if isinstance(args, Namespace):
        args = vars(args)
    filtered_args = {k: args[k] for k in target_args if k in args}
    if return_dict:
        return filtered_args
    return Namespace(**filtered_args)
