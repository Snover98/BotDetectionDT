import itertools


def get_subrun_name(run_name: str, use_gdelt: bool, use_TCN: bool):
    temporal_ext_name = "TCN" if use_TCN else "LSTM"
    subrun_name = f"{run_name}_{temporal_ext_name}"
    if use_gdelt:
        subrun_name += "_GDELT"
    
    return subrun_name


def get_all_subrun_names(run_name: str):
    p = itertools.product([False, True], [False, True])
    return tuple([get_subrun_name(run_name, use_gdelt, use_TCN) for use_gdelt, use_TCN in p])





