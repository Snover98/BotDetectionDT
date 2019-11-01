
def get_TCN_params_from_effective_history(history_len: int):
    # Effective history formula: 1 + 2*(kernel_size-1)*(2^num_levels-1)
    if history_len <= 7:
        # effective history: 7
        kernel_size = 2
        num_levels = 2
    elif 7 <= history_len <= 15:
        # effective history: 15
        kernel_size = 2
        num_levels = 3
    elif 15 <= history_len <= 57:
        # effective history: 57
        kernel_size = 5
        num_levels = 3
    elif 57 <= history_len <= 91:
        # effective history: 91
        kernel_size = 4
        num_levels = 4
    elif 91 <= history_len <= 121:
        # effective history: 121
        kernel_size = 5
        num_levels = 4
    elif 121 <= history_len <= 249:
        # effective history: 249
        kernel_size = 5
        num_levels = 5
    elif 249 <= history_len <= 311:
        # effective history: 311
        kernel_size = 6
        num_levels = 5
    else:
        # effective history: 505
        kernel_size = 5
        num_levels = 6

    return num_levels, kernel_size
