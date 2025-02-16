import numpy as np
import pandas as pd
import os
def compute_dynamic_values_to_code(data, sensitive_attributes):
    value_to_code = {}
    current_code = 1

    # For each sensitive attribute
    for s_attribute in sorted(sensitive_attributes):
        unique_groups = sorted(set(data[s_attribute].unique()))

        # Assign code to independent groups
        for group in unique_groups:
            if group not in value_to_code:
                value_to_code[group] = current_code
                current_code += 1

    return value_to_code




def process_data_input(data, configs, dataset):
    features_col = dataset.feature_cols + [dataset.score_col]

    if "unprivileged_groups" in configs:
        sensitive_attributes = set(configs["unprivileged_groups"]) | set(configs["privileged_groups"])
        value_to_code = compute_dynamic_values_to_code(data, sensitive_attributes)

        unprivileged_groups, privileged_groups, group_weights = {}, {}, {}
        # Encode sensitive attributes
        for s_attribute in sensitive_attributes:
            coded_attribute = s_attribute + '_coded'
            features_col.append(coded_attribute)

            data[coded_attribute] = data[s_attribute].map(value_to_code)

            unprivileged_groups[s_attribute] = [value_to_code[group] for group in
                                                configs["unprivileged_groups"].get(s_attribute, [])]
            privileged_groups[s_attribute] = [value_to_code[group] for group in
                                              configs["privileged_groups"].get(s_attribute, [])]

            # Update group weights
            for group in value_to_code:
                if group in data[s_attribute].unique() and value_to_code[group] not in group_weights:
                    group_weights[value_to_code[group]] = configs['group_weights'].get(group, 0)

        # Store sensitive groups
        sensitive_groups = {"unprivileged_groups": unprivileged_groups, "privileged_groups": privileged_groups}
    else:
        sensitive_attributes = dataset.sensitive_cols
        value_to_code = compute_dynamic_values_to_code(data, sensitive_attributes)
        for s_attribute in sensitive_attributes:
            coded_attribute = s_attribute + '_coded'
            features_col.append(coded_attribute)

            data[coded_attribute] = data[s_attribute].map(value_to_code)

        group_weights = None
        sensitive_groups = None

    # Identify indices of sensitive columns
    sensitive_column_indices = [
        list(data[features_col].columns).index(s_attribute + "_coded") for s_attribute in sensitive_attributes
    ]

    nonsensitive_column_indices = [index for index in range(0, len(features_col)) if
                                   index not in sensitive_column_indices]
    data_processed = data[features_col].to_numpy()
    return data_processed, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices


def process_data_output(data_orig, data_fair, dataset, nonsensitive_column_indices, fair_data_path, file_name=None):
    # Combine relevant columns from configs
    features_col = dataset.feature_cols + [dataset.score_col]

    # Convert data_fair to a numpy array and select relevant columns
    data_fair = np.vstack(data_fair)
    fair_features_col = [f'{col}_fair' for col in features_col]
    selected_fair_features_col = [fair_features_col[i] for i in nonsensitive_column_indices]

    # Create DataFrame for fair data with selected columns
    data_fair = pd.DataFrame(data_fair[:, nonsensitive_column_indices], columns=selected_fair_features_col)

    # Add original data columns to the fair data
    for col in data_orig.columns:
        data_fair[col] = data_orig[col].values
        data_fair[col] = data_orig[col].values

    if file_name != None:
        os.makedirs(fair_data_path, exist_ok=True)
        data_fair.to_csv(os.path.join(fair_data_path, f'fair_{file_name}_data.csv'))

    return data_fair


def save_model_data(model, path):
    with open(os.path.join(path, 'model_parmas.npy'), 'wb') as f:
        np.save(f, model.opt_params)


def load_model_data(path):
    with open(os.path.join(path, 'model_parmas.npy'), 'rb') as f:
       return np.load(f)

