"""Module with merge functions."""


def merge_dicts(dict1, dict2, verbose=False):
    """Deeply merge two dicts.

    The second dict take precedence over the first dict. This means that if the two dicts
    has the same key, the value stored in that key will be the second value.
    Anyway, if the values are dicts too it will call recursively this function to deeply merge
    the two dict values.

    Arguments:
        dict1 (dict): One of the dictionaries to merge.
        dict2 (dict): One of the dictionaries to merge.

    Returns:
        dict: The deeply merged dict.
    """
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                dict1[key] = merge_dicts(dict1[key], dict2[key])
            elif dict1[key] == dict2[key]:
                pass  # same leaf value
            else:
                if verbose:
                    print('Replacing "{}" with value "{}" for "{}"'.format(key, dict1[key], dict2[key]))
                dict1[key] = dict2[key]
        else:
            if verbose:
                print('Adding "{}" with value "{}"'.format(key, dict2[key]))
            dict1[key] = dict2[key]
    return dict1
