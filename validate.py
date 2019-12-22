
class ConfigError(Exception):
    pass
class HyperParamError(Exception):
    pass
class DataPointError(Exception):
    pass

def validate(current_data):
    vars = current_data['dim_ranges']

    #validate dim_ranges
    try:
        for item in vars.values():
            if item['type'] == "discrete":
                if not isinstance(item['possibilities'], int):
                    raise ConfigError()
    except KeyError:
        raise ConfigError()

    disc_vars = [var for var,item in vars.items() if item['type'] == "discrete"]
    contin_vars = [var for var,item in vars.items() if item['type'] == "continuous"]

    if set(contin_vars) != set(vars['guassian_params']):
        raise HyperParamError()
