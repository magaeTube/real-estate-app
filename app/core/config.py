def get_config_info():
    import yaml
    with open("app/core/config.yaml", "r") as file:
        config_data = yaml.safe_load(file)

    return config_data
