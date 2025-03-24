from zenml.steps import BaseParameters

class ModelnameConfig(BaseParameters):
    """"Model Configs"""
    model_name: str = "LinearRegression"