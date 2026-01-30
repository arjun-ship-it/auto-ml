from typing import Any
from google.genai import types

from app.ml.eda import run_eda
from app.ml.preprocessing import preprocess_data
from app.ml.training import train_model
from app.ml.evaluation import evaluate_model
from app.ml.prediction import generate_predictions
from app.data.connectors import query_database, load_dataset
from app.execution.runner import execute_code_safely


# Tool definitions for Gemini API (google-genai)
GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="analyze_data",
                description="Perform exploratory data analysis on a dataset.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "dataset_id": types.Schema(type=types.Type.STRING, description="Dataset ID"),
                        "target_column": types.Schema(type=types.Type.STRING, description="Target column"),
                        "analysis_type": types.Schema(type=types.Type.STRING, enum=["full", "summary", "correlations", "missing_values", "distributions"]),
                    },
                    required=["dataset_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="preprocess_data",
                description="Clean and preprocess the dataset.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "dataset_id": types.Schema(type=types.Type.STRING),
                        "steps": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                        "config": types.Schema(type=types.Type.OBJECT),
                    },
                    required=["dataset_id", "steps"],
                ),
            ),
            types.FunctionDeclaration(
                name="train_model",
                description="Train a machine learning model.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "dataset_id": types.Schema(type=types.Type.STRING),
                        "model_type": types.Schema(type=types.Type.STRING, enum=["linear_regression", "logistic_regression", "random_forest", "xgboost", "lightgbm", "svm", "knn", "neural_network"]),
                        "target_column": types.Schema(type=types.Type.STRING),
                        "hyperparameters": types.Schema(type=types.Type.OBJECT),
                        "test_size": types.Schema(type=types.Type.NUMBER),
                    },
                    required=["dataset_id", "model_type", "target_column"],
                ),
            ),
            types.FunctionDeclaration(
                name="evaluate_model",
                description="Evaluate model performance.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "model_id": types.Schema(type=types.Type.STRING),
                        "metrics": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                    },
                    required=["model_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="predict",
                description="Generate predictions using a trained model.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "model_id": types.Schema(type=types.Type.STRING),
                        "input_data": types.Schema(type=types.Type.OBJECT),
                        "prediction_horizon": types.Schema(type=types.Type.INTEGER),
                        "include_confidence": types.Schema(type=types.Type.BOOLEAN),
                    },
                    required=["model_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="query_database",
                description="Execute a SQL query.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "connection_id": types.Schema(type=types.Type.STRING),
                        "query": types.Schema(type=types.Type.STRING),
                        "limit": types.Schema(type=types.Type.INTEGER),
                    },
                    required=["connection_id", "query"],
                ),
            ),
            types.FunctionDeclaration(
                name="execute_code",
                description="Execute Python code in sandbox.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "code": types.Schema(type=types.Type.STRING),
                        "context": types.Schema(type=types.Type.OBJECT),
                    },
                    required=["code"],
                ),
            ),
        ]
    )
]


async def execute_tool(tool_name: str, tool_input: dict, project_id: str) -> Any:
    handlers = {
        "analyze_data": _handle_analyze_data,
        "preprocess_data": _handle_preprocess_data,
        "train_model": _handle_train_model,
        "evaluate_model": _handle_evaluate_model,
        "predict": _handle_predict,
        "query_database": _handle_query_database,
        "execute_code": _handle_execute_code,
    }
    handler = handlers.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return await handler(tool_input, project_id)
    except Exception as e:
        return {"error": str(e)}


async def _handle_analyze_data(input: dict, project_id: str) -> dict:
    return await run_eda(dataset_id=input["dataset_id"], project_id=project_id, target_column=input.get("target_column"), analysis_type=input.get("analysis_type", "full"))


async def _handle_preprocess_data(input: dict, project_id: str) -> dict:
    return await preprocess_data(dataset_id=input["dataset_id"], project_id=project_id, steps=input["steps"], config=input.get("config", {}))


async def _handle_train_model(input: dict, project_id: str) -> dict:
    return await train_model(dataset_id=input["dataset_id"], project_id=project_id, model_type=input["model_type"], target_column=input["target_column"], hyperparameters=input.get("hyperparameters", {}), test_size=input.get("test_size", 0.2))


async def _handle_evaluate_model(input: dict, project_id: str) -> dict:
    return await evaluate_model(model_id=input["model_id"], project_id=project_id, metrics=input.get("metrics"))


async def _handle_predict(input: dict, project_id: str) -> dict:
    return await generate_predictions(model_id=input["model_id"], project_id=project_id, input_data=input.get("input_data"), prediction_horizon=input.get("prediction_horizon"), include_confidence=input.get("include_confidence", True))


async def _handle_query_database(input: dict, project_id: str) -> dict:
    return await query_database(connection_id=input["connection_id"], query=input["query"], limit=input.get("limit", 1000))


async def _handle_execute_code(input: dict, project_id: str) -> dict:
    return await execute_code_safely(code=input["code"], project_id=project_id, context=input.get("context", {}))
