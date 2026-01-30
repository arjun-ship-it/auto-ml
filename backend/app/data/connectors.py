import pandas as pd
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine, text

from app.config import settings


async def load_dataset(
    file_path: str,
    project_id: str,
    file_type: Optional[str] = None,
) -> dict:
    """Load a dataset from a file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_type or path.suffix.lower()

    if ext in [".csv", "csv"]:
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls", "xlsx", "xls"]:
        df = pd.read_excel(path)
    elif ext in [".parquet", "parquet"]:
        df = pd.read_parquet(path)
    elif ext in [".json", "json"]:
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Save to project uploads
    upload_dir = Path(settings.UPLOAD_DIR) / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = path.stem
    save_path = upload_dir / f"{dataset_id}.csv"
    df.to_csv(save_path, index=False)

    return {
        "dataset_id": dataset_id,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(5).to_dict(orient="records"),
    }


async def query_database(
    connection_id: str,
    query: str,
    limit: int = 1000,
) -> dict:
    """Execute a SELECT query on a connected database."""
    # Security: Only allow SELECT queries
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed for safety"}

    # Block dangerous keywords
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "EXEC"]
    for keyword in dangerous:
        if keyword in query_upper:
            return {"error": f"Query contains forbidden keyword: {keyword}"}

    # Get connection string from stored connections
    connection_string = await _get_connection_string(connection_id)
    if not connection_string:
        return {"error": f"Connection {connection_id} not found"}

    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Add LIMIT if not present
            if "LIMIT" not in query_upper:
                query = f"{query} LIMIT {limit}"

            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = list(result.keys())

            df = pd.DataFrame(rows, columns=columns)

            return {
                "rows": len(df),
                "columns": columns,
                "data": df.head(limit).to_dict(orient="records"),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}


async def _get_connection_string(connection_id: str) -> Optional[str]:
    """Retrieve stored database connection string."""
    from app.db.database import get_db
    from app.db.models import DatabaseConnection
    from sqlalchemy import select

    async for db in get_db():
        stmt = select(DatabaseConnection).where(
            DatabaseConnection.id == connection_id
        )
        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
        if record:
            return record.connection_string
    return None


async def save_uploaded_file(
    file_content: bytes,
    filename: str,
    project_id: str,
) -> dict:
    """Save an uploaded file and return dataset info."""
    upload_dir = Path(settings.UPLOAD_DIR) / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_content)

    return await load_dataset(str(file_path), project_id)


async def list_datasets(project_id: str) -> list[dict]:
    """List all datasets for a project."""
    upload_dir = Path(settings.UPLOAD_DIR) / project_id
    if not upload_dir.exists():
        return []

    datasets = []
    for f in upload_dir.iterdir():
        if f.suffix in [".csv", ".xlsx", ".parquet"]:
            datasets.append({
                "dataset_id": f.stem,
                "filename": f.name,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            })

    return datasets
