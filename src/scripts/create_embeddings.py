import os

from src.utils.ml.embeddings.src.utils.task_utils import (
    create_embeddings_table_for_task,
    destination_table_name,
    insert_embeddings_table_for_task,
    load_tasks_from_yaml,
    table_name,
)


def _resolve_yaml_path() -> str:
    """Return absolute path to embeddings_tasks.yaml located at the embeddings root.

    Structure:
    embeddings/
      - embeddings_tasks.yaml
      - src/
        - scripts/create_embeddings.py  <-- this file
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, "embeddings_tasks.yaml")


if __name__ == "__main__":
    yaml_path = _resolve_yaml_path()
    tasks = load_tasks_from_yaml(yaml_path, tasks_id="tasks_earnings_transcripts")

    if not tasks and table_name and destination_table_name:
        raise ValueError(
            "No tasks found in YAML file. Please define tasks in embeddings_tasks.yaml or provide task parameters directly."
        )

    for task in tasks:
        mode = task.get("mode", "create").lower()
        if mode == "insert":
            insert_embeddings_table_for_task(task)
        else:
            create_embeddings_table_for_task(task)
