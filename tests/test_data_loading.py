import pytest

from text2sql.data_loading import validate_adapter_ref, validate_dataset_ref


@pytest.mark.parametrize(
    "ref",
    [None, "", "your-hf-username/sql-create-context-clean", "<hf-user>/sql-create-context-clean"],
)
def test_placeholder_dataset_rejected(ref):
    with pytest.raises(ValueError, match="dataset is not set"):
        validate_dataset_ref(ref)


@pytest.mark.parametrize("ref", ["naveen/sql-create-context-clean", "data/processed"])
def test_real_dataset_accepted(ref):
    assert validate_dataset_ref(ref) == ref


def test_existing_local_adapter_accepted(tmp_path):
    assert validate_adapter_ref(str(tmp_path)) == str(tmp_path)


def test_hub_adapter_accepted():
    assert validate_adapter_ref("someone/llama32-3b-text2sql-lora") == "someone/llama32-3b-text2sql-lora"


@pytest.mark.parametrize(
    "ref",
    ["outputs/smoke-llama32-3b/final", "outputs/final", "./final", "/content/x/final", "a\\b", "plainname"],
)
def test_missing_local_adapter_rejected(ref):
    with pytest.raises(FileNotFoundError, match="did training finish|Did training finish"):
        validate_adapter_ref(ref)
