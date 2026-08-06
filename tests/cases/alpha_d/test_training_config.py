from pathlib import Path

from omegaconf import OmegaConf

from cases.alpha_d.study_workflow import ALPHA_INPUT_COLUMNS
from training.hpo.config import validate_hpo_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "src/cases/alpha_d/configs"


def _load(name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / name), resolve=False)


def test_conv1d_uses_curated_direct_features_and_clean_search_contract():
    config = _load("train_conv1d.yaml")

    assert config["data"]["input_columns"] == ALPHA_INPUT_COLUMNS
    assert config["data"]["input_columns_file"] is None
    assert config["data"]["include_acceleration_head"] is True
    assert config["data"]["downstream_weight"] == 3.0
    assert all(
        control["params"]["data.downstream_weight"] == 3.0
        for control in config["hpo"]["enqueue_trials"]
    )
    assert config["training"]["loss"] == "relative_l2"
    validate_hpo_config(config["hpo"])

    search = config["hpo"]["search_space"]
    assert search["training.lr"] == {
        "type": "float",
        "low": 1.0e-4,
        "high": 4.0e-4,
        "log": True,
    }
    assert search["training.batch_size"]["choices"] == [16, 32]
    assert search["model.params.hidden_channels"]["choices"] == [64, 96]
    assert search["model.params.num_blocks"]["choices"] == [2, 3]
    assert search["model.params.kernel_size"]["choices"] == [3, 5]
    assert search["data.throat_weight"]["choices"] == [3.0, 5.0]
    assert "data.downstream_weight" not in search


def test_conv1d_hpo_includes_three_scientific_controls():
    config = _load("train_conv1d.yaml")
    controls = {row["name"]: row["params"] for row in config["hpo"]["enqueue_trials"]}

    assert set(controls) == {"july_7", "alpha_d_conv1d_001", "hybrid"}
    assert controls["july_7"]["training.lr"] == 0.00031227062983797785
    assert controls["alpha_d_conv1d_001"]["model.params.kernel_size"] == 5
    assert controls["hybrid"]["model.params.kernel_size"] == 5
    assert controls["hybrid"]["training.lr"] == controls["july_7"]["training.lr"]
    assert config["hpo"]["screening"]["n_trials"] == 40
    assert config["hpo"]["confirmation"]["top_k"] == 5


def test_conv1d_pycaret_config_selects_a_frozen_nine_feature_profile():
    config = _load("pycaret_conv1d.yaml")

    assert config["data"]["selected_from_allowlist"] is None
    assert config["data"]["min_Dr"] == 0.333
    assert config["pycaret"]["setup"]["n_features_to_select"] == 9


def test_mlp_migrates_to_clean_hpo_and_acceleration_contract():
    config = _load("train_mlp.yaml")

    assert config["data"]["include_acceleration_head"] is True
    assert config["data"]["target_transform_kwargs"] == {
        "include_acceleration_head": "${data.include_acceleration_head}"
    }
    validate_hpo_config(config["hpo"])
