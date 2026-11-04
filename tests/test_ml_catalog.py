from src.forecasting.features import FEATURE_COLUMNS
from src.ui.ml_data import FEATURE_DEFINITIONS, feature_catalog


def test_visible_feature_catalog_matches_real_model_features():
    catalog = feature_catalog()
    assert list(catalog.Feature) == FEATURE_COLUMNS
    assert set(FEATURE_DEFINITIONS) == set(FEATURE_COLUMNS)
    assert catalog["Past/present only"].all()
