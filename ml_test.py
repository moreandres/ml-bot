"""UT coverage for ml.py."""

import pandas
import pytest

import ml


@pytest.mark.parametrize(
    "inpt,data,target",
    [
        (
            {
                "label": [0.0, 1.0, 0.0],
                "feature_numerical": [1.0, 0.0, 1.0],
            },
            {
                "feature_numerical": [1.0, 0.0, 1.0],
            },
            {
                "label": [0.0, 1.0, 0.0],
            },
        )
    ],
)
def test_split_data(inpt, data, target):
    """UT coverage for split_data()"""
    actual_data, actual_target = ml.split_data(pandas.DataFrame(inpt))
    print(actual_data)
    print(actual_target)
    pandas.testing.assert_frame_equal(actual_data, pandas.DataFrame(data))
    pandas.testing.assert_frame_equal(actual_target, pandas.DataFrame(target))


@pytest.mark.parametrize(
    "inpt,output",
    [
        (
            {
                "label": [0.0, 1.0, 0.0],
                "constant_categorical": ["true", "true", "true"],
                "constant_numerical": [1.0, 1.0, 1.0],
                "feature_numerical": [1.0, 0, 1.0],
                "feature_categorical": ["good", "bad", "good"],
                "feature_dates": [
                    "2023-10-20T22:05:37.207919",
                    "2022-10-20T22:05:37.207919",
                    "2023-10-20T22:05:37.207919",
                ],
                "missing_numerical": [1.0, 1.0, None],
                "missing_categorical": ["good", "good", None],
                "feature_text": [
                    "this is hello",
                    "this is also hello",
                    "this is not",
                ],
            },
            {
                "label": [0.0, 1.0, 0.0],
                "feature_numerical": [1.0, 0.0, 1.0],
                "feature_dates_year": [1.0, 0.0, 1.0],
                "feature_dates_weekday": [1.0, 0.0, 1.0],
                "feature_dates_since": [0.0, 1.0, 0.0],
                "feature_categorical_bad": [0.0, 1.0, 0.0],
                "feature_categorical_good": [1.0, 0.0, 1.0],
                "feature_text_wc": [0.0, 1.0, 0.0],
                "feature_text_hello": [1.0, 1.0, 0.0],
            },
        )
    ],
)
def test_preprocess(inpt, output):
    """UT coverage for preprocess() with valid data."""
    expected = pandas.DataFrame(output)
    actual = ml.preprocess(pandas.DataFrame(inpt))
    pandas.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "inpt,label,expected",
    [
        (
            {
                "feature_numerical": [1.0, 0.0, 1.0],
            },
            {
                "label": [0.0, 1.0, 0.0],
            },
            1.0,
        )
    ],
)
def test_predict(inpt, label, expected):
    """Test predict()."""
    actual, _ = ml.predict(pandas.DataFrame(inpt), pandas.DataFrame(label))
    assert actual == expected, "actual accuracy should match expected"


@pytest.mark.parametrize(
    "inpt,label,expected",
    [
        (
            "t/mediaciones.csv.bz2",
            "reapertura",
            0.9408,
        ),
        (
            "t/abalone.csv.bz2",
            "rings",
            0.26029,
        ),  # Abalone Age Prediction Using Machine Learning
        ("t/ad.csv.bz2", "class", 0.97073),
        (
            "t/adult.csv.bz2",
            "income",
            0.8472,
        ),  # https://www.cs.toronto.edu/~delve/data/adult/desc.html
        (
            "t/bank.csv.bz2",
            "y",
            0.95536,
        ),  # https://archive.ics.uci.edu/dataset/222/bank+marketing
        ("t/cancer.csv.bz2", "diagnosis", 0.98601),
        ("t/car.csv.bz2", "class", 0.97685),
        (
            "t/defects.csv.bz2",
            "DefectStatus",
            0.96173,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset
        (
            "t/fire.csv.bz2",
            "STATUS",
            0.96927,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/acoustic-extinguisher-fire-dataset
        (
            "t/flights.csv.bz2",
            "satisfaction_v2",
            0.99952,
        ),  # https://www.kaggle.com/datasets/johndddddd/customer-satisfaction
        (
            "t/fruit.csv.bz2",
            "Class",
            0.92444,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets
        (
            "t/gaming.csv.bz2",
            "EngagementLevel",
            0.88928,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
        ("t/heart.csv.bz2", "num", 0.63478),
        (
            "t/hepatitis.csv.bz2",
            "Category",
            0.96104,
        ),  # https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset
        (
            "t/iris.csv.bz2",
            "target",
            1.0,
        ),  # https://archive.ics.uci.edu/dataset/53/iris
        (
            "t/mushroom.csv.bz2",
            "class",
            1.0,
        ),  # https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset
        (
            "t/pistachio.csv.bz2",
            "Class",
            0.92179,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/pistachio-dataset
        (
            "t/pumpkin.csv.bz2",
            "Class",
            0.8752,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset
        (
            "t/stars.csv.bz2",
            "class",
            0.97376,
        ),  # https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
        (
            "t/stroke.csv.bz2",
            "stroke",
            0.9374,
        ),  # https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
        (
            "t/students.csv.bz2",
            "GradeClass",
            0.90468,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
        (
            "t/titanic.csv.bz2",
            "Survived",
            0.84753,
        ),  # Will Cukierski. (2012). Titanic - Machine Learning from Disaster. Kaggle. https://kaggle.com/competitions/titanic
        ("t/titanic.eml", "Survived", 0.84753),
        ("t/titanic.xlsx", "Survived", 0.84753),
        (
            "t/water.csv.bz2",
            "Potability",
            0.6801,
        ),  # https://www.kaggle.com/datasets/adityakadiwal/water-potability
        (
            "t/weather.csv.bz2",
            "RainTomorrow",
            0.844,
        ),  # https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
        (
            "t/purchase.csv.bz2",
            "PurchaseStatus",
            0.94933,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset/data
        (
            "t/recruit.csv.bz2",
            "HiringDecision",
            0.92533,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data
        # ("t/enron.csv.bz2", "label", 0.88551),
    ],
)
def test_datasets(inpt, label, expected):
    """Test well-known datasets."""
    data = ml.read_data(inpt)
    target = data[[label]]
    data.drop(columns=label, inplace=True)

    actual, _ = ml.predict(ml.preprocess(data), target)
    assert round(actual, 5) == round(
        expected, 5
    ), f"actual should match expected for {label}"


def test_read_data():
    """UT coverage for read_data()."""
    assert (
        len(ml.read_data("t/iris.csv.bz2")) == 150
    ), "should read data from compressed CSV"
    assert len(ml.read_data("t/titanic.eml")) == 891, "should read data from EML email"
